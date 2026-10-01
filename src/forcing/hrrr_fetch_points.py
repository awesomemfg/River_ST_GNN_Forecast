"""Pull the archived HRRR forecast history at our rain and wind stations, one forecast cycle at a time.

WHY
---
Dr. Xue asked for objective 3 to be a true forecast: every model driven by the HRRR rainfall forecast
that existed at each issue time, not by observed rain. NOAA publishes every HRRR cycle; the University
of Utah mirrors the surface fields as chunked Zarr on AWS (hrrrzarr), 150 x 150 grid cells per chunk
holding all forecast hours of one cycle. Reading only the chunk that contains our stations makes a
six-month pull feasible.

WHAT
----
For every hourly HRRR cycle from 2025-12-31 12 UTC to 2026-06-30 23 UTC:
  - surface/APCP_1hr_acc_fcst   one-hour precipitation accumulation ending at each forecast hour (mm)
  - 10m_above_ground/UGRD, VGRD 10 m wind components at each forecast hour (m/s)
Hourly cycles carry forecast hours 1 to 18; the 00, 06, 12 and 18 UTC cycles carry 1 to 48.
Each station takes its NEAREST grid cell, which is what Open-Meteo serves (see
Experiments/OPENMETEO_VS_RAW_HRRR_20260807/VERDICT_OPENMETEO_VS_RAW_HRRR.md).

Stations: the 44 rain gauges in the training matrix and the six wind stations, with coordinates parsed
from the production downloader exactly as the earlier Open-Meteo fetchers did.

OUTPUT (resumable, one file per day)
  data/cycles/hrrr_points_YYYYMMDD.npz
      cycle_utc [nc]           cycle initialization times (ISO strings)
      apcp_mm   [nc, 48, 44]   hourly accumulation ending at forecast hour k+1 (NaN beyond the cycle's reach)
      ugrd_ms   [nc, 48, 6]
      vgrd_ms   [nc, 48, 6]
  data/station_grid_cells.csv   station -> lat, lon, grid row/col, cell lat/lon, distance, chunk
  data/fetch_manifest.csv       one row per cycle: status, forecast hours, bytes

Run (<SERVER>, niced, resumable):
  nohup nice -n 10 conda run -n operational python -u hrrr_fetch_points.py > ../logs/01_fetch.log 2>&1 &
"""
import ast
import concurrent.futures
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from numcodecs import Blosc

EXP = "project/Experiments/HRRR_FORCING_AND_LSTM_20260911"
DATA = os.path.join(EXP, "data")
CYCLE_DIR = os.path.join(DATA, "cycles")
os.makedirs(CYCLE_DIR, exist_ok=True)

DOWNLOAD_OPERATIONAL_PY = "project/Inference/Download_Operational.py"
BASE = "https://hrrrzarr.s3.amazonaws.com"
CYCLE_START = pd.Timestamp("2025-12-31 12:00", tz="UTC")
# Extended on 2026-09-18 from 2026-06-30 to cover the January to August hindcast. The fetch is
# resumable and skips any day already written, so the January to June cycle files are untouched.
CYCLE_END = pd.Timestamp("2026-08-31 23:00", tz="UTC")
MAX_FXX = 48
CHUNK = 150
WORKERS = 12
RETRIES = 5
TIMEOUT = 60

# The 44 rain gauges of the training matrix (columns <code>_rain_in), in the matrix's sorted order.
RAIN_CODES = [
    "BCRA1299", "BCRA2400", "BCRA6146", "BCRU0001", "BCRU0003", "BCRU4350", "BCRU4420",
    "BMRA0347", "BMRA1292", "BMRA3548", "BMRU0004", "BMRU0005", "BMRU0006", "BMRU0008",
    "BMRU0009", "BMRU0581", "BMRU2210", "BMRU2620", "HBRA1384", "HBRA2096", "HBRU0001",
    "HBRU0398", "HBRU3213", "HBRU4751", "MBRA3301", "MBRA4314", "MBRA4413", "MBRA4569",
    "MBRA4570", "MBRA5275", "MBRA5715", "MBRA5881", "MBRA6410", "MBRA6512", "MBRA6561",
    "MBRA7806", "MBRA8344", "MBRA8949", "MBRA9909", "MBRU3561", "MBRU4314", "MBRU5688",
    "MBRU6557", "MBRU8999",
]
WIND_CODES = ["BCWU0001", "BCWU0002", "BCWU0003", "BCWU0005", "BCWU0006", "BMWU0004"]

VARIABLES = [
    ("apcp", "surface", "APCP_1hr_acc_fcst", "rain"),
    ("ugrd", "10m_above_ground", "UGRD", "wind"),
    ("vgrd", "10m_above_ground", "VGRD", "wind"),
]

BLOSC = Blosc()


def log(*parts):
    print("[hrrr]", *parts, flush=True)


def http_get(url):
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code in (403, 404):
                return None
            if attempt == RETRIES:
                raise
        except Exception:
            if attempt == RETRIES:
                raise
        time.sleep(2.0 * attempt)
    return None


def decode_array(raw, dtype):
    return np.frombuffer(BLOSC.decode(raw), dtype=dtype)


def resolve_coords(code, node_coords):
    if code in node_coords:
        return node_coords[code], "direct"
    for alias in (code.replace("RA", "SA", 1), code.replace("RU", "SU", 1)):
        if alias in node_coords:
            return node_coords[alias], "alias:" + alias
    prefix_matches = [key for key in node_coords if key[:4] == code[:4]]
    if prefix_matches:
        return node_coords[prefix_matches[0]], "prefix:" + prefix_matches[0]
    return None, "missing"


def load_grid_latlon():
    """Full HRRR latitude and longitude arrays [1059, 1799] from the hrrrzarr chunk index."""
    out = {}
    for name in ("latitude", "longitude"):
        meta = json.loads(http_get(f"{BASE}/grid/HRRR_chunk_index.zarr/{name}/.zarray"))
        ny, nx = meta["shape"]
        cy, cx = meta["chunks"]
        arr = np.full((ny, nx), np.nan)
        for iy in range(int(np.ceil(ny / cy))):
            for ix in range(int(np.ceil(nx / cx))):
                raw = http_get(f"{BASE}/grid/HRRR_chunk_index.zarr/{name}/{iy}.{ix}")
                block = decode_array(raw, meta["dtype"]).reshape(cy, cx)
                y1 = min((iy + 1) * cy, ny)
                x1 = min((ix + 1) * cx, nx)
                arr[iy * cy:y1, ix * cx:x1] = block[:y1 - iy * cy, :x1 - ix * cx]
        out[name] = arr
    return out["latitude"], out["longitude"]


def station_cells():
    src = open(DOWNLOAD_OPERATIONAL_PY).read()
    match = re.search(r"HARDCODED_NODE_COORDS\s*=\s*(\{.*?\n\})", src, re.S)
    if match is None:
        raise SystemExit("[FATAL] HARDCODED_NODE_COORDS not found")
    node_coords = ast.literal_eval(match.group(1))
    lat_grid, lon_grid = load_grid_latlon()
    if not np.nanmean(lat_grid[0]) < np.nanmean(lat_grid[-1]):
        raise SystemExit("[FATAL] grid row 0 is not the southern edge; orientation assumption broken")
    rows = []
    for kind, codes in (("rain", RAIN_CODES), ("wind", WIND_CODES)):
        for code in codes:
            xy, how = resolve_coords(code, node_coords)
            if xy is None:
                raise SystemExit(f"[FATAL] no coordinates for {code}")
            lat, lon = (xy["lat"], xy["lon"]) if isinstance(xy, dict) else (float(xy[0]), float(xy[1]))
            dlat = np.radians(lat_grid - lat)
            dlon = np.radians(lon_grid - lon)
            a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(np.radians(lat_grid)) * np.sin(dlon / 2) ** 2
            dist_km = 2 * 6371.0 * np.arcsin(np.sqrt(a))
            iy, ix = np.unravel_index(int(np.nanargmin(dist_km)), dist_km.shape)
            rows.append({
                "kind": kind, "code": code, "coord_source": how, "lat": lat, "lon": lon,
                "row": int(iy), "col": int(ix),
                "cell_lat": float(lat_grid[iy, ix]), "cell_lon": float(lon_grid[iy, ix]),
                "distance_km": float(dist_km[iy, ix]),
                "chunk_y": int(iy // CHUNK), "chunk_x": int(ix // CHUNK),
            })
    cells = pd.DataFrame(rows)
    if float(cells["distance_km"].max()) > 2.5:
        raise SystemExit("[FATAL] a station is more than 2.5 km from its nearest 3 km cell")
    cells.to_csv(os.path.join(DATA, "station_grid_cells.csv"), index=False)
    log("station cells written; max distance %.2f km; chunks %s" % (
        cells["distance_km"].max(), sorted(set(zip(cells.chunk_y, cells.chunk_x)))))
    return cells


def fetch_cycle(cycle, cells_by_kind, chunks_by_kind):
    ymd = cycle.strftime("%Y%m%d")
    hh = cycle.strftime("%H")
    root = f"{BASE}/sfc/{ymd}/{ymd}_{hh}z_fcst.zarr"
    result = {}
    nbytes = 0
    nfxx = None
    for key, level, var, kind in VARIABLES:
        cells = cells_by_kind[kind]
        values = np.full((MAX_FXX, len(cells)), np.nan, dtype=np.float32)
        for chunk_y, chunk_x in chunks_by_kind[kind]:
            raw = http_get(f"{root}/{level}/{var}/{level}/{var}/0.{chunk_y}.{chunk_x}")
            if raw is None:
                return None, 0, None
            nbytes += len(raw)
            flat = decode_array(raw, "<f4")
            nt = flat.size // (CHUNK * CHUNK)
            block = flat.reshape(nt, CHUNK, CHUNK)
            block = np.where(block == -9999.0, np.nan, block)
            nfxx = nt if nfxx is None else min(nfxx, nt)
            in_chunk = cells[(cells.chunk_y == chunk_y) & (cells.chunk_x == chunk_x)]
            for position, cell in in_chunk.iterrows():
                values[:nt, position] = block[:, cell.row - chunk_y * CHUNK, cell.col - chunk_x * CHUNK]
        result[key] = values
    return result, nbytes, nfxx


def main():
    cells = station_cells()
    cells_by_kind = {}
    chunks_by_kind = {}
    for kind in ("rain", "wind"):
        sub = cells[cells.kind == kind].reset_index(drop=True)
        cells_by_kind[kind] = sub
        chunks_by_kind[kind] = sorted(set(zip(sub.chunk_y, sub.chunk_x)))

    manifest_path = os.path.join(DATA, "fetch_manifest.csv")
    manifest_rows = []
    if os.path.exists(manifest_path):
        manifest_rows = pd.read_csv(manifest_path).to_dict("records")

    days = pd.date_range(CYCLE_START.normalize(), CYCLE_END.normalize(), freq="D")
    for day in days:
        out_path = os.path.join(CYCLE_DIR, "hrrr_points_" + day.strftime("%Y%m%d") + ".npz")
        if os.path.exists(out_path):
            continue
        cycles = [c for c in pd.date_range(day, day + pd.Timedelta(hours=23), freq="h")
                  if CYCLE_START <= c <= CYCLE_END]
        t_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(fetch_cycle, c, cells_by_kind, chunks_by_kind): c for c in cycles}
            fetched = {}
            for future in concurrent.futures.as_completed(futures):
                fetched[futures[future]] = future.result()
        apcp = np.full((len(cycles), MAX_FXX, len(RAIN_CODES)), np.nan, dtype=np.float32)
        ugrd = np.full((len(cycles), MAX_FXX, len(WIND_CODES)), np.nan, dtype=np.float32)
        vgrd = np.full((len(cycles), MAX_FXX, len(WIND_CODES)), np.nan, dtype=np.float32)
        day_bytes = 0
        for index, cycle in enumerate(cycles):
            values, nbytes, nfxx = fetched[cycle]
            status = "ok" if values is not None else "missing"
            if values is not None:
                apcp[index] = values["apcp"]
                ugrd[index] = values["ugrd"]
                vgrd[index] = values["vgrd"]
            day_bytes += nbytes
            manifest_rows.append({"cycle_utc": cycle.isoformat(), "status": status,
                                  "forecast_hours": nfxx if nfxx else 0, "bytes": nbytes})
        np.savez_compressed(out_path, cycle_utc=np.asarray([c.isoformat() for c in cycles]),
                            apcp_mm=apcp, ugrd_ms=ugrd, vgrd_ms=vgrd,
                            rain_codes=np.asarray(RAIN_CODES), wind_codes=np.asarray(WIND_CODES))
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        n_missing = sum(1 for c in cycles if fetched[c][0] is None)
        log(day.strftime("%Y-%m-%d"), f"{len(cycles)} cycles, missing {n_missing}, "
            f"{day_bytes / 1e6:.1f} MB, {time.time() - t_start:.0f} s")
    manifest = pd.read_csv(manifest_path)
    log("DONE: cycles", len(manifest), "ok", int((manifest.status == "ok").sum()),
        "missing", int((manifest.status != "ok").sum()))
    print("HRRR_FETCH_DONE", flush=True)


main()
