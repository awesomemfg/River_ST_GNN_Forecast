#!/usr/bin/env python3
"""
Fetch the external layers needed for the rebuilt Figure 1 (George's comment: land use,
topography, and where the bayous actually are).

Downloads, into .../Fig1_Rebuild/data/ :
  1. nlcd_ascension.tif        - NLCD land cover clipped to the study bbox (MRLC WCS)
  2. nhd_named_flowlines.gpkg  - NHD named flowlines (the bayous) for the study bbox
                                 (USGS National Map hydro REST service)
  3. nhd_waterbody.gpkg        - NHD waterbodies for the same bbox

Everything is written in EPSG:4326 except the NLCD raster, which is kept in its native
Albers (EPSG:5070) and reprojected at plot time.

Run inside the `operational` conda env:
    conda run -n operational python fig01_fetch_map_layers.py
"""

import os
import sys
import time

import geopandas as gpd
import requests
from shapely.geometry import shape

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
os.makedirs(DATA, exist_ok=True)

# Study bounding box in lon/lat: the parish plus every supporting graph gauge, with padding.
# parish bounds  = (-91.1066, 30.0627, -90.6319, 30.3470)
# graph gauges   = (-91.1916, 30.0272, -90.0342, 30.5349)
BBOX_LONLAT = (-91.32, 29.95, -89.92, 30.62)

NLCD_WCS = "https://www.mrlc.gov/geoserver/mrlc_download/wcs"
NHD_REST = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer"

# NHD MapServer layer ids
NHD_FLOWLINE_LAYER = 6
NHD_WATERBODY_LAYER = 10


def log(*a):
    print("[fetch]", *a, flush=True)


# --------------------------------------------------------------------------------------
# 1. NLCD land cover via the MRLC WCS
# --------------------------------------------------------------------------------------
def fetch_nlcd(coverage="NLCD_2021_Land_Cover_L48", out_name="nlcd_ascension.tif"):
    """Pull an NLCD subset for BBOX_LONLAT from the MRLC WCS in EPSG:4326."""
    out = os.path.join(DATA, out_name)
    if os.path.exists(out) and os.path.getsize(out) > 10000:
        log("NLCD already present:", out)
        return out

    west, south, east, north = BBOX_LONLAT
    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageId": coverage,
        "format": "image/geotiff",
        # WCS 2.0.1 axis order for EPSG:4326 subsets on this server is Long/Lat
        "subset": [f"Long({west},{east})", f"Lat({south},{north})"],
        "subsettingCRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
        "outputCRS": "http://www.opengis.net/def/crs/EPSG/0/4326",
    }
    log("requesting NLCD coverage:", coverage)
    r = requests.get(NLCD_WCS, params=params, timeout=300)
    log("  status", r.status_code, "bytes", len(r.content),
        "ctype", r.headers.get("Content-Type"))
    if r.status_code != 200 or len(r.content) < 10000:
        log("  FAILED. First 600 bytes of response:")
        log(r.content[:600].decode("utf-8", "replace"))
        return None
    with open(out, "wb") as f:
        f.write(r.content)
    log("  saved", out)
    return out


# --------------------------------------------------------------------------------------
# 2 + 3. NHD features via the National Map REST service
# --------------------------------------------------------------------------------------
def fetch_nhd_layer(layer_id, out_name, where="1=1", page=1000):
    """Page through an NHD MapServer layer inside BBOX_LONLAT and save as GeoPackage."""
    out = os.path.join(DATA, out_name)
    if os.path.exists(out):
        log("NHD layer already present:", out)
        return out

    west, south, east, north = BBOX_LONLAT
    url = f"{NHD_REST}/{layer_id}/query"
    feats = []
    offset = 0
    while True:
        params = {
            "where": where,
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "outSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": page,
        }
        r = requests.get(url, params=params, timeout=180)
        if r.status_code != 200:
            log(f"  layer {layer_id} HTTP {r.status_code}")
            log(r.text[:500])
            break
        js = r.json()
        got = js.get("features", [])
        feats.extend(got)
        log(f"  layer {layer_id}: +{len(got)} (total {len(feats)})")
        if len(got) < page:
            break
        offset += page
        time.sleep(0.3)

    if not feats:
        log(f"  layer {layer_id}: NO FEATURES")
        return None

    rows = []
    geoms = []
    for f in feats:
        if not f.get("geometry"):
            continue
        rows.append(f.get("properties", {}))
        geoms.append(shape(f["geometry"]))
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    gdf.to_file(out, driver="GPKG")
    log("  saved", out, f"({len(gdf)} features)")
    return out


def main():
    log("data dir:", DATA)
    log("bbox lon/lat:", BBOX_LONLAT)

    nlcd = fetch_nlcd()
    if nlcd is None:
        log("NLCD via WCS failed - will need the fallback path (see report).")

    # Named flowlines only: that is what carries the bayou names George asked for.
    fetch_nhd_layer(NHD_FLOWLINE_LAYER, "nhd_named_flowlines.gpkg",
                    where="GNIS_NAME IS NOT NULL")
    fetch_nhd_layer(NHD_WATERBODY_LAYER, "nhd_waterbody.gpkg")

    log("done. contents of", DATA)
    for n in sorted(os.listdir(DATA)):
        p = os.path.join(DATA, n)
        log(f"   {n}  {os.path.getsize(p)/1e6:.2f} MB")


if __name__ == "__main__":
    sys.exit(main())
