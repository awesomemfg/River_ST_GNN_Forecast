"""Shared basemap for the paper maps — REPLICATES figure_01 VERBATIM:
HEC-RAS EBR geometry polylines as the real river/drainage linework, NAD83 Louisiana South StatePlane
projection (feet), Okabe-Ito basin palette, scale bar + north arrow. Unified paper style (Nimbus serif).
Source recipe: paper_studyarea_ebr_river_context.py (the figure_01 generator)."""
import os
import sys
import json
import numpy as np
import h5py
import geopandas as gpd
from pyproj import CRS, Transformer
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(__file__))
from _card import apply as _apply
_apply()
import matplotlib.pyplot as plt

P23 = "project/manuscript_early/analysis"
PB = "project/manuscript"
LA_PARISHES = PB + "/analysis/la_parishes.geojson"
ASC = P23 + "/ascension_parish.geojson"
STATION_META = P23 + "/station_meta.json"
GRAPH_NPZ = "project/Inference/GNN_struct_model/gnn_graph.npz"   # production 68-node graph (MBSA4570 removed 2026-07-08; identical 497 edges)
PRJ = "project/HECRAS_Files/EBR/_Projection/102682_NAD 1983 StatePlane Louisiana South FIPS 1702 Feet.prj"
EBR = "project/HECRAS_Files/EBR/EBR-APG_Regional.g33.hdf"

# Okabe-Ito palette — identical to figure_01 (paper_studyarea_ebr_river_context.py)
BASIN_COL = {"BC": "#0072B2", "BM": "#009E73", "MB": "#D55E00", "HB": "#CC79A7"}
BASIN_NAME = {"BC": "Bayou Conway", "BM": "Bayou Manchac", "MB": "Marvin Braud", "HB": "Henderson Bayou"}

_TXT = open(PRJ, "r", encoding="utf-8").read()
HECRAS_CRS = CRS.from_wkt(_TXT)
LONLAT = CRS.from_epsg(4326)
_TF = Transformer.from_crs(LONLAT, HECRAS_CRS, always_xy=True)
_RIVERS = None


def basin(g):
    return g[:2]


def source(g):
    return g[3]


def is_external(g):
    return g[4:6] == "00"


def to_xy(lon, lat):
    return _TF.transform(lon, lat)


def load_coords_lonlat():
    met = json.load(open(STATION_META))["STATION_METADATA"]
    return {g: (float(v[1]), float(v[0])) for g, v in met.items()}      # gauge -> (lon, lat)


def load_xy():
    return {g: to_xy(lon, lat) for g, (lon, lat) in load_coords_lonlat().items()}   # gauge -> (E, N) ft


def load_graph():
    g = np.load(GRAPH_NPZ, allow_pickle=True)
    return [str(x) for x in g["nodes"]], g["A_norm"].astype("float32")


def _load_rivers():
    """EBR HEC-RAS geometry polylines (already in StatePlane feet)."""
    global _RIVERS
    if _RIVERS is None:
        segs = []
        with h5py.File(EBR, "r") as h:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and name.endswith("Polyline Points"):
                    grp = name.rsplit("/", 1)[0]
                    info = grp + "/Polyline Info"
                    if info in h:
                        vals = obj[:]
                        pinfo = h[info][:]
                        for r in range(pinfo.shape[0]):
                            s = int(pinfo[r, 0])
                            n = int(pinfo[r, 1])
                            if n >= 2:
                                segs.append(vals[s:s + n, :2])
            h.visititems(visit)
        _RIVERS = segs
    return _RIVERS


# Reference places (lon, lat) for regional context labelling.
CITIES = {
    "Baton Rouge": (-91.1871, 30.4515),
    "Gonzales": (-90.9201, 30.2383),
    "Donaldsonville": (-90.9901, 30.1010),
    "New Orleans": (-90.0715, 29.9511),
}


def add_north(ax, x=0.05, y=0.955, size=0.135):
    """Bold, highly-visible north arrow (axes fraction)."""
    ax.annotate("N", xy=(x, y), xytext=(x, y - size), xycoords="axes fraction", ha="center",
                fontsize=19, fontweight="bold", zorder=30,
                arrowprops={"facecolor": "black", "edgecolor": "black", "width": 6.0,
                            "headwidth": 16.0, "headlength": 13.0})


def add_cities(ax, names, dot=34, fs=11, dy=0.0):
    """Plot city markers + bold labels (names selected from CITIES)."""
    for nm in names:
        lon, lat = CITIES[nm]
        x, y = to_xy(lon, lat)
        ax.scatter([x], [y], s=dot, marker="s", c="black", zorder=18)
        ax.annotate(nm, (x, y), xytext=(6, 5 + dy), textcoords="offset points", fontsize=fs,
                    fontweight="bold", color="#1a1a1a", zorder=19,
                    path_effects=None)


def add_river_label(ax, text, lon, lat, rot=0.0, fs=11.5, color="#13627f"):
    """Italic river label at a geographic point."""
    x, y = to_xy(lon, lat)
    ax.text(x, y, text, fontsize=fs, fontstyle="italic", color=color, rotation=rot,
            ha="center", va="center", zorder=17, fontweight="bold")


# Natural Earth water (ocean/estuary polygon + lakes + river centrelines incl. the Mississippi),
# reprojected to StatePlane. Lake Pontchartrain is a brackish estuary and lives in the OCEAN layer, not lakes.
_NE = {"lakes": None, "rivers": None, "ocean": None}
_NE_BBOX = (-92.2, 28.8, -88.8, 31.3)   # SE-Louisiana clip (lon/lat) covering every panel extent
WATER_FILL = "#cfe3f2"
WATER_EDGE = "#8fb8d6"
MISS_COL = "#3a86b5"
_NE_KEY = {"lakes": "lakes", "rivers_lake_centerlines": "rivers", "ocean": "ocean"}


def _load_ne(name):
    """Cached Natural Earth physical layer, clipped to the SE-LA bbox and reprojected to StatePlane.
    The ocean polygon is hard-clipped to the bbox (it is global) so only the regional water is kept."""
    key = _NE_KEY[name]
    if _NE[key] is None:
        from cartopy.io import shapereader as sr
        from shapely.geometry import box
        bb = box(*_NE_BBOX)
        g = gpd.read_file(sr.natural_earth(resolution="10m", category="physical", name=name))
        if g.crs is None:
            g = g.set_crs(LONLAT)
        g = gpd.clip(g, bb) if name == "ocean" else g[g.intersects(bb)].copy()
        _NE[key] = g.to_crs(HECRAS_CRS)
    return _NE[key]


def add_water(ax, lakes=True, rivers=True, ocean=True, fill=True, zorder=2.6):
    """Draw Natural Earth water on a StatePlane matplotlib axis: the ocean/estuary polygon (Lake
    Pontchartrain, Lake Borgne, the Gulf), inland lakes (Maurepas, Salvador), and river centrelines
    (Mississippi drawn heavier, plus Bayou Lafourche / Pearl)."""
    if ocean:
        oc = _load_ne("ocean")
        if len(oc):
            oc.plot(ax=ax, facecolor=(WATER_FILL if fill else "none"), edgecolor=WATER_EDGE,
                    linewidth=0.7, alpha=(0.9 if fill else 1.0), zorder=zorder - 0.1)
    if lakes:
        lk = _load_ne("lakes")
        if len(lk):
            lk.plot(ax=ax, facecolor=(WATER_FILL if fill else "none"), edgecolor=WATER_EDGE,
                    linewidth=0.8, alpha=(0.9 if fill else 1.0), zorder=zorder)
    if rivers:
        rv = _load_ne("rivers_lake_centerlines")
        if len(rv):
            ncol = next((c for c in rv.columns if c.lower() in ("name", "name_en")), None)
            rv.plot(ax=ax, color=MISS_COL, linewidth=1.0, alpha=0.85, zorder=zorder + 0.3)
            if ncol is not None:
                miss = rv[rv[ncol].astype(str).str.contains("Mississippi", case=False, na=False)]
                if len(miss):
                    miss.plot(ax=ax, color=MISS_COL, linewidth=1.9, alpha=0.95, zorder=zorder + 0.4)


def make_basemap(figsize, gauges_for_extent, title=None, pad_ft=12000.0,
                 river_alpha=0.42, river_lw=0.42, ax=None, extent=None, scalebar=True, axis_labels=True,
                 water=True):
    """Return (fig, ax) in StatePlane with EBR rivers, parishes, parish boundary, scale bar, N arrow.
    river_alpha/river_lw keep the HEC-RAS linework faint so it does not fight the markers/edges.
    Pass an existing `ax` to draw the basemap into a subplot (returns that ax's figure).
    extent=(xmin,xmax,ymin,ymax) overrides the auto extent (for regional/parish panels)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    la = gpd.read_file(LA_PARISHES)
    asc = gpd.read_file(ASC)
    if la.crs is None:
        la = la.set_crs(LONLAT)
    if asc.crs is None:
        asc = asc.set_crs(LONLAT)
    la = la.to_crs(HECRAS_CRS)
    asc = asc.to_crs(HECRAS_CRS)
    ax.set_facecolor("#f6f4ed")
    la.plot(ax=ax, facecolor="#e9e6dc", edgecolor="#b3b3b3", linewidth=0.48, zorder=1)
    asc.plot(ax=ax, facecolor="#fffdf7", edgecolor="black", linewidth=2.0, alpha=0.78, zorder=2)
    if water:
        add_water(ax, zorder=2.6)
    ax.add_collection(LineCollection(_load_rivers(), colors="#1b8bb5", linewidths=river_lw, alpha=river_alpha, zorder=3))

    if extent is not None:
        xmin, xmax, ymin, ymax = extent
    else:
        xy = load_xy()
        xs = [xy[g][0] for g in gauges_for_extent if g in xy]
        ys = [xy[g][1] for g in gauges_for_extent if g in xy]
        ab = asc.total_bounds
        xmin = min(float(ab[0]), min(xs)) - pad_ft
        xmax = max(float(ab[2]), max(xs)) + pad_ft
        ymin = min(float(ab[1]), min(ys)) - pad_ft
        ymax = max(float(ab[3]), max(ys)) + pad_ft
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    if axis_labels:
        ax.set_xlabel("Easting (NAD83 Louisiana South StatePlane, US ft)", fontsize=12)
        ax.set_ylabel("Northing (US ft)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.grid(True, color="#b8b8b8", linewidth=0.32, linestyle="--", alpha=0.42, zorder=0)
    add_north(ax)
    # scale bar (10 km = 32808.4 ft)
    if scalebar:
        L = 32808.4
        sx = xmin + 0.058 * (xmax - xmin)
        sy = ymin + 0.05 * (ymax - ymin)
        ax.plot([sx, sx + L], [sy, sy], color="black", linewidth=3.5, solid_capstyle="butt", zorder=10)
        ax.text(sx + L / 2.0, sy + 0.012 * (ymax - ymin), "10 km", ha="center", va="bottom", fontsize=11, zorder=10)
    if title:
        ax.set_title(title)
    return fig, ax
