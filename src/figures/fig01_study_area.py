#!/usr/bin/env python3
"""
Rebuilt Figure 1 for the ST-GNN HESS paper, answering George Xue's two comments:

  C25  "could use a figure to illustrate 1) land use and 2) topography"
  C103 "but there is no map showing where those bayou exactly are, maybe blend that
        in Fig 1 I suggested? Bathymetry with land use etc.?"

Four panels, all on the SAME native recipe as the existing Figure 1
(`figure_library/fig01_study_area_all_gauges.py` + `figure_library/_mapbase.py`) --
NAD83 Louisiana South StatePlane (US ft), Okabe-Ito basin palette, HEC-RAS EBR
linework, 10 km scale bar, north arrow. Nothing from the original recipe is dropped.

  (a) Regional context   - southeast Louisiana, Ascension Parish in red, the lakes,
                           the Mississippi, Baton Rouge / New Orleans.
  (b) Land cover         - NLCD 2021, official NLCD legend colours.
  (c) Topography and     - the HEC-RAS EBR topo-bathymetric DEM, with the named
      bathymetry           bayous/canals drawn and labelled on top.
  (d) Monitoring network - all 68 ST-GNN graph gauges, coloured by drainage area
                           and shaped by operator, over the drainage linework.

Run inside the `operational` conda env (it has geopandas + rasterio):
    conda run -n operational python fig01_rebuilt_landuse_topo_bayous.py
"""

import os
import sys

import geopandas as gpd
import numpy as np
import rasterio
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter, MaxNLocator
from rasterio.enums import Resampling
from rasterio.warp import Resampling as WarpResampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import from_bounds
from shapely.geometry import box

# Reuse the paper's own map machinery, verbatim.
PAPER = "project/manuscript"
sys.path.insert(0, "src/figures")
from _mapbase import (  # noqa: E402
    ASC, BASIN_COL, BASIN_NAME, CITIES, HECRAS_CRS, LA_PARISHES, LONLAT,
    _load_rivers, add_north, add_river_label, add_water, basin,
    is_external, load_graph, load_xy, source, to_xy,
)
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = ("project/"
        "manuscript/20260811_Revision/"
        "Fig1_Rebuild/data")
FIGS = ("project/"
        "manuscript/20260826_Final_V2/"
        "output/figures")
os.makedirs(FIGS, exist_ok=True)

OUT = os.path.join(FIGS, "fig01")

DEM_TIF = ("project/HECRAS_Files/New_Files/"
           "EBR_Regional_wBathymetry_v4_CRD.EBR_Regional_wBathymetry_v4.Resampled.tif")
NLCD_TIF = os.path.join(DATA, "nlcd_ascension.tif")
NHD_FLOW = os.path.join(DATA, "nhd_named_flowlines.gpkg")

FT2M = 0.3048

# ---------------------------------------------------------------------------------
# NLCD 2021 legend (official MRLC colours). Only classes that occur are drawn.
# ---------------------------------------------------------------------------------
NLCD_CLASSES = [
    (11, "#466B9F", "Open water"),
    (21, "#DEC5C5", "Developed, open space"),
    (22, "#D99282", "Developed, low intensity"),
    (23, "#EB0000", "Developed, medium intensity"),
    (24, "#AB0000", "Developed, high intensity"),
    (31, "#B3AC9F", "Barren land"),
    (41, "#68AB5F", "Deciduous forest"),
    (42, "#1C5F2C", "Evergreen forest"),
    (43, "#B5C58F", "Mixed forest"),
    (52, "#CCB879", "Shrub / scrub"),
    (71, "#DFDFC2", "Herbaceous"),
    (81, "#DCD939", "Hay / pasture"),
    (82, "#AB6C28", "Cultivated crops"),
    (90, "#B8D9EB", "Woody wetlands"),
    (95, "#6C9FB8", "Emergent herbaceous wetlands"),
]

# Named waterways to draw + label. These are the four drainage areas the paper names
# (Bayou Conway, Bayou Manchac, Marvin Braud/New River, Henderson Bayou) plus the
# receiving waters and the largest in-parish channels.
# Farid, 2026-08-12: label ONLY the four channels that name the four operational drainage
# areas. No Amite River, no Muddy Creek, no other named waterways. The point of the panel is
# to show where the four bayous the paper talks about actually are, not to inventory the
# hydrography.
BAYOUS_LABELLED = [
    "Bayou Conway",
    "Bayou Manchac",
    "Henderson Bayou",
    "New River",
]

# The four principal channels that give the operational drainage areas their names, or
# that the area drains to. Drawn heavier so the reader can find them, but deliberately NOT
# colour-coded by drainage area: APG's own station metadata
# (HECRAS_Files/SCADAItemsLatLong (1).xlsx, columns Basin x Waterway) shows the channels do
# NOT map one-to-one onto the areas -- New River carries 23 Bayou Conway items AND 15 Marvin
# Braud items, and Black Bayou / Smith Bayou / Bayou Francois are all Marvin Braud. Drainage-
# area identity is therefore carried by the GAUGE colours (from the station-code prefix),
# which is authoritative, not by the channel colours.
PRINCIPAL_CHANNELS = {"Bayou Conway", "Bayou Manchac", "Henderson Bayou", "New River"}
CHANNEL_MAIN = "#0b3d5c"
CHANNEL_OTHER = "#2f7fa8"

# Manual label anchors (lon, lat, rotation) where the automatic mid-point placement
# collides or falls outside the panel. Anything not listed is auto-placed.
LABEL_OVERRIDE = {
    "Bayou Conway":    (-90.995, 30.145, -32),
    "Bayou Manchac":   (-91.010, 30.310, -8),
    "Henderson Bayou": (-90.855, 30.295, 28),
    "New River":       (-90.850, 30.160, -58),
}


def log(*a):
    print("[fig1]", *a, flush=True)


# ---------------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------------
def load_frames():
    la = gpd.read_file(LA_PARISHES)
    asc = gpd.read_file(ASC)
    if la.crs is None:
        la = la.set_crs(LONLAT)
    if asc.crs is None:
        asc = asc.set_crs(LONLAT)
    return la, asc


def read_dem(bounds_sp):
    """Read the topo-bathy DEM for a StatePlane bbox, downsampled, returned in metres."""
    xmin, xmax, ymin, ymax = bounds_sp
    with rasterio.open(DEM_TIF) as r:
        win = from_bounds(xmin, ymin, xmax, ymax, r.transform)
        scale = max(1, int(max(win.width, win.height) // 1600))
        oh = max(1, int(win.height // scale))
        ow = max(1, int(win.width // scale))
        arr = r.read(1, window=win, out_shape=(oh, ow),
                     resampling=Resampling.average).astype("float32")
        nd = r.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
    arr[arr < -1e4] = np.nan
    arr = arr * FT2M
    log(f"DEM window {ow}x{oh}, elev {np.nanmin(arr):.1f} to {np.nanmax(arr):.1f} m")
    return arr


def read_nlcd_statplane(bounds_sp):
    """Reproject the NLCD subset from EPSG:4326 into StatePlane over the given bbox."""
    xmin, xmax, ymin, ymax = bounds_sp
    with rasterio.open(NLCD_TIF) as src:
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src.crs, HECRAS_CRS, src.width, src.height, *src.bounds)
        dst = np.zeros((dst_h, dst_w), dtype="uint8")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=HECRAS_CRS,
            resampling=WarpResampling.nearest,
        )
    left = dst_transform.c
    top = dst_transform.f
    xres = dst_transform.a
    yres = -dst_transform.e
    right = left + dst_w * xres
    bottom = top - dst_h * yres
    log(f"NLCD reprojected to StatePlane: {dst_w}x{dst_h}")
    return dst, (left, right, bottom, top)


def load_named_flowlines():
    fl = gpd.read_file(NHD_FLOW)
    fl = fl[fl["gnis_name"].notna()].copy()
    return fl.to_crs(HECRAS_CRS)


# ---------------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------------
def draw_station(ax, xy, gauge, size=105, zbase=11):
    x, y = xy[gauge]
    if is_external(gauge):
        ax.scatter(x, y, marker="x", s=size, c="0.20", linewidths=2.0, zorder=zbase)
    else:
        marker = "^" if source(gauge) == "U" else "o"
        ax.scatter(x, y, marker=marker, s=size,
                   c=BASIN_COL.get(basin(gauge), "#555"),
                   edgecolors="black", linewidths=0.95, zorder=zbase + 1)


def draw_bayous(ax, flow, extent_sp, label=True, base_color="#1f6f9c", lw=1.0):
    """Draw + label the named waterways inside the panel extent."""
    xmin, xmax, ymin, ymax = extent_sp
    panel = box(xmin, ymin, xmax, ymax)
    for name in BAYOUS_LABELLED:
        sub = flow[flow["gnis_name"] == name]
        if not len(sub):
            continue
        sub = sub[sub.intersects(panel)]
        if not len(sub):
            continue
        principal = name in PRINCIPAL_CHANNELS
        col = CHANNEL_MAIN if principal else base_color
        width = (lw + 1.90) if principal else lw
        sub.plot(ax=ax, color=col, linewidth=width,
                 alpha=0.95 if principal else 0.75, zorder=6)
        if not label:
            continue
        if name in LABEL_OVERRIDE:
            lon, lat, rot = LABEL_OVERRIDE[name]
            lx, ly = to_xy(lon, lat)
        else:
            merged = sub.geometry.union_all()
            pt = merged.representative_point()
            lx, ly = pt.x, pt.y
            rot = 0
        if not (xmin <= lx <= xmax and ymin <= ly <= ymax):
            continue
        ax.text(lx, ly, name, fontsize=12.0 if principal else 10.5, fontstyle="italic",
                fontweight="bold" if principal else "normal",
                color=CHANNEL_MAIN if principal else "#12496b",
                rotation=rot, ha="center", va="center", zorder=20,
                path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])


def add_scalebar(ax, extent_sp, km=10, y_frac=0.055, x_frac=0.060):
    xmin, xmax, ymin, ymax = extent_sp
    length = km * 3280.84
    sx = xmin + x_frac * (xmax - xmin)
    sy = ymin + y_frac * (ymax - ymin)
    ax.plot([sx, sx + length], [sy, sy], color="black", linewidth=4.0,
            solid_capstyle="butt", zorder=25)
    ax.text(sx + length / 2.0, sy + 0.013 * (ymax - ymin), f"{km} km",
            ha="center", va="bottom", fontsize=12.5, zorder=25,
              path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])


def format_million_feet(value, _position):
    return f"{value / 1_000_000.0:.2f}"


def add_city_labels(ax, city_specs, dot=32, fontsize=10.5):
    for city_name, x_offset, y_offset in city_specs:
        longitude, latitude = CITIES[city_name]
        x_coordinate, y_coordinate = to_xy(longitude, latitude)
        ax.scatter(
            [x_coordinate],
            [y_coordinate],
            s=dot,
            marker="s",
            c="black",
            zorder=18,
        )
        ax.annotate(
            city_name,
            (x_coordinate, y_coordinate),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=fontsize,
            fontweight="bold",
            color="#1a1a1a",
            zorder=19,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
        )


def style_sp_axis(ax, extent_sp, xlabel=True, ylabel=True):
    xmin, xmax, ymin, ymax = extent_sp
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax.xaxis.set_major_formatter(FuncFormatter(format_million_feet))
    ax.yaxis.set_major_formatter(FuncFormatter(format_million_feet))
    ax.tick_params(
        axis="x",
        bottom=xlabel,
        labelbottom=xlabel,
        labelsize=11.5,
    )
    ax.tick_params(
        axis="y",
        left=ylabel,
        labelleft=ylabel,
        labelsize=11.5,
    )
    ax.set_xlabel("Easting (million US ft)" if xlabel else "", fontsize=12.5)
    ax.set_ylabel("Northing (million US ft)" if ylabel else "", fontsize=12.5)
    ax.grid(True, color="#b8b8b8", linewidth=0.3, linestyle="--", alpha=0.40, zorder=0)


# ---------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------
def main():
    la, asc = load_frames()
    la_ll = la.to_crs(LONLAT)
    asc_ll = asc.to_crs(LONLAT)
    la_sp = la.to_crs(HECRAS_CRS)
    asc_sp = asc.to_crs(HECRAS_CRS)
    rivers = _load_rivers()

    xy = load_xy()
    graph_nodes, _ = load_graph()
    have = [g for g in graph_nodes if g in xy]
    log(f"graph nodes {len(graph_nodes)}, mapped {len(have)}")

    flow = load_named_flowlines()

    # --- extents -------------------------------------------------------------
    # Parish-focused extent for the land-cover and topography panels.
    ab = asc_sp.total_bounds
    pad_p = 9000.0
    ext_parish = (ab[0] - pad_p, ab[2] + pad_p, ab[1] - pad_p, ab[3] + pad_p)

    # Full network extent for the monitoring-network panel.
    city_xy = [to_xy(*CITIES[n]) for n in ["Baton Rouge", "Gonzales", "Donaldsonville"]]
    xs = [xy[g][0] for g in have] + [p[0] for p in city_xy]
    ys = [xy[g][1] for g in have] + [p[1] for p in city_xy]
    pad_n = 13000.0
    ext_net = (min(xs) - pad_n, max(xs) + pad_n, min(ys) - pad_n, max(ys) + pad_n)

    log("parish extent:", ext_parish)
    log("network extent:", ext_net)

    # --- rasters -------------------------------------------------------------
    dem = read_dem(ext_parish)
    nlcd, nlcd_ext = read_nlcd_statplane(ext_parish)

    # --- figure --------------------------------------------------------------
    fig = plt.figure(figsize=(12.0, 9.0), facecolor="white")
    ax_a = fig.add_axes([0.055, 0.615, 0.405, 0.320])
    ax_b = fig.add_axes([0.535, 0.615, 0.420, 0.320])
    legend_axis_b = fig.add_axes([0.535, 0.505, 0.420, 0.085])
    ax_c = fig.add_axes([0.055, 0.175, 0.390, 0.300])
    colorbar_axis_c = fig.add_axes([0.462, 0.175, 0.014, 0.300])
    legend_axis_c = fig.add_axes([0.055, 0.018, 0.421, 0.108])
    ax_d = fig.add_axes([0.535, 0.175, 0.420, 0.300])
    legend_axis_d = fig.add_axes([0.535, 0.018, 0.420, 0.108])
    legend_axis_b.axis("off")
    legend_axis_c.axis("off")
    legend_axis_d.axis("off")

    # ---------------- (a) regional context, lon/lat --------------------------
    from cartopy.io import shapereader as sr

    def ne_layer(name, bbox, category="physical"):
        g = gpd.read_file(sr.natural_earth(resolution="10m", category=category, name=name))
        if g.crs is None:
            g = g.set_crs(LONLAT)
        return gpd.clip(g.to_crs(LONLAT), bbox)

    ctx_box = box(-92.2, 28.35, -88.6, 30.75)
    ax_a.set_facecolor("#cfe6f2")
    states = ne_layer("admin_1_states_provinces", ctx_box, category="cultural")
    states = states[(states["admin"] == "United States of America") &
                    (states["name"] != "Louisiana")]
    if len(states):
        states.plot(ax=ax_a, facecolor="#f3f0e7", edgecolor="#b1b1b1",
                    linewidth=0.45, zorder=0.7)
    la_ll.plot(ax=ax_a, facecolor="#e8e6dd", edgecolor="#9d9d9d", linewidth=0.42, zorder=1)
    for nm, z in [("ocean", 2), ("lakes", 3)]:
        lay = ne_layer(nm, ctx_box)
        if len(lay):
            lay.plot(ax=ax_a, facecolor="#cfe3f2", edgecolor="#8fb8d6",
                     linewidth=0.55, alpha=0.95, zorder=z)
    riv_ll = ne_layer("rivers_lake_centerlines", ctx_box)
    if len(riv_ll):
        riv_ll.plot(ax=ax_a, color="#3a86b5", linewidth=0.85, alpha=0.85, zorder=4)
    asc_ll.plot(ax=ax_a, facecolor="#d62728", edgecolor="black", linewidth=0.9, zorder=5)
    ax_a.text(-90.90, 30.12, "Ascension\nParish", fontsize=11.5, fontweight="bold",
              color="white", ha="center", va="center", zorder=9,
              path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])
    ax_a.text(-89.94, 30.27, "Lake\nPontchartrain", fontsize=11.0, fontstyle="italic",
              color="#2c6b94", ha="center", va="center", fontweight="bold", zorder=8,
              path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    ax_a.text(-90.43, 30.40, "Lake Maurepas", fontsize=10.0, fontstyle="italic",
              color="#2c6b94", ha="center", va="center", fontweight="bold", zorder=8,
              path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    ax_a.text(-91.43, 30.49, "Baton Rouge", fontsize=11.0, fontweight="bold",
              color="0.15", zorder=8,
              path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    ax_a.text(-90.06, 29.92, "New Orleans", fontsize=11.0, fontweight="bold",
              color="0.15", zorder=8,
              path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    ax_a.text(-89.28, 29.03, "Mississippi\nRiver Delta", fontsize=11.5,
              fontstyle="italic", fontweight="bold", color="#2c6b94",
              ha="center", va="center", zorder=8,
              path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    ax_a.text(-90.30, 28.60, "Gulf of Mexico", fontsize=13.5,
              fontstyle="italic", fontweight="bold", color="#2c6b94",
              ha="center", va="center", zorder=8,
              path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    ax_a.set_xlim(-92.2, -88.6)
    ax_a.set_ylim(28.35, 30.75)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.grid(True, color="#ffffff", linewidth=0.35, alpha=0.65)
    ax_a.set_xlabel("Longitude", fontsize=13.5)
    ax_a.set_ylabel("Latitude", fontsize=13.5)
    ax_a.tick_params(labelsize=12.5)

    # ---------------- (b) land cover -----------------------------------------
    present = sorted(set(np.unique(nlcd).tolist()))
    used = [(c, col, lab) for c, col, lab in NLCD_CLASSES if c in present]
    codes = [c for c, _, _ in used]
    cmap = ListedColormap([col for _, col, _ in used])
    bounds = codes + [max(codes) + 1]
    norm = BoundaryNorm(bounds, cmap.N)
    masked = np.ma.masked_where(~np.isin(nlcd, codes), nlcd)
    log("NLCD classes drawn:", codes)

    ax_b.set_facecolor("#ffffff")
    ax_b.imshow(masked, extent=[nlcd_ext[0], nlcd_ext[1], nlcd_ext[2], nlcd_ext[3]],
                origin="upper", cmap=cmap, norm=norm, interpolation="nearest", alpha=0.82,
                zorder=1)
    asc_sp.boundary.plot(ax=ax_b, color="black", linewidth=2.0, zorder=8)
    draw_bayous(ax_b, flow, ext_parish, label=False, base_color=CHANNEL_OTHER, lw=0.8)
    add_city_labels(
        ax_b,
        [("Gonzales", 7, 7), ("Donaldsonville", 7, -15)],
        dot=34,
        fontsize=10.5,
    )
    style_sp_axis(ax_b, ext_parish, xlabel=False, ylabel=False)
    add_north(ax_b, x=0.055, y=0.955, size=0.11)
    add_scalebar(ax_b, ext_parish)
    lc_handles = [mpatches.Patch(facecolor=col, edgecolor="0.4", linewidth=0.3, label=lab)
                  for _, col, lab in used]
    legend_axis_b.legend(
        handles=lc_handles,
        loc="center",
        fontsize=8.5,
        frameon=False,
        ncol=3,
        labelspacing=0.10,
        handlelength=0.90,
        handletextpad=0.35,
        columnspacing=0.70,
    )
    # Name the study-area boundary once, on the land-cover panel. Anchor it to a point
    # guaranteed to lie inside the parish polygon rather than a hand-typed coordinate.
    rp = asc_sp.geometry.union_all().representative_point()
    plx, ply = rp.x, rp.y - 0.13 * (ext_parish[3] - ext_parish[2])
    ax_b.text(plx, ply, "Ascension Parish", fontsize=11.5, fontweight="bold",
              color="black", ha="center", va="center", zorder=22,
              path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])

    # ---------------- (c) topography + bathymetry + bayous -------------------
    vmin, vmax = np.nanpercentile(dem, 2), np.nanpercentile(dem, 98)
    ax_c.set_facecolor("#f6f4ed")
    im = ax_c.imshow(dem, extent=[ext_parish[0], ext_parish[1], ext_parish[2], ext_parish[3]],
                     origin="upper", cmap="terrain", vmin=vmin, vmax=vmax, alpha=0.84,
                     zorder=1)
    asc_sp.boundary.plot(ax=ax_c, color="black", linewidth=2.0, zorder=8)
    draw_bayous(ax_c, flow, ext_parish, label=True, base_color=CHANNEL_OTHER, lw=1.0)
    for g in have:
        draw_station(ax_c, xy, g, size=80, zbase=11)
    style_sp_axis(ax_c, ext_parish, xlabel=True, ylabel=True)
    add_north(ax_c, x=0.055, y=0.955, size=0.11)
    add_scalebar(ax_c, ext_parish)
    cb = fig.colorbar(im, cax=colorbar_axis_c)
    cb.ax.tick_params(labelsize=11.5)
    cb.set_label("Elevation (m, NAVD88)", fontsize=11.5, labelpad=7.0)
    bayou_handles = [
        Line2D([0], [0], color=CHANNEL_MAIN, linewidth=2.6,
               label="named channel of a drainage area"),
    ]
    bayou_handles.extend(
        Line2D([0], [0], marker="o", color="white", markerfacecolor=BASIN_COL[b],
               markeredgecolor="black", markersize=6.5,
               label=f"gauge: {BASIN_NAME[b]}")
        for b in ["BC", "BM", "MB", "HB"]
    )
    legend_axis_c.text(
        0.5,
        0.92,
        "Elevation source: Ascension Parish Government regional HEC-RAS terrain model",
        ha="center",
        va="top",
        fontsize=9.0,
        color="0.15",
    )
    legend_axis_c.legend(
        handles=bayou_handles,
        loc="lower center",
        bbox_to_anchor=(0.0, 0.0, 1.0, 0.70),
        fontsize=8.8,
        frameon=False,
        ncol=3,
        labelspacing=0.18,
        handlelength=1.35,
        handletextpad=0.35,
        columnspacing=0.75,
    )

    # ---------------- (d) monitoring network ---------------------------------
    ax_d.set_facecolor("#f6f4ed")
    la_sp.plot(ax=ax_d, facecolor="#e9e6dc", edgecolor="#b3b3b3", linewidth=0.48, zorder=1)
    asc_sp.plot(ax=ax_d, facecolor="#fffdf7", edgecolor="black", linewidth=2.0,
                alpha=0.78, zorder=2)
    add_water(ax_d, zorder=2.6)
    ax_d.add_collection(LineCollection(rivers, colors="#1b8bb5", linewidths=0.36,
                                       alpha=0.34, zorder=4))
    for g in have:
        draw_station(ax_d, xy, g, size=105, zbase=11)
    add_river_label(ax_d, "Mississippi River", -91.03, 30.13, rot=-52, fs=11.0)
    add_river_label(ax_d, "Lake Pontchartrain", -90.29, 30.20, rot=0, fs=11.5, color="#2c6b94")
    add_river_label(ax_d, "Lake\nMaurepas", -90.49, 30.26, rot=0, fs=10.5, color="#2c6b94")
    style_sp_axis(ax_d, ext_net, xlabel=True, ylabel=False)
    add_north(ax_d, x=0.045, y=0.95, size=0.11)
    add_scalebar(ax_d, ext_net)
    net_handles = [mpatches.Patch(facecolor=BASIN_COL[b], edgecolor="black",
                                  label=BASIN_NAME[b])
                   for b in ["BC", "BM", "MB", "HB"]]
    net_handles.extend([
        Line2D([0], [0], marker="o", color="white", markerfacecolor="0.72",
               markeredgecolor="black", markersize=7.5, label="APG in-house gauge"),
        Line2D([0], [0], marker="^", color="white", markerfacecolor="0.72",
               markeredgecolor="black", markersize=7.5, label="USGS in-parish gauge"),
        Line2D([0], [0], marker="x", color="0.25", linewidth=0, markersize=7.0,
               markeredgewidth=1.3, label="supporting gauge (out of parish)"),
    ])
    legend_axis_d.legend(
        handles=net_handles,
        loc="center",
        frameon=False,
        ncol=2,
        title="Drainage area and gauge type",
        title_fontsize=10.0,
        labelspacing=0.18,
        handlelength=1.15,
        handletextpad=0.35,
        columnspacing=0.80,
        fontsize=8.8,
    )

    # ---------------- panel titles -------------------------------------------
    titles = [
        (ax_a, "(a) Regional setting"),
        (ax_b, "(b) Land cover (NLCD 2021)"),
        (ax_c, "(c) Topography and bathymetry"),
        (ax_d, f"(d) Monitoring network ({len(have)} gauges)"),
    ]
    for ax, txt in titles:
        pos = ax.get_position()
        fig.text(pos.x0, pos.y1 + 0.009, txt, ha="left", va="bottom",
                 fontsize=15.0, fontweight="bold")

    fig.savefig(OUT + ".png", dpi=300)
    fig.savefig(OUT + ".pdf")
    plt.show()
    log("SAVED", OUT + ".png")
    log("SAVED", OUT + ".pdf")


main()
