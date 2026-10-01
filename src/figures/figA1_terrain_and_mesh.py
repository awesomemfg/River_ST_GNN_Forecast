"""Fig 3a/3b (3.2.2) - paper-grade DEM and HEC-RAS mesh context maps (the terrain and hydraulic sources
for the candidate routing graphs). Unified paper style (Nimbus), Okabe-Ito stations, parish boundary,
scale bar + north arrow. StatePlane LA-South (EPSG:3452 / the EBR projection).
"""
import os
import sys
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import h5py
import geopandas as gpd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(__file__))
from _mapbase import (load_xy, load_graph, basin, source, is_external, BASIN_COL, BASIN_NAME, ASC,
                      HECRAS_CRS, LONLAT, add_water, add_cities, add_river_label, to_xy, CITIES)
from _units import FT2M
import matplotlib.pyplot as plt

TIF = "project/HECRAS_Files/New_Files/EBR_Regional_wBathymetry_v4_CRD.EBR_Regional_wBathymetry_v4.Resampled.tif"
G33 = "project/HECRAS_Files/EBR/EBR-APG_Regional.g33.hdf"
DST = "project/manuscript/figure_library_outputs"

xy = load_xy()
graph_nodes, _ = load_graph()
have = [g for g in graph_nodes if g in xy]
missing_graph_coords = [g for g in graph_nodes if g not in xy]
print("[INFO] Graph nodes:", len(graph_nodes))
print("[INFO] Mapped graph nodes:", len(have))
print("[INFO] Graph nodes missing coordinates:", missing_graph_coords)
inp = [g for g in xy if not is_external(g)]
asc = gpd.read_file(ASC)
if asc.crs is None:
    asc = asc.set_crs(LONLAT)
asc = asc.to_crs(HECRAS_CRS)
ab = asc.total_bounds
xs = [xy[g][0] for g in inp]
ys = [xy[g][1] for g in inp]
pad = 12000.0
L = min(float(ab[0]), min(xs)) - pad
R = max(float(ab[2]), max(xs)) + pad
B = min(float(ab[1]), min(ys)) - pad
T = max(float(ab[3]), max(ys)) + pad


def add_stations(ax):
    for g in inp:
        x, y = xy[g]
        m = "o" if source(g) == "A" else "^"
        ax.scatter(x, y, marker=m, s=64, c=BASIN_COL.get(basin(g), "#555"),
                   edgecolors="black", linewidths=0.7, zorder=8)


def decorate(ax, title):
    add_water(ax, lakes=True, rivers=True, fill=False, zorder=2.6)
    asc.boundary.plot(ax=ax, color="black", linewidth=2.0, alpha=0.85, zorder=6)
    ax.set_xlim(L, R)
    ax.set_ylim(B, T)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting (NAD83 Louisiana South StatePlane, US ft)", fontsize=12)
    ax.set_ylabel("Northing (US ft)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.annotate("N", xy=(0.045, 0.95), xytext=(0.045, 0.82), xycoords="axes fraction", ha="center",
                fontsize=15, fontweight="bold",
                arrowprops={"facecolor": "black", "edgecolor": "black", "width": 4.0, "headwidth": 11.0})
    sb = 32808.4
    sx = L + 0.06 * (R - L)
    sy = B + 0.05 * (T - B)
    ax.plot([sx, sx + sb], [sy, sy], color="black", lw=3.5, solid_capstyle="butt", zorder=10)
    ax.text(sx + sb / 2, sy + 0.012 * (T - B), "10 km", ha="center", va="bottom", fontsize=11, zorder=10)
    ax.set_title(title)


def station_legend(ax):
    leg = [mpatches.Patch(fc=BASIN_COL[b], ec="black", label=BASIN_NAME[b]) for b in BASIN_COL]
    leg += [Line2D([0], [0], marker="o", color="w", markerfacecolor="0.72", markeredgecolor="black", ms=9, label="APG in-house gauge"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="0.72", markeredgecolor="black", ms=9, label="USGS gauge")]
    lg = ax.legend(handles=leg, loc="lower right", framealpha=0.96, fontsize=9.5, title="In-parish gauge")
    lg.set_zorder(20)


# ---------------- 3a: DEM ----------------
with rasterio.open(TIF) as r:
    win = from_bounds(L, B, R, T, r.transform)
    ow = min(1400, int(win.width))
    oh = int(win.height * ow / max(win.width, 1))
    dem = r.read(1, window=win, out_shape=(oh, ow), resampling=Resampling.average).astype("float32")
    nd = r.nodata
if nd is not None:
    dem[dem == nd] = np.nan
dem[dem < -1e4] = np.nan
dem = dem * FT2M                       # elevation feet -> metres
vmin, vmax = np.nanpercentile(dem, 2), np.nanpercentile(dem, 98)
fig, ax = plt.subplots(figsize=(8.8, 8.0))
im = ax.imshow(dem, extent=[L, R, B, T], origin="upper", cmap="terrain", vmin=vmin, vmax=vmax, zorder=1)
add_stations(ax)
decorate(ax, "(a) Digital elevation model (bathymetry) with the in-parish gauges")
cb = fig.colorbar(im, ax=ax, shrink=0.66, pad=0.02)
cb.set_label("Ground / bed elevation (m)", fontsize=12)
station_legend(ax)
fig.tight_layout()
fig.savefig(f"{DST}/fig03a_context_dem.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{DST}/fig03a_context_dem.pdf", bbox_inches="tight")
print("[SAVED] fig03a_context_dem.png")

# ---------------- 3b: HEC-RAS mesh ----------------
geo = h5py.File(G33, "r")["Geometry"]["2D Flow Areas"]["2D_REG_01"]
cc = geo["Cells Center Coordinate"][:]
inb = (cc[:, 0] >= L) & (cc[:, 0] <= R) & (cc[:, 1] >= B) & (cc[:, 1] <= T)
ccv = cc[inb]
sub = ccv[::3] if len(ccv) > 60000 else ccv
fig, ax = plt.subplots(figsize=(8.8, 8.0))
ax.set_facecolor("#f6f4ed")
ax.scatter(sub[:, 0], sub[:, 1], s=0.6, c="#7aa6c2", alpha=0.5, zorder=1, linewidths=0)
add_stations(ax)
decorate(ax, "(b) HEC-RAS two-dimensional hydraulic mesh with the in-parish gauges")
leg = [mpatches.Patch(fc=BASIN_COL[b], ec="black", label=BASIN_NAME[b]) for b in BASIN_COL]
leg += [Line2D([0], [0], marker="o", color="w", markerfacecolor="0.72", markeredgecolor="black", ms=9, label="APG in-house gauge"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="0.72", markeredgecolor="black", ms=9, label="USGS gauge"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#7aa6c2", markeredgecolor="#7aa6c2", ms=6, label=f"HEC-RAS mesh cell ({len(ccv)//1000}k in view)")]
lg = ax.legend(handles=leg, loc="upper right", framealpha=0.96, fontsize=9.0, title="In-parish gauge")
lg.set_zorder(20)
fig.tight_layout()
fig.savefig(f"{DST}/fig03b_context_hecras.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{DST}/fig03b_context_hecras.pdf", bbox_inches="tight")
print("[SAVED] fig03b_context_hecras.png")

# ---------------- All-gauge companions ----------------
cities_all = ["Baton Rouge", "Gonzales", "Donaldsonville", "New Orleans"]
city_xy = [to_xy(*CITIES[name]) for name in cities_all]
xs_all = [xy[g][0] for g in have] + [point[0] for point in city_xy]
ys_all = [xy[g][1] for g in have] + [point[1] for point in city_xy]
pad_all = 14000.0
L_all = min(xs_all) - pad_all
R_all = max(xs_all) + pad_all
B_all = min(ys_all) - pad_all
T_all = max(ys_all) + pad_all
print("[INFO] All-gauge context bounds:", (L_all, R_all, B_all, T_all))


def add_stations_all(ax):
    for gauge in have:
        x, y = xy[gauge]
        if is_external(gauge):
            ax.scatter(x, y, marker="x", s=70, c="0.25", linewidths=1.5, zorder=9)
        else:
            marker = "o"
            if source(gauge) == "U":
                marker = "^"
            ax.scatter(x, y, marker=marker, s=64, c=BASIN_COL.get(basin(gauge), "#555"),
                       edgecolors="black", linewidths=0.7, zorder=10)


def decorate_all(ax, title):
    add_water(ax, lakes=True, rivers=True, fill=True, zorder=2.6)
    asc.boundary.plot(ax=ax, color="black", linewidth=2.0, alpha=0.85, zorder=6)
    ax.set_xlim(L_all, R_all)
    ax.set_ylim(B_all, T_all)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting (NAD83 Louisiana South StatePlane, US ft)", fontsize=12)
    ax.set_ylabel("Northing (US ft)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.annotate("N", xy=(0.045, 0.95), xytext=(0.045, 0.82), xycoords="axes fraction", ha="center",
                fontsize=15, fontweight="bold",
                arrowprops={"facecolor": "black", "edgecolor": "black", "width": 4.0, "headwidth": 11.0})
    sb = 32808.4
    sx = L_all + 0.06 * (R_all - L_all)
    sy = B_all + 0.05 * (T_all - B_all)
    ax.plot([sx, sx + sb], [sy, sy], color="black", lw=3.5, solid_capstyle="butt", zorder=10)
    ax.text(sx + sb / 2, sy + 0.012 * (T_all - B_all), "10 km", ha="center", va="bottom", fontsize=11, zorder=10)
    add_river_label(ax, "Lake Pontchartrain", -90.29, 30.20, rot=0, fs=10.5, color="#2c6b94")
    add_river_label(ax, "Lake\nMaurepas", -90.49, 30.26, rot=0, fs=8.0, color="#2c6b94")
    ax.set_title(title)


def station_legend_all(ax, extra_handle=None):
    leg = [mpatches.Patch(fc=BASIN_COL[b], ec="black", label=BASIN_NAME[b]) for b in BASIN_COL]
    leg += [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.72", markeredgecolor="black", ms=9,
               label="APG in-house gauge"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="0.72", markeredgecolor="black", ms=9,
               label="USGS in-parish gauge"),
        Line2D([0], [0], marker="x", color="0.25", lw=0, ms=8, markeredgewidth=1.4,
               label="supporting gauge (out of parish)"),
    ]
    if extra_handle is not None:
        leg.append(extra_handle)
    lg = ax.legend(handles=leg, loc="upper right", bbox_to_anchor=(0.985, 0.985),
                   framealpha=0.96, fontsize=9.0, title="Mapped graph gauge")
    lg.set_zorder(20)


# ---------------- 3a all gauges: DEM ----------------
with rasterio.open(TIF) as r:
    win = from_bounds(L_all, B_all, R_all, T_all, r.transform)
    ow = min(1800, max(1, int(win.width)))
    oh = max(1, int(win.height * ow / max(win.width, 1)))
    dem_masked = r.read(
        1,
        window=win,
        out_shape=(oh, ow),
        resampling=Resampling.average,
        boundless=True,
        masked=True,
    )
    dem_all = dem_masked.astype("float32").filled(np.nan)
dem_all[dem_all < -1e4] = np.nan
dem_all = dem_all * FT2M              # elevation feet -> metres
vmin_all, vmax_all = np.nanpercentile(dem_all, 2), np.nanpercentile(dem_all, 98)
fig, ax = plt.subplots(figsize=(12.0, 7.2))
ax.set_facecolor("#f6f4ed")
im = ax.imshow(dem_all, extent=[L_all, R_all, B_all, T_all], origin="upper",
               cmap="terrain", vmin=vmin_all, vmax=vmax_all, zorder=1)
add_stations_all(ax)
decorate_all(ax, f"(a) Digital elevation model (bathymetry) with all mapped graph gauges ({len(have)} of {len(graph_nodes)})")
cb = fig.colorbar(im, ax=ax, shrink=0.68, pad=0.02)
cb.set_label("Ground / bed elevation (m)", fontsize=12)
station_legend_all(ax)
fig.tight_layout()
fig.savefig(f"{DST}/fig03a_context_dem_all_gauges.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{DST}/fig03a_context_dem_all_gauges.pdf", bbox_inches="tight")
print("[SAVED] fig03a_context_dem_all_gauges.png")

# ---------------- 3b all gauges: HEC-RAS mesh ----------------
inb_all = (cc[:, 0] >= L_all) & (cc[:, 0] <= R_all) & (cc[:, 1] >= B_all) & (cc[:, 1] <= T_all)
ccv_all = cc[inb_all]
step_all = max(1, len(ccv_all) // 90000)
sub_all = ccv_all[::step_all]
fig, ax = plt.subplots(figsize=(12.0, 7.2))
ax.set_facecolor("#f6f4ed")
ax.scatter(sub_all[:, 0], sub_all[:, 1], s=0.6, c="#7aa6c2", alpha=0.5, zorder=1, linewidths=0)
add_stations_all(ax)
decorate_all(ax, f"(b) HEC-RAS two-dimensional hydraulic mesh with all mapped graph gauges ({len(have)} of {len(graph_nodes)})")
mesh_handle = Line2D([0], [0], marker="o", color="w", markerfacecolor="#7aa6c2", markeredgecolor="#7aa6c2",
                     ms=6, label=f"HEC-RAS mesh cell ({len(ccv_all)//1000}k in view)")
station_legend_all(ax, extra_handle=mesh_handle)
fig.tight_layout()
fig.savefig(f"{DST}/fig03b_context_hecras_all_gauges.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{DST}/fig03b_context_hecras_all_gauges.pdf", bbox_inches="tight")
print("[SAVED] fig03b_context_hecras_all_gauges.png")
