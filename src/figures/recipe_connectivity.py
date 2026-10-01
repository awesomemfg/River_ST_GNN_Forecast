"""Fig S1 (supp) - the six candidate connectivity graphs on the real HEC-RAS river basemap.
2x3 montage: no-graph Bi-LSTM hub; observation lead-lag (deployed); observation within-basin;
DEM downslope; DEM within-basin; HEC-RAS. In-parish gauges + edges over the river/drainage linework.
"""
import os
import sys
import numpy as np
import geopandas as gpd
from matplotlib.collections import LineCollection
sys.path.insert(
    0,
    "project/"
    "manuscript/figure_library",
)
from _mapbase import (load_xy, load_coords_lonlat, basin, source, is_external, BASIN_COL, BASIN_NAME,
                      ASC, LA_PARISHES, HECRAS_CRS, LONLAT, _load_rivers, add_north, add_water,
                      add_cities, add_river_label, to_xy, CITIES)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

GDIR = "project/hpc/Training/GNN"
OUT = ("project/"
       "manuscript/20260826_Final_V2/"
       "output/figures/fig03_inparish")
OUT_ALL = ("project/"
           "manuscript/20260826_Final_V2/"
           "output/figures/fig03")

PANELS = [
    (None, "(1) No-graph Bi-LSTM\n(shared network, no river graph)"),
    (f"{GDIR}/gnn_graph.npz", "(2) Observation lead-lag\n(deployed)"),
    (f"{GDIR}/gnn_graph_basinmask_v2.npz", "(3) Observation, within-basin"),
    (f"{GDIR}/gnn_graph_tiff.npz", "(4) DEM downslope"),
    (f"{GDIR}/gnn_graph_tiff_basinmask.npz", "(5) DEM, within-basin"),
    (f"{GDIR}/gnn_graph_physics.npz", "(6) HEC-RAS hydraulic"),
]

xy = load_xy()
inp = [g for g in xy if not is_external(g)]
inset = set(inp)
rivers = _load_rivers()
asc = gpd.read_file(ASC)
asc = (asc.set_crs(LONLAT) if asc.crs is None else asc).to_crs(HECRAS_CRS)
la = gpd.read_file(LA_PARISHES)
la = (la.set_crs(LONLAT) if la.crs is None else la).to_crs(HECRAS_CRS)
ab = asc.total_bounds
pad = 9000.0
xs = [xy[g][0] for g in inp]
ys = [xy[g][1] for g in inp]
L = min(float(ab[0]), min(xs)) - pad
R = max(float(ab[2]), max(xs)) + pad
B = min(float(ab[1]), min(ys)) - pad
T = max(float(ab[3]), max(ys)) + pad


def draw_base(ax):
    ax.set_facecolor("#f6f4ed")
    la.plot(ax=ax, facecolor="#e9e6dc", edgecolor="#b3b3b3", linewidth=0.4, zorder=1)
    asc.plot(ax=ax, facecolor="#fffdf7", edgecolor="black", linewidth=1.4, alpha=0.78, zorder=2)
    add_water(ax, zorder=2.6)
    ax.add_collection(LineCollection(rivers, colors="#1b8bb5", linewidths=0.28, alpha=0.30, zorder=3))
    ax.set_xlim(L, R)
    ax.set_ylim(B, T)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    add_north(ax, x=0.065, y=0.95, size=0.16)


def draw_nodes(ax):
    for g in inp:
        x, y = xy[g]
        ax.scatter(x, y, s=66, c=BASIN_COL.get(basin(g), "#555"), edgecolors="black", linewidths=0.7, zorder=8)


fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.8), gridspec_kw={"wspace": 0.04, "hspace": 0.22})
for ax, (path, title) in zip(axes.ravel(), PANELS):
    draw_base(ax)
    if path is None:
        cx = float(np.mean([xy[g][0] for g in inp]))
        cy = float(np.mean([xy[g][1] for g in inp]))
        segs = [[(cx, cy), xy[g]] for g in inp]
        ax.add_collection(LineCollection(segs, colors=[(0.3, 0.3, 0.3, 0.30)], linewidths=0.4, zorder=5))
        ax.scatter([cx], [cy], s=120, marker="s", c="0.35", edgecolors="black", zorder=9)
        ne = len(segs)
    else:
        g = np.load(path, allow_pickle=True)
        nodes = [str(x) for x in g["nodes"]]
        A = g["A_norm"].astype("float32")
        segs = []
        for d in range(len(nodes)):
            for u in range(len(nodes)):
                if A[d, u] > 0 and nodes[u] in inset and nodes[d] in inset:
                    segs.append([xy[nodes[u]], xy[nodes[d]]])
        ax.add_collection(LineCollection(segs, colors=[(0.20, 0.33, 0.48, 0.30)], linewidths=0.4, zorder=5))
        ne = len(segs)
    draw_nodes(ax)
    ax.set_title(f"{title}\n{ne} connections", fontsize=15.5, fontweight="bold", pad=6)

leg = [mpatches.Patch(fc=BASIN_COL[b], ec="black", label=BASIN_NAME[b]) for b in BASIN_COL]
leg += [Line2D([0], [0], color="#1b8bb5", lw=1.4, label="river / drainage (HEC-RAS)")]
fig.legend(handles=leg, loc="lower center", ncol=5, fontsize=12.5, frameon=True, bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Candidate connectivity graphs for the ST-GNN (in-parish gauges shown)",
             fontsize=18.5, fontweight="bold", y=0.945)
fig.subplots_adjust(left=0.012, right=0.988, top=0.825, bottom=0.095, wspace=0.04, hspace=0.22)
fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
print("[SAVED]", OUT + ".png")

# ---- All-gauge companion: same six candidate graphs, expanded to the mapped graph-node extent. ----
graph_node_set = set()
for graph_path, _ in PANELS:
    if graph_path is None:
        continue
    graph_data = np.load(graph_path, allow_pickle=True)
    graph_nodes = [str(x) for x in graph_data["nodes"]]
    graph_node_set.update(graph_nodes)

all_mapped = [g for g in sorted(graph_node_set) if g in xy]
missing_coords = [g for g in sorted(graph_node_set) if g not in xy]
print("[INFO] All-gauge companion graph nodes:", len(graph_node_set))
print("[INFO] All-gauge companion mapped nodes:", len(all_mapped))
print("[INFO] All-gauge companion missing coordinates:", missing_coords)

cities_all = ["Baton Rouge", "Gonzales", "Donaldsonville", "New Orleans"]
city_xy = [to_xy(*CITIES[name]) for name in cities_all]
xs_all = [xy[g][0] for g in all_mapped] + [point[0] for point in city_xy]
ys_all = [xy[g][1] for g in all_mapped] + [point[1] for point in city_xy]
pad_all = 14000.0
L_all = min(xs_all) - pad_all
R_all = max(xs_all) + pad_all
B_all = min(ys_all) - pad_all
T_all = max(ys_all) + pad_all
allset = set(all_mapped)


def draw_base_all(ax):
    ax.set_facecolor("#f6f4ed")
    la.plot(ax=ax, facecolor="#e9e6dc", edgecolor="#b3b3b3", linewidth=0.4, zorder=1)
    asc.plot(ax=ax, facecolor="#fffdf7", edgecolor="black", linewidth=1.4, alpha=0.78, zorder=2)
    add_water(ax, zorder=2.6)
    ax.add_collection(LineCollection(rivers, colors="#1b8bb5", linewidths=0.28, alpha=0.30, zorder=3))
    ax.set_xlim(L_all, R_all)
    ax.set_ylim(B_all, T_all)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    add_north(ax, x=0.055, y=0.94, size=0.15)
    add_river_label(ax, "Lake Pontchartrain", -90.07, 30.20, rot=0, fs=10.5, color="#2c6b94")
    add_river_label(ax, "Lake\nMaurepas", -90.49, 30.26, rot=0, fs=9.5, color="#2c6b94")


def draw_nodes_all(ax):
    for gauge in all_mapped:
        x, y = xy[gauge]
        if is_external(gauge):
            ax.scatter(x, y, marker="x", s=80, c="0.25", linewidths=1.8, zorder=9)
        else:
            marker = "o"
            if source(gauge) == "U":
                marker = "^"
            ax.scatter(x, y, marker=marker, s=74, c=BASIN_COL.get(basin(gauge), "#555"),
                       edgecolors="black", linewidths=0.7, zorder=10)


fig_all, axes_all = plt.subplots(2, 3, figsize=(12.0, 7.8), gridspec_kw={"wspace": 0.04, "hspace": 0.22})
for ax, (path, title) in zip(axes_all.ravel(), PANELS):
    draw_base_all(ax)
    if path is None:
        cx = float(np.mean([xy[g][0] for g in all_mapped]))
        cy = float(np.mean([xy[g][1] for g in all_mapped]))
        segs = [[(cx, cy), xy[g]] for g in all_mapped]
        ax.add_collection(LineCollection(segs, colors=[(0.3, 0.3, 0.3, 0.28)], linewidths=0.34, zorder=5))
        ax.scatter([cx], [cy], s=180, marker="s", c="0.35", edgecolors="black", zorder=9)
        ne = len(segs)
    else:
        graph_data = np.load(path, allow_pickle=True)
        graph_nodes = [str(x) for x in graph_data["nodes"]]
        adjacency = graph_data["A_norm"].astype("float32")
        segs = []
        for d in range(len(graph_nodes)):
            for u in range(len(graph_nodes)):
                if adjacency[d, u] <= 0:
                    continue
                if graph_nodes[u] not in allset:
                    continue
                if graph_nodes[d] not in allset:
                    continue
                segs.append([xy[graph_nodes[u]], xy[graph_nodes[d]]])
        ax.add_collection(LineCollection(segs, colors=[(0.20, 0.33, 0.48, 0.28)], linewidths=0.34, zorder=5))
        ne = len(segs)
    draw_nodes_all(ax)
    ax.set_title(f"{title}\n{ne} connections", fontsize=15.5, fontweight="bold", pad=6)

leg_all = [mpatches.Patch(fc=BASIN_COL[b], ec="black", label=BASIN_NAME[b]) for b in BASIN_COL]
leg_all += [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="0.72", markeredgecolor="black", ms=8,
           label="APG in-house gauge"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="0.72", markeredgecolor="black", ms=8,
           label="USGS in-parish gauge"),
    Line2D([0], [0], marker="x", color="0.25", lw=0, ms=7, markeredgewidth=1.1,
           label="supporting gauge (out of parish)"),
    Line2D([0], [0], color="#1b8bb5", lw=1.4, label="river / drainage (HEC-RAS)"),
]
fig_all.legend(handles=leg_all, loc="lower center", ncol=4, fontsize=12.0, frameon=True,
               bbox_to_anchor=(0.5, 0.005))
fig_all.suptitle("Candidate connectivity graphs for the ST-GNN (all mapped graph gauges shown)",
                 fontsize=18.5, fontweight="bold", y=0.950)
fig_all.subplots_adjust(left=0.012, right=0.988, top=0.820, bottom=0.105, wspace=0.04, hspace=0.22)
fig_all.savefig(OUT_ALL + ".png", dpi=300, bbox_inches="tight")
fig_all.savefig(OUT_ALL + ".pdf", bbox_inches="tight")
print("[SAVED]", OUT_ALL + ".png")
plt.show()
