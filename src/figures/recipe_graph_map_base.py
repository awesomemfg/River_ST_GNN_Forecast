"""Fig 8 (4.1) - deployed observation lead-lag routing graph, TWO panels on the real HEC-RAS basemap.

(a) Regional setting: the parish in its SE-Louisiana context - Mississippi River, Baton Rouge, Gonzales,
    Donaldsonville, New Orleans, and ALL 68 network gauges, including BCSU0002 (NOAA 8761927) as an
    out-of-parish supporting gauge. A dashed box marks the parish-focus extent of (b).
(b) Forecast network (Ascension Parish): the in-parish gauges and the routing edges among them, nodes
    coloured by basin and shaped by source. HEC-RAS river kept faint; bold north arrow.
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _mapbase import (make_basemap, load_xy, load_graph, basin, source, is_external,
                      BASIN_COL, BASIN_NAME, add_cities, add_river_label, to_xy, CITIES)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

OUT = "project/manuscript/figure_library_outputs/fig08_graphmap"
OUT_ALL = "project/manuscript/figure_library_outputs/fig08_graphmap_all_gauges"
RIVER_A = 0.30

xy = load_xy()
nodes, A = load_graph()
have = [n for n in nodes if n in xy]
inp = [n for n in have if not is_external(n)]
ext = [n for n in have if is_external(n)]

# --- extents ---
PAD_R = 9000.0
PAD_R_EAST_LEGEND = 56000.0
xs_i = [xy[g][0] for g in inp]
ys_i = [xy[g][1] for g in inp]
rL, rR = min(xs_i) - PAD_R, max(xs_i) + PAD_R
rB, rT = min(ys_i) - PAD_R, max(ys_i) + PAD_R   # parish-focus extent (panel b)
rR_legend = rR + PAD_R_EAST_LEGEND

cities_all = ["Baton Rouge", "Gonzales", "Donaldsonville", "New Orleans"]
cxy = [to_xy(*CITIES[c]) for c in cities_all]
allx = [xy[g][0] for g in have] + [p[0] for p in cxy]
ally = [xy[g][1] for g in have] + [p[1] for p in cxy]
PAD_REG = 14000.0
gL, gR = min(allx) - PAD_REG, max(allx) + PAD_REG
gB, gT = min(ally) - PAD_REG, max(ally) + PAD_REG  # regional extent (panel a)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(16.6, 5.9))

# ============================== (a) regional ==============================
make_basemap(None, [], ax=axL, extent=(gL, gR, gB, gT), river_alpha=RIVER_A, river_lw=0.40)
for n in have:
    x, y = xy[n]
    if is_external(n):
        axL.scatter(x, y, marker="x", s=44, c="0.35", linewidths=1.4, zorder=7)
    else:
        m = "o" if source(n) == "A" else "^"
        axL.scatter(x, y, marker=m, s=46, c=BASIN_COL.get(basin(n), "#555"),
                    edgecolors="black", linewidths=0.55, zorder=8)
bx, by = xy["BCSU0002"]
#axL.annotate("BCSU0002\nNOAA supporting gauge", (bx, by), xytext=(-8, 10), textcoords="offset points",
#            fontsize=9.5, fontweight="bold", color="0.25", ha="right", zorder=13)
add_cities(axL, cities_all)
add_river_label(axL, "Mississippi River", -90.95, 30.20, rot=-52, fs=12)
add_river_label(axL, "Lake Pontchartrain", -90.07, 30.20, rot=0, fs=10.5, color="#2c6b94")
add_river_label(axL, "Lake\nMaurepas", -90.49, 30.26, rot=0, fs=8.0, color="#2c6b94")
# parish-focus box (extent of panel b)
axL.add_patch(Rectangle((rL, rB), rR_legend - rL, rT - rB, fill=False, edgecolor="#444", linewidth=1.6,
                        linestyle="--", zorder=15))
axL.text(rL, rT, " parish focus (b)", fontsize=9.5, fontstyle="italic", color="#444", va="bottom", zorder=15)
axL.set_title("(a) Regional setting")

# ============================== (b) parish network ==============================
make_basemap(None, [], ax=axR, extent=(rL, rR_legend, rB, rT), river_alpha=RIVER_A + 0.04, river_lw=0.45)
inset = set(inp)
wmax = float(A.max()) if A.max() > 0 else 1.0
segs, cols, edge_widths, edge_rows = [], [], [], []
for d in range(len(nodes)):
    for u in range(len(nodes)):
        w = A[d, u]
        if w <= 0 or nodes[u] not in inset or nodes[d] not in inset:
            continue
        segs.append([xy[nodes[u]], xy[nodes[d]]])
        cols.append((0.12, 0.25, 0.42, 0.20 + 0.60 * (w / wmax)))
        edge_widths.append(0.75 + 1.10 * (w / wmax))
        edge_rows.append((float(w), xy[nodes[u]], xy[nodes[d]]))
axR.add_collection(LineCollection(segs, colors=cols, linewidths=edge_widths, zorder=5))
edge_rows = sorted(edge_rows, key=lambda row: row[0], reverse=True)
for edge_weight, start_xy, end_xy in edge_rows[:45]:
    arrow_alpha = 0.32 + 0.45 * (edge_weight / wmax)
    arrow = FancyArrowPatch(
        start_xy,
        end_xy,
        arrowstyle="-|>",
        mutation_scale=10.5,
        linewidth=1.05,
        color=(0.10, 0.23, 0.36, arrow_alpha),
        shrinkA=8.0,
        shrinkB=8.0,
        zorder=6,
    )
    axR.add_patch(arrow)
for n in inp:
    x, y = xy[n]
    m = "o" if source(n) == "A" else "^"
    axR.scatter(x, y, marker=m, s=88, c=BASIN_COL.get(basin(n), "#555"),
                edgecolors="black", linewidths=0.8, zorder=8)
add_cities(axR, ["Gonzales", "Donaldsonville"], dot=40, fs=12)
add_river_label(axR, "Mississippi River", -90.99, 30.16, rot=-58, fs=12)
'''
axR.annotate("edge direction", xy=(0.33, 0.935), xytext=(0.13, 0.935), xycoords="axes fraction",
             textcoords="axes fraction", ha="center", va="center", fontsize=10.0, fontweight="bold",
             color="#263f5c",
             arrowprops={"arrowstyle": "-|>", "lw": 2.2, "color": "#263f5c",
                         "mutation_scale": 18.0},
             bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "0.55",
                   "alpha": 0.90},
             zorder=28)
'''
axR.set_title(f"(b) Forecast network - Ascension Parish ({len(inp)} in-parish gauges, {len(segs)} edges)")

leg = [mpatches.Patch(fc=BASIN_COL[b], ec="black", label=BASIN_NAME[b]) for b in BASIN_COL]
leg += [Line2D([0], [0], marker="o", color="w", markerfacecolor="0.72", markeredgecolor="black", ms=9, label="APG in-house gauge"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="0.72", markeredgecolor="black", ms=9, label="USGS gauge"),
        Line2D([0], [0], marker="x", color="0.35", lw=0, ms=8, markeredgewidth=1.2, label="supporting gauge (out of parish)"),
        Line2D([0], [0], color="#1b8bb5", lw=1.4, label="river / drainage (HEC-RAS)"),
        Line2D([0], [0], color="#34557a", lw=2.4, label="routing edge with arrowheads")]
lg = axR.legend(handles=leg, loc="upper right", framealpha=1.0, borderpad=0.6, fontsize=9.0,
                title="Legend", title_fontsize=10.5)
lg.get_frame().set_linewidth(0.7)
lg.set_zorder(25)

n_edges_total = int((A > 0).sum() - np.trace(A > 0))
fig.suptitle(f"Deployed observation lead-lag routing graph ({len(nodes)} gauges, {n_edges_total} directed edges)", fontsize=17, y=0.89)
fig.tight_layout(rect=[0, 0, 1, 0.925])
fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
print("[SAVED]", OUT + ".png", "in-parish:", len(inp), "edges(in-parish):", len(segs))

# ============================== all gauges graph ==============================
fig_all, ax_all = plt.subplots(figsize=(12.2, 7.0))
make_basemap(None, [], ax=ax_all, extent=(gL, gR, gB, gT), river_alpha=RIVER_A + 0.02, river_lw=0.42)

allset = set(have)
all_segs = []
all_cols = []
all_widths = []
all_edge_rows = []
for d in range(len(nodes)):
    for u in range(len(nodes)):
        w = A[d, u]
        if w <= 0:
            continue
        if nodes[u] not in allset:
            continue
        if nodes[d] not in allset:
            continue
        all_segs.append([xy[nodes[u]], xy[nodes[d]]])
        all_cols.append((0.12, 0.25, 0.42, 0.16 + 0.58 * (w / wmax)))
        all_widths.append(0.55 + 1.05 * (w / wmax))
        all_edge_rows.append((float(w), xy[nodes[u]], xy[nodes[d]]))

ax_all.add_collection(LineCollection(all_segs, colors=all_cols, linewidths=all_widths, zorder=5))
all_edge_rows = sorted(all_edge_rows, key=lambda row: row[0], reverse=True)
for edge_weight, start_xy, end_xy in all_edge_rows[:70]:
    arrow_alpha = 0.30 + 0.45 * (edge_weight / wmax)
    arrow = FancyArrowPatch(
        start_xy,
        end_xy,
        arrowstyle="-|>",
        mutation_scale=9.5,
        linewidth=0.95,
        color=(0.10, 0.23, 0.36, arrow_alpha),
        shrinkA=6.5,
        shrinkB=6.5,
        zorder=6,
    )
    ax_all.add_patch(arrow)

for n in have:
    x, y = xy[n]
    if is_external(n):
        ax_all.scatter(x, y, marker="x", s=56, c="0.25", linewidths=1.6, zorder=9)
    else:
        m = "o" if source(n) == "A" else "^"
        ax_all.scatter(x, y, marker=m, s=64, c=BASIN_COL.get(basin(n), "#555"),
                       edgecolors="black", linewidths=0.7, zorder=10)

add_cities(ax_all, cities_all, dot=38, fs=11)
add_river_label(ax_all, "Mississippi River", -90.95, 30.20, rot=-52, fs=12)
add_river_label(ax_all, "Lake Pontchartrain", -90.07, 30.20, rot=0, fs=10.5, color="#2c6b94")
add_river_label(ax_all, "Lake\nMaurepas", -90.49, 30.26, rot=0, fs=8.0, color="#2c6b94")
ax_all.set_title(f"Observation lead-lag graph for all mapped gauges ({len(have)} of {len(nodes)} graph nodes, "
                 f"{len(all_segs)} directed edges)")

leg_all = [mpatches.Patch(fc=BASIN_COL[b], ec="black", label=BASIN_NAME[b]) for b in BASIN_COL]
leg_all += [Line2D([0], [0], marker="o", color="w", markerfacecolor="0.72", markeredgecolor="black", ms=9, label="APG in-house gauge"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="0.72", markeredgecolor="black", ms=9, label="USGS in-parish gauge"),
            Line2D([0], [0], marker="x", color="0.25", lw=0, ms=8, markeredgewidth=1.4, label="supporting gauge (out of parish)"),
            Line2D([0], [0], color="#1b8bb5", lw=1.4, label="river / drainage (HEC-RAS)"),
            Line2D([0], [0], color="#34557a", lw=2.2, label="routing edge with arrowheads")]
lg_all = ax_all.legend(handles=leg_all, loc="upper right", framealpha=1.0, borderpad=0.6, fontsize=9.0,
                       title="Legend", title_fontsize=10.5)
lg_all.get_frame().set_linewidth(0.7)
lg_all.set_zorder(25)

fig_all.tight_layout()
fig_all.savefig(OUT_ALL + ".png", dpi=300, bbox_inches="tight")
fig_all.savefig(OUT_ALL + ".pdf", bbox_inches="tight")
print("[SAVED]", OUT_ALL + ".png", "gauges:", len(have), "edges(all):", len(all_segs))
