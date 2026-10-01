"""Re-plot manuscript Figure 6 with larger fonts.

Figure 6 in
project/manuscript/revision_2026_08_31/Operational_ST_GNN_Ascension_HESS_draft_MFG2.docx
is the all-gauges panel from
src/figures/recipe_graph_map_base.py
(title: Observation lead-lag graph for all mapped gauges, 68 nodes, 497 directed edges).

This script is that same all-gauges recipe: same 68 nodes, same 497 edges, same HEC-RAS
basemap via make_basemap, same lakes, north arrow, 10 km bar, city and river labels, and
the same top-70 arrowheads. Only type sizes and the Baton Rouge label offset change.
The Baton Rouge offset is so the larger city name is not hidden under the north arrow.
"""
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

sys.path.insert(
    0,
    "project/"
    "manuscript/figure_library",
)
from _mapbase import (
    BASIN_COL,
    BASIN_NAME,
    CITIES,
    add_cities,
    add_river_label,
    basin,
    is_external,
    load_graph,
    load_xy,
    make_basemap,
    source,
    to_xy,
)

OUT = (
    "project/"
    "manuscript/revision_2026_08_31/"
    "figures/fig06_graphmap_larger_fonts"
)
RIVER_A = 0.30
PAD_R = 9000.0
PAD_R_EAST_LEGEND = 56000.0
PAD_REG = 14000.0

print("[CWD]", os.getcwd())
print("[OUT]", OUT + ".png")

xy = load_xy()
nodes, A = load_graph()
have = [n for n in nodes if n in xy]
inp = [n for n in have if not is_external(n)]
print("[GRAPH] nodes", len(nodes), "mapped", len(have), "in-parish", len(inp))
print("[GRAPH] directed edges", int((A > 0).sum() - (A.diagonal() > 0).sum()))

xs_i = [xy[g][0] for g in inp]
ys_i = [xy[g][1] for g in inp]
rL = min(xs_i) - PAD_R
rR = max(xs_i) + PAD_R
rB = min(ys_i) - PAD_R
rT = max(ys_i) + PAD_R
rR_legend = rR + PAD_R_EAST_LEGEND
print("[EXTENT] parish box", rL, rR_legend, rB, rT)

cities_all = ["Baton Rouge", "Gonzales", "Donaldsonville", "New Orleans"]
cxy = [to_xy(*CITIES[c]) for c in cities_all]
allx = [xy[g][0] for g in have] + [p[0] for p in cxy]
ally = [xy[g][1] for g in have] + [p[1] for p in cxy]
gL = min(allx) - PAD_REG
gR = max(allx) + PAD_REG
gB = min(ally) - PAD_REG
gT = max(ally) + PAD_REG
print("[EXTENT] regional", gL, gR, gB, gT)

wmax = float(A.max()) if A.max() > 0 else 1.0
print("[GRAPH] max edge weight", wmax)

fig_all, ax_all = plt.subplots(figsize=(12.2, 7.0))
make_basemap(
    None,
    [],
    ax=ax_all,
    extent=(gL, gR, gB, gT),
    river_alpha=RIVER_A + 0.02,
    river_lw=0.42,
)
ax_all.tick_params(labelsize=14)
ax_all.xaxis.label.set_size(16)
ax_all.yaxis.label.set_size(16)
for text_artist in ax_all.texts:
    label = text_artist.get_text().strip()
    if label == "10 km":
        text_artist.set_fontsize(15)
    if label == "N":
        text_artist.set_fontsize(22)

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

print("[GRAPH] drawn directed edges", len(all_segs))
ax_all.add_collection(
    LineCollection(all_segs, colors=all_cols, linewidths=all_widths, zorder=5)
)
all_edge_rows = sorted(all_edge_rows, key=lambda row: row[0], reverse=True)
n_arrows = 0
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
    n_arrows += 1
print("[GRAPH] arrowheads", n_arrows)

n_in = 0
n_ext = 0
for n in have:
    x, y = xy[n]
    if is_external(n):
        ax_all.scatter(x, y, marker="x", s=56, c="0.25", linewidths=1.6, zorder=9)
        n_ext += 1
    else:
        m = "o" if source(n) == "A" else "^"
        ax_all.scatter(
            x,
            y,
            marker=m,
            s=64,
            c=BASIN_COL.get(basin(n), "#555"),
            edgecolors="black",
            linewidths=0.7,
            zorder=10,
        )
        n_in += 1
print("[MARKERS] in-parish", n_in, "external", n_ext)

add_cities(ax_all, ["Gonzales", "Donaldsonville", "New Orleans"], dot=42, fs=14)
bx, by = to_xy(*CITIES["Baton Rouge"])
ax_all.scatter([bx], [by], s=42, marker="s", c="black", zorder=18)
ax_all.annotate(
    "Baton Rouge",
    (bx, by),
    xytext=(16, -18),
    textcoords="offset points",
    fontsize=14,
    fontweight="bold",
    color="#1a1a1a",
    zorder=19,
)
add_river_label(ax_all, "Mississippi River", -90.95, 30.20, rot=-52, fs=15)
add_river_label(ax_all, "Lake Pontchartrain", -90.07, 30.20, rot=0, fs=14, color="#2c6b94")
add_river_label(ax_all, "Lake\nMaurepas", -90.49, 30.26, rot=0, fs=12, color="#2c6b94")
ax_all.set_title(
    "Observation lead-lag graph for all mapped gauges ("
    + str(len(have))
    + " of "
    + str(len(nodes))
    + " graph nodes, "
    + str(len(all_segs))
    + " directed edges)",
    fontsize=16,
)

leg_all = [mpatches.Patch(fc=BASIN_COL[b], ec="black", label=BASIN_NAME[b]) for b in BASIN_COL]
leg_all += [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="0.72",
        markeredgecolor="black",
        ms=11,
        label="APG in-house gauge",
    ),
    Line2D(
        [0],
        [0],
        marker="^",
        color="w",
        markerfacecolor="0.72",
        markeredgecolor="black",
        ms=11,
        label="USGS in-parish gauge",
    ),
    Line2D(
        [0],
        [0],
        marker="x",
        color="0.25",
        lw=0,
        ms=10,
        markeredgewidth=1.4,
        label="supporting gauge (out of parish)",
    ),
    Line2D([0], [0], color="#1b8bb5", lw=1.4, label="river / drainage (HEC-RAS)"),
    Line2D([0], [0], color="#34557a", lw=2.2, label="routing edge with arrowheads"),
]
lg_all = ax_all.legend(
    handles=leg_all,
    loc="upper right",
    framealpha=1.0,
    borderpad=0.6,
    fontsize=12.0,
    title="Legend",
    title_fontsize=13.5,
)
lg_all.get_frame().set_linewidth(0.7)
lg_all.set_zorder(25)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig_all.tight_layout()
fig_all.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
fig_all.savefig(OUT + ".pdf", bbox_inches="tight")
print("[SAVED]", OUT + ".png")
print("[SAVED]", OUT + ".pdf")
plt.show()
