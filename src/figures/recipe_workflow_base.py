"""Fig 1 (3.1) — operational lifecycle of the ST-GNN system, sophisticated icon version.

Built on the shared diagram primitives (databases, funnels, chips, network, document stacks, clock,
globe) and the shared paper style, matching the visual language of the other paper diagrams. Labels are
plain-English (no telemetry / harmonize / guard / versioned / immutable / P80 jargon).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from _card import apply
apply()
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from paper_diagram_primitives import (draw_database, draw_funnel, draw_chip, draw_network,
                                      draw_tensor_stack, draw_document_stack, draw_clock, draw_web_globe)

OUT = "project/manuscript/figure_library_outputs/fig01_workflow"

BLUE, BLUE_L = "#2F6FAE", "#EAF2F8"
ORANGE, ORANGE_L = "#D58200", "#FFF4E2"
GREEN, GREEN_L = "#17875B", "#E8F5EF"
PURPLE, PURPLE_L = "#6A4195", "#F2ECF7"
RED, DARK, GRAY = "#B84435", "#252525", "#666666"


def arrow(ax, p0, p1, color, dashed=False, lw=1.5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
                                 linestyle="--" if dashed else "-", color=color, shrinkA=1, shrinkB=1, zorder=8))


def label(ax, cx, ty, title, detail, color):
    ax.text(cx, ty, title, ha="center", va="top", fontsize=12.5, fontweight="bold", color=color, linespacing=0.96, zorder=10)
    ax.text(cx, ty - 0.86, detail, ha="center", va="top", fontsize=10.0, color=DARK, linespacing=1.12, zorder=10)


fig, ax = plt.subplots(figsize=(11.6, 6.0))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")

ax.add_patch(FancyBboxPatch((0.12, 5.55), 17.76, 3.95, boxstyle="round,pad=0.02,rounding_size=0.12",
                            lw=0.8, edgecolor="#A9C8E2", facecolor=BLUE_L, zorder=0))
ax.add_patch(FancyBboxPatch((0.12, 0.28), 17.76, 4.85, boxstyle="round,pad=0.02,rounding_size=0.12",
                            lw=0.8, edgecolor="#B8DCCB", facecolor=GREEN_L, zorder=0))
ax.text(0.5, 9.15, "(a)  Daily model retrain  —  high-performance computer", ha="left", va="center", fontsize=13.5, fontweight="bold", color=DARK)
ax.text(0.5, 4.78, "(b)  Hourly forecast  —  operational server", ha="left", va="center", fontsize=13.5, fontweight="bold", color=DARK)

# ---- (a) nightly ----
ny = 7.95
nx = [2.15, 6.55, 10.95, 15.20]
draw_database(ax, nx[0], ny, 1.2, 1.3, BLUE, "#D3E5F3")
label(ax, nx[0], 7.10, "Observation\narchive", "stage and weather records,\n2023-present", BLUE)

draw_funnel(ax, nx[1] - 0.5, ny, 0.95, 1.2, BLUE, "#D3E5F3")
draw_tensor_stack(ax, nx[1] + 0.45, ny, 0.85, 0.8, 4, BLUE, "#BFD9EC")
label(ax, nx[1], 7.10, "Clean and build\nmodel inputs", "15 min grid; quality control;\ninput features", BLUE)

gpos = [(nx[2] - 0.78, ny + 0.30), (nx[2] - 0.28, ny + 0.50), (nx[2] - 0.40, ny - 0.28), (nx[2] + 0.06, ny + 0.04)]
draw_network(ax, gpos, [(0, 1), (0, 2), (1, 3), (2, 3)], PURPLE, PURPLE, 0.10, highlighted_nodes=[3])
draw_chip(ax, nx[2] + 0.72, ny, 1.05, 1.05, BLUE, "#C8DEEF")
ax.text(nx[2] + 0.72, ny, "GRU", ha="center", va="center", fontsize=10.0, fontweight="bold", color=BLUE, zorder=9)
label(ax, nx[2], 7.10, "Train ST-GNN model\non a predefined graph", "68 gauges, 497-edge graph;\nquantile loss", BLUE)

draw_document_stack(ax, nx[3], ny, 1.0, 1.22, BLUE, "#D3E5F3")
ax.text(nx[3] + 0.07, ny, "W", ha="center", va="center", fontsize=10.5, fontweight="bold", color=BLUE, zorder=9)
label(ax, nx[3], 7.10, "Updated daily\nmodel weights", "uploaded to the\nforecast server", BLUE)

for i in range(3):
    arrow(ax, (nx[i] + 1.15, ny), (nx[i + 1] - 1.2, ny), BLUE)

# ---- (b) hourly ----
hy = 3.25
hx = [1.7, 4.6, 7.5, 10.4, 13.3, 16.1]
draw_clock(ax, hx[0] - 0.35, hy, 0.42, ORANGE, "white")
draw_database(ax, hx[0] + 0.42, hy, 0.75, 0.9, ORANGE, "#FFE5B9")
label(ax, hx[0], 2.5, "Live data\nand forecast", "last 72 h observed;\nnext 24 h weather forecast", ORANGE)

draw_funnel(ax, hx[1] - 0.5, hy, 0.9, 1.15, ORANGE, "#FFE1AD")
draw_tensor_stack(ax, hx[1] + 0.45, hy, 0.8, 0.75, 4, ORANGE, "#FFD9A0")
label(ax, hx[1], 2.5, "Clean and build\nmodel inputs", "same steps\nas training", ORANGE)

draw_chip(ax, hx[2], hy, 1.2, 1.1, GREEN, "#C9E8D9")
ax.text(hx[2], hy, "ST-GNN", ha="center", va="center", fontsize=9.5, fontweight="bold", color=GREEN, zorder=9)
label(ax, hx[2], 2.5, "Run ST-GNN\ninference forecast", "24 h ahead,\nall 68 gauges", GREEN)

# safeguards icon: a forecast curve anchored to the latest observation (continuity) + a check mark
bx, by = hx[3], hy
ax.add_patch(FancyBboxPatch((bx - 0.6, by - 0.55), 1.2, 1.1, boxstyle="round,pad=0.02,rounding_size=0.08",
                            lw=2.0, edgecolor=GREEN, facecolor="#D4EDDF", zorder=4))
import numpy as np
_t = np.linspace(0, 1, 40)
_c = by - 0.30 + 0.75 * _t ** 1.3
ax.plot(bx - 0.45 + 0.9 * _t, _c, color=GREEN, lw=1.8, zorder=6)
ax.scatter([bx - 0.45], [by - 0.30], s=20, color="#d8741c", zorder=7)
ax.plot([bx + 0.10, bx + 0.27, bx + 0.50], [by + 0.18, by + 0.02, by + 0.40], color=GREEN, lw=2.0, solid_capstyle="round", zorder=8)
label(ax, hx[3], 2.5, "Service\nsafeguards", "anchor to latest obs;\ndata + plausibility checks", GREEN)

draw_web_globe(ax, hx[4] - 0.32, hy + 0.05, 0.42, GREEN, "white")
draw_document_stack(ax, hx[4] + 0.42, hy, 0.78, 1.0, GREEN, "#D4EDDF")
label(ax, hx[4], 2.5, "Publish\nforecast", "public 15 min\nstation products", GREEN)

# score: paired eval scatter (forecast vs truth)
sx = hx[5]
ax.plot([sx - 0.5, sx - 0.5, sx + 0.55], [hy + 0.5, hy - 0.45, hy - 0.45], color=GRAY, lw=0.9, zorder=5)
ax.plot([sx - 0.5, sx + 0.5], [hy - 0.45, hy + 0.45], color=GRAY, lw=0.8, ls="--", zorder=5)
ax.scatter([sx - 0.32, sx - 0.05, sx + 0.22], [hy - 0.18, hy + 0.04, hy + 0.26], s=15, color=GREEN, zorder=7)
ax.scatter([sx - 0.3, sx + 0.0, sx + 0.28], [hy + 0.02, hy + 0.22, hy + 0.36], s=15, color=PURPLE, zorder=7)
label(ax, hx[5], 2.5, "Validate against\nobservations", "previous day's forecast\nscored when obs arrive", PURPLE)

for i in range(5):
    c = GREEN
    if i == 0:
        c = ORANGE
    if i >= 4:
        c = PURPLE
    arrow(ax, (hx[i] + 0.85, hy), (hx[i + 1] - 0.85, hy), c)

# weights flow nightly -> Run ST-GNN (label placed right where the arrow enters the hourly run)
arrow(ax, (nx[3], ny - 0.7), (hx[2] + 0.2, hy + 0.78), BLUE, lw=1.8)
ax.text(9.45, 4.34, "latest model weights", ha="left", va="center", fontsize=10.0, fontstyle="italic", color=BLUE, zorder=11)

# Bi-LSTM baseline (parallel, unpublished): a clean parallel-track card below the row, fed by two short
# dashed arrows dropped in the gaps between columns (so nothing crosses the detail text).
ax.add_patch(FancyBboxPatch((4.55, 0.34), 9.0, 0.70, boxstyle="round,pad=0.03,rounding_size=0.08",
                            lw=1.2, linestyle="--", edgecolor=PURPLE, facecolor=PURPLE_L, zorder=4))
ax.text(9.05, 0.69, "Previous Bi-LSTM model  —  parallel hourly baseline (not published)",
        ha="center", va="center", fontsize=9.0, fontweight="bold", color=PURPLE, zorder=7)
arrow(ax, (6.05, 1.20), (6.05, 1.06), PURPLE, dashed=True, lw=1.1)
arrow(ax, (11.85, 1.20), (11.85, 1.06), PURPLE, dashed=True, lw=1.1)

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
fig.savefig(OUT + ".png", dpi=400, bbox_inches="tight", facecolor="white")
fig.savefig(OUT + ".pdf", bbox_inches="tight", facecolor="white")
print("[SAVED]", OUT + ".png")
