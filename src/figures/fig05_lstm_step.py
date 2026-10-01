"""Figure 5 for revision 6: one forecast step of the LSTM, drawn in the graphics of Figure 4.

Farid, 16 September 2026: "make figure 5 ... verbatim like Figure 4: Operational ST-GNN workflow (b figure 5b)
(with graphics, etc etc thanks). NO need to discuss st gnn in figure 5".
Round four: "tighten figure 5 bro to many white spaces!" The panel now closes around the row of boxes.

The drawing vocabulary is taken from the native Figure 4 recipe, not re-invented: this script executes the
header of that recipe (its card style, its palette, and the helpers add_arrow, add_panel, add_box, draw_tensor)
and then draws one panel with the same box sizes, the same eight column positions, the same arrow style and the
same type sizes. Only the contents change, from the ST-GNN to the LSTM.

Recipe:
  src/figures/recipe_training_inference_cycle.py
The recipe is read up to the line that creates its own canvas, so nothing of the original figure is drawn or
overwritten.

Output:
  project/manuscript/20260915_Revision_6/figures/fig05_lstm_step.png
Run with: conda run -n operational python fig05_lstm_step.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

PAPER = "project/manuscript/"
RECIPE = "src/figures/recipe_training_inference_cycle.py"
REV6 = PAPER + "20260915_Revision_6"
FIGURES = os.path.join(REV6, "figures")
STEM = os.path.join(FIGURES, "fig05_lstm_step")
os.makedirs(FIGURES, exist_ok=True)

CANVAS_LINE = "figure, axis = plt.subplots(figsize=(7.15, 8.65))"

source = open(RECIPE, encoding="utf-8").read()
if source.count(CANVAS_LINE) != 1:
    raise Exception("expected exactly one canvas line in the recipe")
header = source.split(CANVAS_LINE)[0]
recipe = {"__file__": RECIPE, "__name__": "rev6_fig05_header"}
exec(compile(header, RECIPE, "exec"), recipe)
print("[VERIFY] loaded the Figure 4 palette and helpers from", RECIPE, flush=True)

add_arrow = recipe["add_arrow"]
add_box = recipe["add_box"]
draw_tensor = recipe["draw_tensor"]
BLUE = recipe["BLUE"]
BLUE_LIGHT = recipe["BLUE_LIGHT"]
ORANGE = recipe["ORANGE"]
ORANGE_LIGHT = recipe["ORANGE_LIGHT"]
GREEN = recipe["GREEN"]
GREEN_LIGHT = recipe["GREEN_LIGHT"]
PURPLE = recipe["PURPLE"]
PURPLE_LIGHT = recipe["PURPLE_LIGHT"]
RED = recipe["RED"]
DARK = recipe["DARK"]
LIGHT_GRAY = recipe["LIGHT_GRAY"]
BORDER = recipe["BORDER"]

# The canvas closes around the panel: the row of boxes sits at ROW_Y and everything else is placed from it.
PANEL_X = 0.18
PANEL_Y = 0.20
PANEL_WIDTH = 13.94
PANEL_HEIGHT = 3.66
ROW_Y = 1.83
LABEL_ABOVE = ROW_Y + 0.82
LABEL_BELOW = ROW_Y - 0.92

figure, axis = plt.subplots(figsize=(7.15, 2.08))
axis.set_xlim(0.0, 14.3)
axis.set_ylim(0.0, PANEL_Y + PANEL_HEIGHT + 0.16)
axis.axis("off")

axis.add_patch(FancyBboxPatch((PANEL_X, PANEL_Y), PANEL_WIDTH, PANEL_HEIGHT,
                              boxstyle="round,pad=0.015,rounding_size=0.10", linewidth=0.9,
                              edgecolor=BORDER, facecolor="white", zorder=0))
axis.text(PANEL_X + 0.20, PANEL_Y + PANEL_HEIGHT - 0.18, "One forecast step of the LSTM", ha="left", va="top",
          fontsize=10.2, fontweight="bold", color=DARK, zorder=20)
cadence_box = FancyBboxPatch((PANEL_X + PANEL_WIDTH - 2.82, PANEL_Y + PANEL_HEIGHT - 0.54), 2.45, 0.34,
                             boxstyle="round,pad=0.01,rounding_size=0.08", linewidth=0.9, edgecolor=ORANGE,
                             facecolor="white", zorder=7)
axis.add_patch(cadence_box)
axis.text(PANEL_X + PANEL_WIDTH - 1.595, PANEL_Y + PANEL_HEIGHT - 0.37, "one gauge at a time", ha="center",
          va="center", fontsize=7.1, fontweight="bold", color=ORANGE, zorder=10)

column_x = [1.15, 3.05, 4.85, 6.65, 8.45, 10.25, 11.75, 13.20]

print("[DRAW] Inputs and preprocessing.", flush=True)
add_box(axis, column_x[0], ROW_Y, 1.55, 1.32, "Live pull", "72 h observations\n24 h rain forecast", ORANGE,
        ORANGE_LIGHT, title_size=7.6, detail_size=6.3)
add_box(axis, column_x[1], ROW_Y, 1.55, 1.32, "Preprocess", "same features\nsaved scalers", BLUE, LIGHT_GRAY,
        title_size=7.6, detail_size=6.3)

print("[DRAW] History tensor for one gauge.", flush=True)
draw_tensor(axis, column_x[2], ROW_Y, BLUE, BLUE_LIGHT, scale=0.88)
axis.text(column_x[2], LABEL_ABOVE, "History tensor", ha="center", va="center", fontsize=7.4, fontweight="bold",
          color=BLUE)
axis.text(column_x[2], LABEL_BELOW, "[B, 288, 1, 9]", ha="center", va="center", fontsize=6.5, color=DARK)

print("[DRAW] Shared LSTM encoder.", flush=True)
encoder_box = FancyBboxPatch((5.88, ROW_Y - 0.68), 1.48, 1.36, boxstyle="round,pad=0.02,rounding_size=0.07",
                             linewidth=1.1, edgecolor=BLUE, facecolor=BLUE_LIGHT, zorder=5)
axis.add_patch(encoder_box)
for encoder_row in range(3):
    encoder_y = ROW_Y + 0.38 - encoder_row * 0.38
    axis.add_patch(Circle((6.25, encoder_y), 0.08, edgecolor=BLUE, facecolor="white", linewidth=0.9, zorder=7))
    axis.add_patch(Circle((6.95, encoder_y), 0.10, edgecolor=BLUE, facecolor=BLUE, linewidth=0.9, zorder=7))
    add_arrow(axis, (6.36, encoder_y), (6.81, encoder_y), BLUE, line_width=0.9, mutation_scale=6)
axis.text(column_x[3], LABEL_ABOVE, "Shared LSTM", ha="center", va="center", fontsize=7.4, fontweight="bold",
          color=BLUE)
axis.text(column_x[3], LABEL_BELOW, "encoder; 64 units", ha="center", va="center", fontsize=6.5, color=DARK)

print("[DRAW] Hidden state carried to the decoder.", flush=True)
draw_tensor(axis, column_x[4], ROW_Y, PURPLE, PURPLE_LIGHT, scale=0.70)
axis.text(column_x[4], LABEL_ABOVE, "Hidden state", ha="center", va="center", fontsize=7.4, fontweight="bold",
          color=PURPLE)
axis.text(column_x[4], LABEL_BELOW, "64 values; one gauge", ha="center", va="center", fontsize=6.5, color=DARK)

print("[DRAW] LSTM decoder steps.", flush=True)
for decoder_index in range(3):
    decoder_x = column_x[5] - 0.48 + decoder_index * 0.48
    axis.add_patch(Circle((decoder_x, ROW_Y), 0.28, edgecolor=GREEN, facecolor="white", linewidth=1.1, zorder=6))
    axis.text(decoder_x, ROW_Y, f"t{decoder_index + 1}", ha="center", va="center", fontsize=6.5,
              fontweight="bold", color=GREEN, zorder=8)
    if decoder_index < 2:
        add_arrow(axis, (decoder_x + 0.29, ROW_Y), (decoder_x + 0.43, ROW_Y), GREEN, line_width=0.9,
                  mutation_scale=6)
axis.text(column_x[5], LABEL_ABOVE, "LSTM decoder", ha="center", va="center", fontsize=7.4, fontweight="bold",
          color=GREEN)
axis.text(column_x[5], LABEL_BELOW, "96 lead steps", ha="center", va="center", fontsize=6.5, color=DARK)

print("[DRAW] Quantile outputs.", flush=True)
quantile_offsets = [-0.34, -0.17, 0.0, 0.17, 0.34]
quantile_colors = [PURPLE, BLUE, GREEN, ORANGE, RED]
for quantile_index, quantile_offset in enumerate(quantile_offsets):
    curve_x = np.linspace(column_x[6] - 0.52, column_x[6] + 0.52, 50)
    curve_y = ROW_Y + quantile_offset + 0.13 * np.tanh((curve_x - column_x[6]) * 2.5)
    quantile_width = 2.0 if quantile_index == 3 else 1.0
    axis.plot(curve_x, curve_y, color=quantile_colors[quantile_index], linewidth=quantile_width, zorder=7)
axis.text(column_x[6], LABEL_ABOVE, "Quantiles", ha="center", va="center", fontsize=7.4, fontweight="bold",
          color=ORANGE)
axis.text(column_x[6], LABEL_BELOW, "P80 selected", ha="center", va="center", fontsize=6.5, color=DARK)

add_box(axis, column_x[7], ROW_Y, 1.45, 1.32, "Forecast", "one gauge\n96 steps of 15 min", GREEN, GREEN_LIGHT,
        title_size=7.6, detail_size=6.2)

arrow_points = [
    (1.95, 2.24, ORANGE),
    (3.84, 4.08, BLUE),
    (5.55, 5.84, BLUE),
    (7.42, 7.72, PURPLE),
    (9.24, 9.53, GREEN),
    (10.98, 11.18, ORANGE),
    (12.32, 12.44, GREEN),
]
for start_x, end_x, arrow_color in arrow_points:
    add_arrow(axis, (start_x, ROW_Y), (end_x, ROW_Y), arrow_color, line_width=1.2, mutation_scale=8)

axis.add_patch(FancyBboxPatch((4.08, ROW_Y - 1.40), 8.22, 2.62, boxstyle="round,pad=0.01,rounding_size=0.08",
                              linewidth=0.9, edgecolor=BLUE, facecolor="none", linestyle="--", zorder=2))
axis.text(8.19, ROW_Y - 1.26, "one set of weights serves every gauge; no information passes between gauges",
          ha="center", va="center", fontsize=6.7, fontstyle="italic", color=BLUE)

figure.subplots_adjust(left=0.008, right=0.992, bottom=0.010, top=0.990)
for suffix in (".png", ".pdf"):
    figure.savefig(STEM + suffix, dpi=300, bbox_inches="tight", facecolor="white")
    print("[SAVED]", STEM + suffix, flush=True)
plt.close(figure)
print("REV6_FIG05_LSTM_DONE", flush=True)
