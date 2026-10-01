"""Create an operational-cadence version of the combined ST-GNN methods figure.

The three panels explain:
  (a) daily model retraining and controlled deployment,
  (b) hourly inference and forecast publication, and
  (c) rolling forecast windows with successive forecast hydrographs.

Outputs:
  figure_library_outputs/fig05_daily_training_hourly_inference.png
  figure_library_outputs/fig05_daily_training_hourly_inference.pdf
  figure_library_outputs/fig05_daily_training_hourly_inference.svg
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Rectangle


SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIRECTORY = "project/manuscript/figure_library_outputs"

sys.path.insert(0, SCRIPT_DIRECTORY)

from _card import apply


apply()

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

PNG_PATH = os.path.join(OUTPUT_DIRECTORY, "fig05_daily_training_hourly_inference.png")
PDF_PATH = os.path.join(OUTPUT_DIRECTORY, "fig05_daily_training_hourly_inference.pdf")
SVG_PATH = os.path.join(OUTPUT_DIRECTORY, "fig05_daily_training_hourly_inference.svg")

print(f"[CONFIG] Current working directory: {os.getcwd()}")
print(f"[CONFIG] Script directory: {SCRIPT_DIRECTORY}")
print(f"[CONFIG] Output directory: {OUTPUT_DIRECTORY}")
print("[DESIGN] Panel (a) explicitly represents daily retraining.")
print("[DESIGN] Panel (b) explicitly represents hourly inference.")
print("[DESIGN] Panel (c) includes sliding windows and forecast hydrograph curves.")

BLUE = "#2F6FAE"
BLUE_LIGHT = "#DCEBF6"
ORANGE = "#D58200"
ORANGE_LIGHT = "#FFE7BA"
GREEN = "#17875B"
GREEN_LIGHT = "#D8EFE3"
PURPLE = "#6A4195"
PURPLE_LIGHT = "#E9DFF1"
RED = "#B84435"
RED_LIGHT = "#F7DDD8"
DARK = "#252525"
GRAY = "#666666"
MID_GRAY = "#9A9A9A"
LIGHT_GRAY = "#F4F5F6"
BORDER = "#C9CED4"


def add_arrow(axis, start, end, color, line_width=1.4, mutation_scale=9, line_style="-"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=line_width,
        linestyle=line_style,
        color=color,
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=15,
    )
    axis.add_patch(arrow)


def add_panel(axis, x, y, width, height, label, title, cadence, cadence_color):
    panel = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.10",
        linewidth=0.9,
        edgecolor=BORDER,
        facecolor="white",
        zorder=0,
    )
    axis.add_patch(panel)
    axis.text(
        x + 0.20,
        y + height - 0.20,
        f"({label})",
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=DARK,
        zorder=20,
    )
    axis.text(
        x + 0.67,
        y + height - 0.20,
        title,
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=DARK,
        zorder=20,
    )
    cadence_box = FancyBboxPatch(
        (x + width - 2.82, y + height - 0.58),
        2.45,
        0.34,
        boxstyle="round,pad=0.01,rounding_size=0.08",
        linewidth=0.9,
        edgecolor=cadence_color,
        facecolor="white",
        zorder=7,
    )
    axis.add_patch(cadence_box)
    axis.text(
        x + width - 1.595,
        y + height - 0.41,
        cadence,
        ha="center",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color=cadence_color,
        zorder=10,
    )


def add_box(axis, center_x, center_y, width, height, title, detail, edge, fill, title_size=7.7, detail_size=6.6):
    box = FancyBboxPatch(
        (center_x - width / 2.0, center_y - height / 2.0),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.07",
        linewidth=1.1,
        edgecolor=edge,
        facecolor=fill,
        zorder=5,
    )
    axis.add_patch(box)
    axis.text(
        center_x,
        center_y + 0.19,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=edge,
        linespacing=0.96,
        zorder=10,
    )
    axis.text(
        center_x,
        center_y - 0.28,
        detail,
        ha="center",
        va="center",
        fontsize=detail_size,
        color=DARK,
        linespacing=1.05,
        zorder=10,
    )


def draw_network(axis, center_x, center_y, scale, new_node=False):
    positions = [
        (-0.68, 0.30),
        (-0.20, 0.67),
        (0.44, 0.50),
        (-0.54, -0.35),
        (0.08, -0.05),
        (0.63, -0.30),
    ]
    edges = [
        (0, 1),
        (0, 3),
        (1, 2),
        (1, 4),
        (2, 5),
        (3, 4),
        (4, 5),
    ]
    for start_index, end_index in edges:
        start_x = center_x + positions[start_index][0] * scale
        start_y = center_y + positions[start_index][1] * scale
        end_x = center_x + positions[end_index][0] * scale
        end_y = center_y + positions[end_index][1] * scale
        axis.plot(
            [start_x, end_x],
            [start_y, end_y],
            color=PURPLE,
            linewidth=0.9,
            alpha=0.50,
            zorder=4,
        )
    for node_index, position in enumerate(positions):
        node_color = PURPLE
        if new_node and node_index == len(positions) - 1:
            node_color = ORANGE
        node = Circle(
            (
                center_x + position[0] * scale,
                center_y + position[1] * scale,
            ),
            0.13 * scale,
            linewidth=0.8,
            edgecolor=DARK,
            facecolor=node_color,
            zorder=7,
        )
        axis.add_patch(node)


def draw_tensor(axis, center_x, center_y, color, fill, scale=1.0):
    for layer_index in range(4):
        offset = layer_index * 0.07 * scale
        polygon_x = [
            center_x - 0.43 * scale + offset,
            center_x + 0.43 * scale + offset,
            center_x + 0.28 * scale + offset,
            center_x - 0.58 * scale + offset,
            center_x - 0.43 * scale + offset,
        ]
        polygon_y = [
            center_y + 0.20 * scale + offset,
            center_y + 0.20 * scale + offset,
            center_y - 0.20 * scale + offset,
            center_y - 0.20 * scale + offset,
            center_y + 0.20 * scale + offset,
        ]
        axis.fill(
            polygon_x,
            polygon_y,
            facecolor=fill,
            edgecolor=color,
            linewidth=0.8,
            alpha=0.95,
            zorder=5 + layer_index,
        )


figure, axis = plt.subplots(figsize=(7.15, 8.65))
axis.set_xlim(0.0, 14.3)
axis.set_ylim(0.0, 17.3)
axis.axis("off")

print("[DRAW] Panel (a): daily retraining and deployment.")
add_panel(
    axis,
    0.18,
    11.78,
    13.94,
    5.30,
    "a",
    "Daily model retraining and controlled deployment",
    "once per day",
    BLUE,
)

training_y = 14.55
training_x = [1.30, 4.00, 6.85, 9.65, 12.45]
training_half_widths = [1.00, 1.075, 1.30, 1.075, 1.025]

add_box(axis, training_x[0], training_y, 2.00, 1.45, "Verified archive", "Ascension Parish Data\nUSGS + NOAA\n2023 to present", BLUE, BLUE_LIGHT, detail_size=6.3)
add_box(axis, training_x[1], training_y, 2.15, 1.45, "QC + preprocessing", "NAVD88; quality control\n15 min grid", BLUE, LIGHT_GRAY)
add_box(axis, training_x[2], training_y, 2.60, 1.45, "Training matrix", "68 gauges; 72 h windows\n9 features: stage, rainfall,\nAPI, tendency, and\nrainfall interactions", BLUE, LIGHT_GRAY, detail_size=6.0)
add_box(axis, training_x[3], training_y, 2.15, 1.45, "ST-GNN training", "shared encoder + graph\nP50 to P90 loss", PURPLE, PURPLE_LIGHT)
add_box(axis, training_x[4], training_y, 2.05, 1.45, "Deploy assets", "trained weights + scalers\nto forecast server", GREEN, GREEN_LIGHT)

for training_index in range(len(training_x) - 1):
    arrow_color = BLUE
    if training_index == 2:
        arrow_color = PURPLE
    if training_index >= 3:
        arrow_color = GREEN
    add_arrow(
        axis,
        (training_x[training_index] + training_half_widths[training_index], training_y),
        (training_x[training_index + 1] - training_half_widths[training_index + 1], training_y),
        arrow_color,
    )

axis.text(
    0.55,
    13.30,
    "Network expansion scenario: new gauge installation",
    ha="left",
    va="center",
    fontsize=7.4,
    fontweight="bold",
    color=GREEN,
)
draw_network(axis, 3.32, 12.58, 0.72, new_node=True)
axis.text(3.32, 11.98, "new gauge = new node", ha="center", va="center", fontsize=6.7, color=DARK)
add_arrow(axis, (3.95, 12.58), (4.46, 12.58), GREEN, line_width=1.2)

mask_x = 4.58
mask_y = 12.28
mask_width = 3.15
mask_height = 0.60
axis.add_patch(Rectangle((mask_x, mask_y), mask_width * 0.52, mask_height, facecolor=RED_LIGHT, edgecolor=RED, linewidth=0.9, zorder=5))
axis.add_patch(Rectangle((mask_x + mask_width * 0.52, mask_y), mask_width * 0.48, mask_height, facecolor=GREEN_LIGHT, edgecolor=GREEN, linewidth=0.9, zorder=5))
axis.text(mask_x + mask_width * 0.26, mask_y + mask_height / 2.0, "before installation\nloss = 0", ha="center", va="center", fontsize=6.5, fontweight="bold", color=RED, zorder=8)
axis.text(mask_x + mask_width * 0.76, mask_y + mask_height / 2.0, "after installation\nloss counted", ha="center", va="center", fontsize=6.5, fontweight="bold", color=GREEN, zorder=8)
axis.plot([mask_x + mask_width * 0.52, mask_x + mask_width * 0.52], [mask_y - 0.12, mask_y + mask_height + 0.12], color=DARK, linewidth=1.0, zorder=9)

add_arrow(axis, (7.85, 12.58), (8.58, 12.58), GREEN, line_width=1.2)
add_box(axis, 9.73, 12.58, 2.45, 1.05, "Shared parameters", "same GRU weights\nfor all nodes", GREEN, GREEN_LIGHT, title_size=7.5, detail_size=6.3)
add_arrow(axis, (10.98, 12.58), (11.48, 12.58), GREEN, line_width=1.2)
draw_tensor(axis, 12.35, 12.58, GREEN, GREEN_LIGHT, scale=0.92)
axis.text(12.42, 11.98, "one gauge-indexed matrix", ha="center", va="center", fontsize=6.7, fontweight="bold", color=GREEN)

print("[DRAW] Panel (b): hourly inference and publication.")
add_panel(
    axis,
    0.18,
    6.20,
    13.94,
    5.28,
    "b",
    "Hourly inference and forecast publication",
    "every hour",
    ORANGE,
)

inference_y = 8.80
inference_x = [1.15, 3.05, 4.85, 6.65, 8.45, 10.25, 11.75, 13.20]

add_box(axis, inference_x[0], inference_y, 1.55, 1.32, "Live pull", "72 h observations\n24 h rain forecast", ORANGE, ORANGE_LIGHT, title_size=7.6, detail_size=6.3)
add_box(axis, inference_x[1], inference_y, 1.55, 1.32, "Preprocess", "same features\nsaved scalers", BLUE, LIGHT_GRAY, title_size=7.6, detail_size=6.3)

draw_tensor(axis, inference_x[2], inference_y, BLUE, BLUE_LIGHT, scale=0.88)
axis.text(inference_x[2], 9.62, "History tensor", ha="center", va="center", fontsize=7.4, fontweight="bold", color=BLUE)
axis.text(inference_x[2], 7.88, "[B, 288, 68, 9]", ha="center", va="center", fontsize=6.5, color=DARK)

encoder_box = FancyBboxPatch((5.88, 8.12), 1.48, 1.36, boxstyle="round,pad=0.02,rounding_size=0.07", linewidth=1.1, edgecolor=BLUE, facecolor=BLUE_LIGHT, zorder=5)
axis.add_patch(encoder_box)
for encoder_row in range(3):
    encoder_y = 9.18 - encoder_row * 0.38
    axis.add_patch(Circle((6.25, encoder_y), 0.08, edgecolor=BLUE, facecolor="white", linewidth=0.9, zorder=7))
    axis.add_patch(Circle((6.95, encoder_y), 0.10, edgecolor=BLUE, facecolor=BLUE, linewidth=0.9, zorder=7))
    add_arrow(axis, (6.36, encoder_y), (6.81, encoder_y), BLUE, line_width=0.9, mutation_scale=6)
axis.text(inference_x[3], 9.62, "Shared GRU", ha="center", va="center", fontsize=7.4, fontweight="bold", color=BLUE)
axis.text(inference_x[3], 7.88, "node encoder", ha="center", va="center", fontsize=6.5, color=DARK)

draw_network(axis, inference_x[4], inference_y, 0.83, new_node=False)
axis.text(inference_x[4], 9.62, "Graph mixing", ha="center", va="center", fontsize=7.4, fontweight="bold", color=PURPLE)
axis.text(inference_x[4], 7.88, "68 nodes; 497 edges", ha="center", va="center", fontsize=6.5, color=DARK)

for decoder_index in range(3):
    decoder_x = inference_x[5] - 0.48 + decoder_index * 0.48
    axis.add_patch(Circle((decoder_x, inference_y), 0.28, edgecolor=GREEN, facecolor="white", linewidth=1.1, zorder=6))
    axis.text(decoder_x, inference_y, f"t{decoder_index + 1}", ha="center", va="center", fontsize=6.5, fontweight="bold", color=GREEN, zorder=8)
    if decoder_index < 2:
        add_arrow(axis, (decoder_x + 0.29, inference_y), (decoder_x + 0.43, inference_y), GREEN, line_width=0.9, mutation_scale=6)
axis.text(inference_x[5], 9.62, "Graph-GRU", ha="center", va="center", fontsize=7.4, fontweight="bold", color=GREEN)
axis.text(inference_x[5], 7.88, "96 lead steps", ha="center", va="center", fontsize=6.5, color=DARK)

quantile_offsets = [-0.34, -0.17, 0.0, 0.17, 0.34]
quantile_colors = [PURPLE, BLUE, GREEN, ORANGE, RED]
for quantile_index, quantile_offset in enumerate(quantile_offsets):
    curve_x = np.linspace(inference_x[6] - 0.52, inference_x[6] + 0.52, 50)
    curve_y = inference_y + quantile_offset + 0.13 * np.tanh((curve_x - inference_x[6]) * 2.5)
    quantile_width = 2.0 if quantile_index == 3 else 1.0
    axis.plot(curve_x, curve_y, color=quantile_colors[quantile_index], linewidth=quantile_width, zorder=7)
axis.text(inference_x[6], 9.62, "Quantiles", ha="center", va="center", fontsize=7.4, fontweight="bold", color=ORANGE)
axis.text(inference_x[6], 7.88, "P80 selected", ha="center", va="center", fontsize=6.5, color=DARK)

add_box(axis, inference_x[7], inference_y, 1.45, 1.32, "Publish", "anchor + safeguards\nproducts + archive", GREEN, GREEN_LIGHT, title_size=7.6, detail_size=6.2)

inference_arrow_points = [
    (1.95, 2.24, ORANGE),
    (3.84, 4.08, BLUE),
    (5.55, 5.84, BLUE),
    (7.42, 7.72, PURPLE),
    (9.24, 9.53, GREEN),
    (10.98, 11.18, ORANGE),
    (12.32, 12.44, GREEN),
]
for start_x, end_x, arrow_color in inference_arrow_points:
    add_arrow(axis, (start_x, inference_y), (end_x, inference_y), arrow_color, line_width=1.2, mutation_scale=8)

axis.add_patch(FancyBboxPatch((4.08, 7.28), 8.22, 2.75, boxstyle="round,pad=0.01,rounding_size=0.08", linewidth=0.9, edgecolor=GREEN, facecolor="none", linestyle="--", zorder=2))
axis.text(8.19, 7.43, "active ST-GNN weights deployed by the daily cycle", ha="center", va="center", fontsize=6.7, fontstyle="italic", color=GREEN)

axis.plot(
    [13.49, 13.95, 13.95, 12.65],
    [14.55, 14.55, 10.45, 10.45],
    color=GREEN,
    linewidth=1.0,
    linestyle="--",
    zorder=12,
)
add_arrow(axis, (12.65, 10.45), (12.20, 10.04), GREEN, line_width=1.0, mutation_scale=8, line_style="--")

print("[DRAW] Panel (c): rolling windows and forecast curves.")
add_panel(
    axis,
    0.18,
    0.25,
    13.94,
    5.65,
    "c",
    "Rolling forecast origins and successive hydrographs",
    "live forecast delivery",
    GREEN,
)

axis.text(3.80, 4.95, "Sliding input and output windows", ha="center", va="center", fontsize=7.8, fontweight="bold", color=DARK)

window_left = 1.55
window_origin = 5.30
window_right = 7.15
window_rows = [4.05, 3.25, 2.45]
window_shifts = [-0.18, 0.0, 0.18]
window_labels = ["origin t0 - 1 h", "origin t0", "origin t0 + 1 h"]

for window_index, window_y in enumerate(window_rows):
    shifted_left = window_left + window_shifts[window_index]
    shifted_origin = window_origin + window_shifts[window_index]
    shifted_right = window_right + window_shifts[window_index]
    axis.add_patch(Rectangle((shifted_left, window_y - 0.22), shifted_origin - shifted_left, 0.44, facecolor=BLUE_LIGHT, edgecolor=GRAY, linewidth=0.8, zorder=4))
    axis.add_patch(Rectangle((shifted_origin, window_y - 0.22), shifted_right - shifted_origin, 0.44, facecolor=ORANGE_LIGHT, edgecolor=GRAY, linewidth=0.8, zorder=4))
    axis.plot([shifted_origin, shifted_origin], [window_y - 0.34, window_y + 0.34], color=DARK, linewidth=0.9, linestyle="--", zorder=7)
    axis.add_patch(Circle((shifted_origin, window_y), 0.10, edgecolor=DARK, facecolor="white", linewidth=0.9, zorder=8))
    axis.plot([shifted_origin - 0.07, shifted_origin + 0.07], [window_y - 0.07, window_y + 0.07], color=DARK, linewidth=0.8, zorder=9)
    axis.plot([shifted_origin - 0.07, shifted_origin + 0.07], [window_y + 0.07, window_y - 0.07], color=DARK, linewidth=0.8, zorder=9)
    axis.text(1.25, window_y, window_labels[window_index], ha="right", va="center", fontsize=6.6, fontstyle="italic", color=DARK)

axis.text(3.23, 4.54, "72 h observed", ha="center", va="center", fontsize=7.0, fontweight="bold", color=BLUE)
axis.text(6.08, 4.54, "24 h forecast", ha="center", va="center", fontsize=7.0, fontweight="bold", color=ORANGE)
axis.plot([1.35, 7.45], [1.72, 1.72], color=GRAY, linewidth=0.9, zorder=3)
for tick_x, tick_label in [(window_left, "t0 - 72 h"), (window_origin, "t0"), (window_right, "t0 + 24 h")]:
    axis.plot([tick_x, tick_x], [1.63, 1.81], color=GRAY, linewidth=0.9, zorder=3)
    axis.text(tick_x, 1.47, tick_label, ha="center", va="top", fontsize=6.5, color=GRAY)

axis.plot([7.57, 7.57], [1.22, 4.95], color=BORDER, linewidth=0.8, zorder=2)
axis.text(10.72, 4.95, "Observed stage and successive forecasts", ha="center", va="center", fontsize=7.8, fontweight="bold", color=DARK)

curve_x_left = 7.95
curve_x_origin = 10.76
curve_x_right = 13.55
curve_y_bottom = 1.48
curve_y_top = 4.50

axis.add_patch(Rectangle((curve_x_left, curve_y_bottom), curve_x_origin - curve_x_left, curve_y_top - curve_y_bottom, facecolor=BLUE_LIGHT, edgecolor="none", alpha=0.62, zorder=1))
axis.add_patch(Rectangle((curve_x_origin, curve_y_bottom), curve_x_right - curve_x_origin, curve_y_top - curve_y_bottom, facecolor=ORANGE_LIGHT, edgecolor="none", alpha=0.55, zorder=1))
axis.plot([curve_x_origin, curve_x_origin], [curve_y_bottom, curve_y_top], color=GRAY, linewidth=1.0, linestyle="--", zorder=5)

observed_x = np.linspace(curve_x_left + 0.12, curve_x_origin, 140)
observed_fraction = (observed_x - observed_x.min()) / (observed_x.max() - observed_x.min())
observed_y = 1.85 + 0.28 * observed_fraction + 1.55 / (1.0 + np.exp(-11.0 * (observed_fraction - 0.72)))
observed_y = observed_y - 0.34 * np.exp(-((observed_fraction - 0.93) / 0.07) ** 2)
axis.plot(observed_x, observed_y, color=DARK, linewidth=2.1, zorder=8)

forecast_x = np.linspace(curve_x_origin, curve_x_right - 0.12, 105)
forecast_fraction = (forecast_x - forecast_x.min()) / (forecast_x.max() - forecast_x.min())
forecast_origins = [observed_y[-1] - 0.18, observed_y[-1] - 0.05, observed_y[-1] + 0.10]
forecast_colors = ["#A5A5A5", "#777777", GREEN]
forecast_styles = ["--", "--", "-"]
forecast_widths = [1.1, 1.3, 2.2]

for forecast_index in range(3):
    forecast_y = forecast_origins[forecast_index] + 0.72 * np.sin(np.pi * forecast_fraction) + 0.26 * forecast_fraction
    forecast_y = forecast_y - 0.10 * forecast_index * forecast_fraction
    axis.plot(
        forecast_x,
        forecast_y,
        color=forecast_colors[forecast_index],
        linewidth=forecast_widths[forecast_index],
        linestyle=forecast_styles[forecast_index],
        zorder=7 + forecast_index,
    )

axis.scatter([curve_x_origin], [observed_y[-1]], s=20, color=DARK, zorder=11)
axis.text(curve_x_origin, 4.62, "origin t0", ha="center", va="bottom", fontsize=6.7, fontstyle="italic", color=DARK)
axis.text(9.30, 1.67, "observed", ha="center", va="center", fontsize=6.6, fontweight="bold", color=BLUE)
axis.text(12.16, 1.67, "forecast", ha="center", va="center", fontsize=6.6, fontweight="bold", color=ORANGE)

legend_y = 1.15
axis.plot([8.05, 8.48], [legend_y, legend_y], color=DARK, linewidth=2.0, zorder=8)
axis.text(8.57, legend_y, "observed", ha="left", va="center", fontsize=6.3, color=DARK)
axis.plot([9.62, 10.05], [legend_y, legend_y], color=MID_GRAY, linewidth=1.2, linestyle="--", zorder=8)
axis.text(10.14, legend_y, "earlier forecasts", ha="left", va="center", fontsize=6.3, color=DARK)
axis.plot([11.82, 12.25], [legend_y, legend_y], color=GREEN, linewidth=2.2, zorder=8)
axis.text(12.34, legend_y, "current P80", ha="left", va="center", fontsize=6.3, color=DARK)

figure.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.985)

print(f"[SAVE] Writing PNG: {PNG_PATH}")
figure.savefig(PNG_PATH, dpi=500, bbox_inches="tight", facecolor="white")
print(f"[SAVE] Writing PDF: {PDF_PATH}")
figure.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
print(f"[SAVE] Writing SVG: {SVG_PATH}")
figure.savefig(SVG_PATH, bbox_inches="tight", facecolor="white")

for output_path in [PNG_PATH, PDF_PATH, SVG_PATH]:
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Expected output was not created: {output_path}")
    output_size = os.path.getsize(output_path)
    if output_size <= 0:
        raise RuntimeError(f"Output file is empty: {output_path}")
    print(f"[VERIFY] Created {output_path} ({output_size:,} bytes)")

print("[SUCCESS] Daily-training and hourly-inference figure created successfully.")
plt.show()
