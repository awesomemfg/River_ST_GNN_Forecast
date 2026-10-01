"""Figures 2 and 4 for revision 5 (System B): the published workflow diagrams with nightly training removed.

System B trains each model once: the epoch count is fixed on a chronological hold-out (fit 2023 to 2024,
validate 2025), then the model is refitted from a new random initialization on all data through 31 December 2025
with the 471-edge graph built from observations through the same date, and frozen.

Preservation rule: run the authoritative recipes after exact, counted text substitutions.
  Figure 2  src/figures/recipe_workflow_base.py
            (with the two CPU-chip replacements of
            src/figures/recipe_workflow.py,
            the version in the manuscript)
  Figure 4  src/figures/recipe_training_inference_cycle.py
The substitutions are those of revision 4 (trained once) plus the System B graph size (471 edges) and, in
Figure 4, the epoch-selection line of the training card. No geometry, colors, icons or panels change.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

RECIPES = "src/figures/"
OUTPUT_DIRECTORY = ("project/manuscript/"
                    "20260914_System_B_Revision_5/figures")
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
FIG02_STEM = os.path.join(OUTPUT_DIRECTORY, "fig02_workflow_system_b")


def draw_model_box_no_cpu(axis, center_x, center_y, width, height, color_value, face_value):
    """The original chip body without processor pins (as in the manuscript's Figure 2)."""
    model_box = FancyBboxPatch((center_x - width / 2.0, center_y - height / 2.0), width, height,
                               boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=2.0,
                               edgecolor=color_value, facecolor=face_value, zorder=5)
    axis.add_patch(model_box)


JOBS = [
    ("recipe_workflow_base.py", [
        ('OUT = "project/manuscript/figure_library_outputs/fig01_workflow"',
         'OUT = "' + FIG02_STEM + '"', "output path"),
        ('draw_chip(ax, nx[2] + 0.72, ny, 1.05, 1.05, BLUE, "#C8DEEF")',
         'draw_model_box_no_cpu(ax, nx[2] + 0.72, ny, 1.05, 1.05, BLUE, "#C8DEEF")', "GRU icon"),
        ('draw_chip(ax, hx[2], hy, 1.2, 1.1, GREEN, "#C9E8D9")',
         'draw_model_box_no_cpu(ax, hx[2], hy, 1.2, 1.1, GREEN, "#C9E8D9")', "ST-GNN icon"),
        ('"(a)  Daily model retrain  —  high-performance computer"',
         '"(a)  One-time model training  —  high-performance computer"', "panel (a) title"),
        ('"stage and weather records,\\n2023-present"',
         '"stage and weather records,\\n1 Jan 2023 to 31 Dec 2025"', "archive period"),
        ('"68 gauges, 497-edge graph;\\nquantile loss"',
         '"68 gauges, 471-edge graph;\\nquantile loss"', "graph size"),
        ('"Updated daily\\nmodel weights", "uploaded to the\\nforecast server"',
         '"Frozen\\nmodel weights", "copied once to the\\nforecast server"', "weights card"),
        ('"latest model weights"', '"frozen model weights"', "weights arrow label"),
    ], [FIG02_STEM + ".png", FIG02_STEM + ".pdf"]),
    ("recipe_training_inference_cycle.py", [
        ('OUTPUT_DIRECTORY = "project/manuscript/figure_library_outputs"',
         'OUTPUT_DIRECTORY = "' + OUTPUT_DIRECTORY + '"', "output directory"),
        ('"fig05_daily_training_hourly_inference.png"', '"fig04_workflow_system_b.png"', "png name"),
        ('"fig05_daily_training_hourly_inference.pdf"', '"fig04_workflow_system_b.pdf"', "pdf name"),
        ('"fig05_daily_training_hourly_inference.svg"', '"fig04_workflow_system_b.svg"', "svg name"),
        ('"Daily model retraining and controlled deployment",\n    "once per day",',
         '"One-time model training and deployment",\n    "trained once",', "panel (a) title and cadence"),
        ('"Ascension Parish Data\\nUSGS + NOAA\\n2023 to present"',
         '"Ascension Parish Data\\nUSGS + NOAA\\nJan 2023 to Dec 2025"', "archive period"),
        ('"ST-GNN training", "shared encoder + graph\\nP50 to P90 loss"',
         '"ST-GNN training", "shared encoder + graph\\nP50 to P90 loss\\nepochs: 2025 hold-out"',
         "training card"),
        ('"Deploy assets", "trained weights + scalers\\nto forecast server"',
         '"Deploy assets", "frozen weights + scalers\\nto forecast server"', "deploy card"),
        ('"68 nodes; 497 edges"', '"68 nodes; 471 edges"', "graph size"),
        ('"active ST-GNN weights deployed by the daily cycle"',
         '"frozen ST-GNN weights, trained once"', "inference weights note"),
    ], [os.path.join(OUTPUT_DIRECTORY, "fig04_workflow_system_b.png"),
        os.path.join(OUTPUT_DIRECTORY, "fig04_workflow_system_b.pdf")]),
]

plt.show = lambda *args, **kwargs: None
for recipe, substitutions, outputs in JOBS:
    source_path = RECIPES + recipe
    source = open(source_path, encoding="utf-8").read()
    for old, new, name in substitutions:
        count = source.count(old)
        if count != 1:
            raise Exception(f"{recipe}: expected exactly one '{name}' target, found {count}")
        source = source.replace(old, new, 1)
    print("[VERIFY]", recipe, "applied", len(substitutions), "counted substitutions:",
          ", ".join(s[2] for s in substitutions), flush=True)
    namespace = {"__file__": source_path, "__name__": "__main__", "draw_model_box_no_cpu": draw_model_box_no_cpu}
    exec(compile(source, source_path, "exec"), namespace)
    plt.close("all")
    for path in outputs:
        if not os.path.isfile(path):
            raise Exception("missing output " + path)
        print("[VERIFY] created", path, flush=True)
