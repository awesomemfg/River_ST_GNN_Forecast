"""Figure 2 for revision 6: the revision 5 workflow diagram without the Bi-LSTM card.

Farid, 15 September 2026: the Bi-LSTM leaves the paper. Revision 5's Figure 2 still carried the dashed card
"Previous Bi-LSTM model, parallel hourly baseline (not published)" below panel (b).

Recipe: src/figures/recipe_workflow_base.py
Substitutions: the eight of revision 5
(src/figures/fig04_workflow.py)
with the output path moved to revision 6, plus one more: the Bi-LSTM card and its two dashed feed arrows are
removed. Nothing else in the drawing changes.
Output: project/manuscript/20260915_Revision_6/figures/fig02_workflow.png (and .pdf)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

RECIPE = "src/figures/recipe_workflow_base.py"
STEM = "project/manuscript/20260915_Revision_6/figures/fig02_workflow"
os.makedirs(os.path.dirname(STEM), exist_ok=True)


def draw_model_box_no_cpu(axis, center_x, center_y, width, height, color_value, face_value):
    """The original chip body without processor pins (as in the manuscript's Figure 2)."""
    model_box = FancyBboxPatch((center_x - width / 2.0, center_y - height / 2.0), width, height,
                               boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=2.0,
                               edgecolor=color_value, facecolor=face_value, zorder=5)
    axis.add_patch(model_box)


SUBSTITUTIONS = [
    ('OUT = "project/manuscript/figure_library_outputs/fig01_workflow"',
     'OUT = "' + STEM + '"', "output path"),
    ('draw_chip(ax, nx[2] + 0.72, ny, 1.05, 1.05, BLUE, "#C8DEEF")',
     'draw_model_box_no_cpu(ax, nx[2] + 0.72, ny, 1.05, 1.05, BLUE, "#C8DEEF")', "GRU icon"),
    ('draw_chip(ax, hx[2], hy, 1.2, 1.1, GREEN, "#C9E8D9")',
     'draw_model_box_no_cpu(ax, hx[2], hy, 1.2, 1.1, GREEN, "#C9E8D9")', "ST-GNN icon"),
    ('"(a)  Daily model retrain  —  high-performance computer"',
     '"(a)  One-time model training  —  high-performance computer"', "panel (a) title"),
    ('"stage and weather records,\\n2023-present"', '"stage and weather records,\\n1 Jan 2023 to 31 Dec 2025"', "archive period"),
    ('"68 gauges, 497-edge graph;\\nquantile loss"', '"68 gauges, 471-edge graph;\\nquantile loss"', "graph size"),
    ('"Updated daily\\nmodel weights", "uploaded to the\\nforecast server"',
     '"Frozen\\nmodel weights", "copied once to the\\nforecast server"', "weights card"),
    ('"latest model weights"', '"frozen model weights"', "weights arrow label"),
    ('# Bi-LSTM baseline (parallel, unpublished): a clean parallel-track card below the row, fed by two short\n'
     '# dashed arrows dropped in the gaps between columns (so nothing crosses the detail text).\n'
     'ax.add_patch(FancyBboxPatch((4.55, 0.34), 9.0, 0.70, boxstyle="round,pad=0.03,rounding_size=0.08",\n'
     '                            lw=1.2, linestyle="--", edgecolor=PURPLE, facecolor=PURPLE_L, zorder=4))\n'
     'ax.text(9.05, 0.69, "Previous Bi-LSTM model  —  parallel hourly baseline (not published)",\n'
     '        ha="center", va="center", fontsize=9.0, fontweight="bold", color=PURPLE, zorder=7)\n'
     'arrow(ax, (6.05, 1.20), (6.05, 1.06), PURPLE, dashed=True, lw=1.1)\n'
     'arrow(ax, (11.85, 1.20), (11.85, 1.06), PURPLE, dashed=True, lw=1.1)\n',
     "", "Bi-LSTM card and its two arrows removed"),
]

source = open(RECIPE, encoding="utf-8").read()
for old, new, name in SUBSTITUTIONS:
    count = source.count(old)
    if count != 1:
        raise Exception(f"expected exactly one '{name}' target, found {count}")
    source = source.replace(old, new, 1)
print("[VERIFY] recipe_workflow_base.py applied", len(SUBSTITUTIONS), "counted substitutions:",
      ", ".join(s[2] for s in SUBSTITUTIONS), flush=True)
plt.show = lambda *args, **kwargs: None
exec(compile(source, RECIPE, "exec"), {"__file__": RECIPE, "__name__": "__main__", "draw_model_box_no_cpu": draw_model_box_no_cpu})
plt.close("all")
for suffix in (".png", ".pdf"):
    if not os.path.isfile(STEM + suffix):
        raise Exception("missing output " + STEM + suffix)
    print("[VERIFY] created", STEM + suffix, flush=True)
print("REV6_FIG02_DONE", flush=True)
