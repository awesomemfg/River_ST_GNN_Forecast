"""Rebuild manuscript Figure 15 from its original six-network map recipe.

The experiment recipe supplies the map data, projection, colors, symbols, and edges.
This script selects the six frozen networks used in the paper. It keeps the
original two-row, three-column layout and adds space around its titles and
legend. Only a PNG is written.
"""

import os

import matplotlib.pyplot as plt


paper_directory = "project/manuscript"
experiment_directory = "project/Experiments"
revision_directory = os.path.join(paper_directory, "20260924_Revision_14")
figure_directory = os.path.join(revision_directory, "figures")
work_directory = os.path.join(revision_directory, "work")
recipe_path = os.path.join(
    experiment_directory,
    "NODE_SUBSAMPLE_20260813",
    "scripts",
    "plot_subsample_maps.py",
)
full_graph_path = os.path.join(
    experiment_directory,
    "SYSTEM_B_471_CHRONOLOGICAL_20260913",
    "frozen_assets",
    "graph_obs_pre2026_471edges.npz",
)
inparish_graph_directory = os.path.join(
    experiment_directory,
    "SYSTEM_B_BOUNDARY_INPARISH_20260914",
    "frozen_assets",
)
figure_stem = "fig15_gauge_network_maps"
figure_path = os.path.join(figure_directory, figure_stem + ".png")
summary_path = os.path.join(work_directory, "fig15_gauge_network_panels.json")

print("[FIGURE 15] Native recipe:", recipe_path, flush=True)
print("[FIGURE 15] Full graph:", full_graph_path, flush=True)
print("[FIGURE 15] Reduced graphs:", inparish_graph_directory, flush=True)
print("[FIGURE 15] Output:", figure_path, flush=True)

if not os.path.isfile(recipe_path):
    raise FileNotFoundError("Missing native map recipe: " + recipe_path)
if not os.path.isfile(full_graph_path):
    raise FileNotFoundError("Missing full network graph: " + full_graph_path)
for gauge_count in (51, 40, 30, 20, 10):
    if gauge_count == 51:
        graph_name = "graph_inparish51_final.npz"
    else:
        graph_name = f"graph_ipsub{gauge_count}_d1_final.npz"
    graph_path = os.path.join(inparish_graph_directory, graph_name)
    if not os.path.isfile(graph_path):
        raise FileNotFoundError("Missing reduced network graph: " + graph_path)

os.makedirs(figure_directory, exist_ok=True)
os.makedirs(work_directory, exist_ok=True)

with open(recipe_path, encoding="utf-8") as recipe_file:
    source = recipe_file.read()

substitutions = [
    (
        'CONTROL = os.path.join(MIKE, "graph_obs_pre2026.npz")',
        f'CONTROL = "{full_graph_path}"\nINPARISH_GRAPHS = "{inparish_graph_directory}"',
        "full network graph",
    ),
    (
        '    (os.path.join(MIKE, "graph_inparish51.npz"), "51 gauges", "out-of-parish feeds removed"),',
        '    (os.path.join(INPARISH_GRAPHS, "graph_inparish51_final.npz"), "51 gauges", "out-of-parish feeds removed"),',
        "51-gauge graph",
    ),
    (
        '    (os.path.join(MIKE, "graph_sub50.npz"), "50 gauges", "random draw"),\n',
        "",
        "unused 50-gauge graph",
    ),
]

for gauge_count in (40, 30, 20, 10):
    substitutions.append(
        (
            f'    (os.path.join(MIKE, "graph_sub{gauge_count}.npz"), "{gauge_count} gauges", "random draw"),',
            f'    (os.path.join(INPARISH_GRAPHS, "graph_ipsub{gauge_count}_d1_final.npz"), "{gauge_count} gauges", "in-parish, draw 1"),',
            f"{gauge_count}-gauge in-parish graph",
        )
    )

substitutions.extend(
    [
        ('    NROW, NCOL = 2, 4', '    NROW, NCOL = 2, 3', "original two-row, three-column panel grid"),
        ('    FIG_W = 21.0', '    FIG_W = 17.0', "original figure width"),
        ('    MARGIN = 0.10', '    MARGIN = 0.22', "outer margin"),
        ('    H_GAP = 0.10', '    H_GAP = 0.16', "column gap"),
        ('    V_GAP = 0.05', '    V_GAP = 0.28', "row gap"),
        ('    TITLE_H = 0.40', '    TITLE_H = 0.62', "panel title space"),
        (
            '    FIG_H = 2 * MARGIN + NROW * (TITLE_H + panel_h) + (NROW - 1) * V_GAP',
            '    LEGEND_H = 1.12\n'
            '    SUPTITLE_H = 0.38\n'
            '    FIG_H = 2 * MARGIN + NROW * (TITLE_H + panel_h) + (NROW - 1) * V_GAP + LEGEND_H + SUPTITLE_H',
            "space around the original title and legend",
        ),
        (
            '        top_in = MARGIN + row * (TITLE_H + panel_h + V_GAP) + TITLE_H',
            '        top_in = MARGIN + SUPTITLE_H + row * (TITLE_H + panel_h + V_GAP) + TITLE_H',
            "panels below the original figure title",
        ),
        (
            '    axes.ravel()[-1].axis("off")   # the eighth slot carries the legend',
            '    # All six axes carry maps; the original legend remains below them.',
            "six map panels",
        ),
        (
            '    axes.ravel()[-1].legend(handles=leg, loc="center", ncol=1, fontsize=10.0,\n'
            '                            frameon=True, borderpad=0.7, labelspacing=0.7)',
            '    fig.legend(handles=leg, loc="center", ncol=5, fontsize=14.0,\n'
            '               frameon=True, borderpad=0.7, labelspacing=0.7,\n'
            '               bbox_to_anchor=(0.5, (MARGIN + LEGEND_H / 2) / FIG_H))',
            "original five-column legend",
        ),
        (
            '                     fontsize=11.0, pad=5)',
            '                     fontsize=15.5, pad=6)',
            "original panel heading style",
        ),
        (
            '        fig.suptitle("Reduced gauge networks tested against the full 68-gauge network",\n'
            '                     fontsize=15.5, y=0.995)',
            '        fig.suptitle("Reduced gauge networks", fontsize=19.0, fontweight="bold", y=0.975)',
            "original figure title",
        ),
        (
            '    for ext in ("png", "pdf"):\n'
            '        fig.savefig(f"{stem}.{ext}", dpi=200)',
            '    for ext in ("png",):\n'
            '        fig.savefig(f"{stem}.{ext}", dpi=300)',
            "PNG-only high-resolution output",
        ),
        (
            'os.path.join(SUB, "outputs", "subsample_map_panels.json")',
            f'"{summary_path}"',
            "panel summary path",
        ),
    ]
)

for old_text, new_text, description in substitutions:
    occurrence_count = source.count(old_text)
    if occurrence_count != 1:
        raise ValueError(
            f"Expected one source location for {description}; found {occurrence_count}"
        )
    source = source.replace(old_text, new_text)
    print("[FIGURE 15] Applied:", description, flush=True)

os.environ["SUBSAMPLE_MAPS_TITLE"] = "on"
os.environ["SUBSAMPLE_MAPS_DIR"] = figure_directory
os.environ["SUBSAMPLE_MAPS_NAME"] = figure_stem

namespace = {"__file__": recipe_path, "__name__": "__main__"}
exec(compile(source, recipe_path, "exec"), namespace)

if not os.path.isfile(figure_path):
    raise FileNotFoundError("Figure was not created: " + figure_path)
if not os.path.isfile(summary_path):
    raise FileNotFoundError("Panel summary was not created: " + summary_path)

print("[FIGURE 15] Created:", figure_path, flush=True)
print("[FIGURE 15] Verified panel counts:", summary_path, flush=True)
plt.show()
