"""Figures 3 and 6 for revision 5 (System B): the candidate graphs and the selected graph, drawn from the graph
files the System B models were actually trained and evaluated with.

  Figure 3  the six candidate connectivity graphs. Recipe:
            src/figures/recipe_connectivity.py
            Substituted: its PANELS list (the operational graph files, including the 497-edge graph built in June
            2026, become the System B final graph files of
            src/graphs/alternative_graph_manifest.json,
            and the no-graph Bi-LSTM panel becomes the identity graph that the six-graph comparison actually
            trained), the panel (2) label and the output paths.
  Figure 6  the selected observation lead-lag graph. Recipe:
            src/figures/recipe_graph_map.py
            Substituted: its graph loader (the operational 497-edge graph becomes the System B 471-edge graph
            built from observations through 31 December 2025) and the output path. The title counts nodes and
            edges from the loaded graph, as before.
Basemap, colors, markers, arrows and layout are unchanged.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = "project/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913/frozen_assets/comparisons"
GRAPH_471 = ("project/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913/"
             "frozen_assets/graph_obs_pre2026_471edges.npz")
OUTPUT_DIRECTORY = ("project/manuscript/"
                    "20260914_System_B_Revision_5/figures")
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def load_system_b_graph():
    graph = np.load(GRAPH_471, allow_pickle=True)
    return [str(x) for x in graph["nodes"]], graph["A_norm"].astype("float32")


FIG03 = ("project/manuscript/20260826_Final_V2/"
         "figure_revision_scripts/recipe_connectivity.py")
FIG06 = ("project/manuscript/revision_2026_08_31/"
         "scripts/recipe_graph_map.py")
JOBS = [
    (FIG03, [
        ('OUT = ("project/"\n       "manuscript/20260826_Final_V2/"\n'
         '       "output/figures/fig03_inparish")',
         'OUT = "' + os.path.join(OUTPUT_DIRECTORY, "fig03_connectivity_system_b_inparish") + '"', "in-parish output"),
        ('OUT_ALL = ("project/"\n           "manuscript/20260826_Final_V2/"\n'
         '           "output/figures/fig03")',
         'OUT_ALL = "' + os.path.join(OUTPUT_DIRECTORY, "fig03_connectivity_system_b") + '"', "all-gauge output"),
        ('    (None, "(1) No-graph Bi-LSTM\\n(shared network, no river graph)"),\n'
         '    (f"{GDIR}/gnn_graph.npz", "(2) Observation lead-lag\\n(deployed)"),\n'
         '    (f"{GDIR}/gnn_graph_basinmask_v2.npz", "(3) Observation, within-basin"),\n'
         '    (f"{GDIR}/gnn_graph_tiff.npz", "(4) DEM downslope"),\n'
         '    (f"{GDIR}/gnn_graph_tiff_basinmask.npz", "(5) DEM, within-basin"),\n'
         '    (f"{GDIR}/gnn_graph_physics.npz", "(6) HEC-RAS hydraulic"),\n',
         '    ("' + C + '/graph_identity_final.npz", "(1) Identity, no cross-gauge\\nmessages (self only)"),\n'
         '    ("' + C + '/graph_obs_pre2026_final.npz", "(2) Observation lead-lag\\n(selected)"),\n'
         '    ("' + C + '/graph_obs_pre2026_basin_final.npz", "(3) Observation, within-basin"),\n'
         '    ("' + C + '/graph_dem_final.npz", "(4) DEM downslope"),\n'
         '    ("' + C + '/graph_dem_basin_final.npz", "(5) DEM, within-basin"),\n'
         '    ("' + C + '/graph_hecras_final.npz", "(6) HEC-RAS hydraulic"),\n',
         "graph panels"),
    ], [os.path.join(OUTPUT_DIRECTORY, "fig03_connectivity_system_b.png")]),
    (FIG06, [
        ('OUT = (\n    "project/"\n'
         '    "manuscript/revision_2026_08_31/"\n    "figures/fig06_graphmap_larger_fonts"\n)',
         'OUT = "' + os.path.join(OUTPUT_DIRECTORY, "fig06_graphmap_471_system_b") + '"', "output path"),
        ("nodes, A = load_graph()\n", "nodes, A = load_system_b_graph()\n", "graph loader"),
    ], [os.path.join(OUTPUT_DIRECTORY, "fig06_graphmap_471_system_b.png")]),
]

plt.show = lambda *args, **kwargs: None
for source_path, substitutions, outputs in JOBS:
    source = open(source_path, encoding="utf-8").read()
    for old, new, name in substitutions:
        count = source.count(old)
        if count != 1:
            raise Exception(f"{os.path.basename(source_path)}: expected exactly one '{name}' target, found {count}")
        source = source.replace(old, new, 1)
    print("[VERIFY]", os.path.basename(source_path), "applied", len(substitutions), "counted substitutions:",
          ", ".join(s[2] for s in substitutions), flush=True)
    namespace = {"__file__": source_path, "__name__": "__main__", "load_system_b_graph": load_system_b_graph}
    exec(compile(source, source_path, "exec"), namespace)
    plt.close("all")
    for path in outputs:
        if not os.path.isfile(path):
            raise Exception("missing output " + path)
        print("[VERIFY] created", path, flush=True)
