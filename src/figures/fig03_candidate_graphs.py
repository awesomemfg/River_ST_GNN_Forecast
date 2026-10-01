"""Figure 3 for revision 6: the six candidate graphs, with the connections drawn so they can be seen.

Farid, 16 September 2026: "the line is not visible enough. i think we can make alpha = 1 for the line, make
line black, and make the icons of stages smaller and add idk alpha for the icon is 0.7".

Recipe: src/figures/recipe_connectivity.py
Substitutions, in two groups:
  1. the six graph files and the output paths, as in revision 5
     (20260914_System_B_Revision_5/scripts/fig07_selected_graph.py);
  2. visibility, applied to both the in-parish panel set and the all-gauge set that the manuscript uses:
     connections become black at full opacity and slightly thicker, and the gauge markers become smaller
     and 70 % opaque.
Basemap, projection, colors of the basins, panel layout, titles and legend are untouched.

Output: project/manuscript/20260915_Revision_6/figures/fig03_connectivity_rev6.png
Run with: conda run -n operational python fig03_candidate_graphs.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER = "project/manuscript/"
COMPARISONS = ("project/Experiments/"
               "SYSTEM_B_471_CHRONOLOGICAL_20260913/frozen_assets/comparisons")
FIGURES = PAPER + "20260915_Revision_6/figures"
RECIPE = "src/figures/recipe_connectivity.py"
STEM_ALL = os.path.join(FIGURES, "fig03_connectivity_rev6")
STEM_INPARISH = os.path.join(FIGURES, "fig03_connectivity_rev6_inparish")
os.makedirs(FIGURES, exist_ok=True)

SUBSTITUTIONS = [
    # ---- 1. inputs and outputs (revision 5)
    ('OUT = ("project/"\n       "manuscript/20260826_Final_V2/"\n'
     '       "output/figures/fig03_inparish")', 'OUT = "' + STEM_INPARISH + '"', "in-parish output"),
    ('OUT_ALL = ("project/"\n           "manuscript/20260826_Final_V2/"\n'
     '           "output/figures/fig03")', 'OUT_ALL = "' + STEM_ALL + '"', "all-gauge output"),
    ('    (None, "(1) No-graph Bi-LSTM\\n(shared network, no river graph)"),\n'
     '    (f"{GDIR}/gnn_graph.npz", "(2) Observation lead-lag\\n(deployed)"),\n'
     '    (f"{GDIR}/gnn_graph_basinmask_v2.npz", "(3) Observation, within-basin"),\n'
     '    (f"{GDIR}/gnn_graph_tiff.npz", "(4) DEM downslope"),\n'
     '    (f"{GDIR}/gnn_graph_tiff_basinmask.npz", "(5) DEM, within-basin"),\n'
     '    (f"{GDIR}/gnn_graph_physics.npz", "(6) HEC-RAS hydraulic"),\n',
     '    ("' + COMPARISONS + '/graph_identity_final.npz", "(1) Identity, no cross-gauge\\nmessages (self only)"),\n'
     '    ("' + COMPARISONS + '/graph_obs_pre2026_final.npz", "(2) Observation lead-lag\\n(selected)"),\n'
     '    ("' + COMPARISONS + '/graph_obs_pre2026_basin_final.npz", "(3) Observation, within-basin"),\n'
     '    ("' + COMPARISONS + '/graph_dem_final.npz", "(4) DEM downslope"),\n'
     '    ("' + COMPARISONS + '/graph_dem_basin_final.npz", "(5) DEM, within-basin"),\n'
     '    ("' + COMPARISONS + '/graph_hecras_final.npz", "(6) HEC-RAS hydraulic"),\n', "graph panels"),
    # ---- 2. visibility: connections
    ('ax.add_collection(LineCollection(segs, colors=[(0.3, 0.3, 0.3, 0.30)], linewidths=0.4, zorder=5))',
     'ax.add_collection(LineCollection(segs, colors=[(0.0, 0.0, 0.0, 1.0)], linewidths=0.45, zorder=5))',
     "in-parish identity connections"),
    ('ax.add_collection(LineCollection(segs, colors=[(0.20, 0.33, 0.48, 0.30)], linewidths=0.4, zorder=5))',
     'ax.add_collection(LineCollection(segs, colors=[(0.0, 0.0, 0.0, 1.0)], linewidths=0.45, zorder=5))',
     "in-parish graph connections"),
    ('ax.add_collection(LineCollection(segs, colors=[(0.3, 0.3, 0.3, 0.28)], linewidths=0.34, zorder=5))',
     'ax.add_collection(LineCollection(segs, colors=[(0.0, 0.0, 0.0, 1.0)], linewidths=0.40, zorder=5))',
     "all-gauge identity connections"),
    ('ax.add_collection(LineCollection(segs, colors=[(0.20, 0.33, 0.48, 0.28)], linewidths=0.34, zorder=5))',
     'ax.add_collection(LineCollection(segs, colors=[(0.0, 0.0, 0.0, 1.0)], linewidths=0.40, zorder=5))',
     "all-gauge graph connections"),
    # ---- 2. visibility: gauge markers
    ('ax.scatter(x, y, s=66, c=BASIN_COL.get(basin(g), "#555"), edgecolors="black", linewidths=0.7, zorder=8)',
     'ax.scatter(x, y, s=34, c=BASIN_COL.get(basin(g), "#555"), edgecolors="black", linewidths=0.5, alpha=0.7,\n'
     '               zorder=8)', "in-parish gauge markers"),
    ('            ax.scatter(x, y, marker="x", s=80, c="0.25", linewidths=1.8, zorder=9)',
     '            ax.scatter(x, y, marker="x", s=44, c="0.25", linewidths=1.2, alpha=0.7, zorder=9)',
     "all-gauge supporting markers"),
    ('            ax.scatter(x, y, marker=marker, s=74, c=BASIN_COL.get(basin(gauge), "#555"),\n'
     '                       edgecolors="black", linewidths=0.7, zorder=10)',
     '            ax.scatter(x, y, marker=marker, s=38, c=BASIN_COL.get(basin(gauge), "#555"),\n'
     '                       edgecolors="black", linewidths=0.5, alpha=0.7, zorder=10)', "all-gauge gauge markers"),
]

source = open(RECIPE, encoding="utf-8").read()
for old, new, name in SUBSTITUTIONS:
    count = source.count(old)
    if count != 1:
        raise Exception(f"expected exactly one '{name}' target, found {count}")
    source = source.replace(old, new, 1)
print("[VERIFY] recipe_connectivity.py applied", len(SUBSTITUTIONS), "counted substitutions:",
      ", ".join(s[2] for s in SUBSTITUTIONS), flush=True)
plt.show = lambda *args, **kwargs: None
exec(compile(source, RECIPE, "exec"), {"__file__": RECIPE, "__name__": "__main__"})
plt.close("all")
for stem in (STEM_ALL, STEM_INPARISH):
    for suffix in (".png", ".pdf"):
        if not os.path.isfile(stem + suffix):
            raise Exception("missing output " + stem + suffix)
        print("[VERIFY] created", stem + suffix, flush=True)
print("REV6_FIG03_DONE", flush=True)
