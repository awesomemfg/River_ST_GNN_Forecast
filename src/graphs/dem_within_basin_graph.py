"""DEM-routing graph STRICTLY CONFINED BY BASIN (the untried mayor's-rule variant on the DEM graph).
Source = gnn_graph_tiff.npz (least-cost downhill routing on the bathy DEM). Keep an edge if:
  - the TARGET is a '**00**' external station            -> keep ALL its feeders (never isolate externals)
  - else feeder & target share a basin prefix AND the feeder is not a 00 external (within-basin only)
  - FALLBACK: an in-parish target with NO within-basin feeder keeps its single strongest cross-basin DEM
    feeder (only these few gauges get one cross-basin edge), so nothing is isolated -> no runaway.
Row-renormalize. -> gnn_graph_tiff_basinmask.npz   (+ a sanity graph-on-map PNG)
Run with the operational env python (needs geopandas for the plot).
"""
import os
import sys
import json
import numpy as np

GNN = "project/hpc/Training/GNN"
SRC = f"{GNN}/gnn_graph_tiff.npz"
OUT = f"{GNN}/gnn_graph_tiff_basinmask.npz"
FIG = "project/Experiments/MODEL_COMPARE_20260615/FOURWAY_AB_20260615/figs"
ANALYSIS = "project/manuscript_early/analysis"

g = np.load(SRC, allow_pickle=True)
nodes = [str(x) for x in g["nodes"]]
A = g["A"].copy()
N = len(nodes)


def is00(n):
    return n[4:6] == "00"


def basin(n):
    return n[:2]


keep = np.zeros((N, N), bool)
fallback_gauges = []
for d in range(N):
    within = []
    cross = []
    for u in range(N):
        if A[d, u] <= 0:
            continue
        if is00(nodes[d]):
            keep[d, u] = True                 # external target: keep all feeders (stay stable)
        elif is00(nodes[u]):
            continue                          # in-parish target NEVER fed by an external 00
        elif basin(nodes[d]) == basin(nodes[u]):
            keep[d, u] = True                 # within-basin in-parish feeder
            within.append(u)
        else:
            cross.append((float(A[d, u]), u))  # cross-basin candidate (fallback only)
    if (not is00(nodes[d])) and len(within) == 0 and cross:
        u = max(cross, key=lambda t: t[0])[1]
        keep[d, u] = True                     # anti-isolation fallback: 1 strongest cross-basin feeder
        fallback_gauges.append(nodes[d])

Am = np.where(keep, A, 0.0).astype("float32")
# last-resort: an in-parish gauge with NO DEM feeder at all (none within, none cross) gets a SELF-LOOP
# (A_norm[d,d]=1 -> message = its own state) so the row is never all-zero (pure-local, stable, no runaway).
inpar = [i for i, n in enumerate(nodes) if not is00(n)]
selfloop = []
for d in inpar:
    if (Am[d] > 0).sum() == 0:
        Am[d, d] = 1.0
        keep[d, d] = True
        selfloop.append(nodes[d])
rs = Am.sum(1, keepdims=True)
An = np.where(rs > 0, Am / np.where(rs == 0, 1, rs), 0.0).astype("float32")
LAG = g["LAG"] * keep.astype("int32") if "LAG" in g.files else np.zeros((N, N), "int32")
np.savez(OUT, nodes=np.array(nodes), A=Am, A_norm=An, LAG=LAG)

deg = (Am > 0).sum(1)
iso_real = [nodes[i] for i in inpar if deg[i] == 0]
print(f"self-loop fallback (no DEM feeder at all): {len(selfloop)} {selfloop}")
print(f"DEM basin-confined graph: {int((Am>0).sum())} edges  (DEM tiff had {int((A>0).sum())})")
print(f"in-parish gauges with a within-basin feeder kept strict; {len(fallback_gauges)} needed a 1-edge cross-basin fallback: {fallback_gauges}")
print(f"isolated in-parish gauges remaining: {len(iso_real)} {iso_real}")
print(f"-> wrote {OUT}")

# ---------------- sanity plot ----------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D
    import geopandas as gpd
    SM = json.load(open(f"{ANALYSIS}/station_meta.json"))["STATION_METADATA"]
    la = gpd.read_file(f"{ANALYSIS}/la_parishes.geojson")
    asc = gpd.read_file(f"{ANALYSIS}/ascension_parish.geojson")
    BASIN_COL = {"BC": "#1f77b4", "BM": "#2ca02c", "MB": "#d62728", "HB": "#9467bd"}
    BASIN_NM = {"BC": "Bayou Conway", "BM": "Bayou Manchac", "MB": "Marvin Broud", "HB": "Henderson Bayou"}
    keepn = [n for n in nodes if n in SM and not is00(n)]
    lon = {n: SM[n][1] for n in keepn}
    lat = {n: SM[n][0] for n in keepn}
    lo0, lo1 = min(lon.values()) - 0.06, max(lon.values()) + 0.06
    la0, la1 = min(lat.values()) - 0.05, max(lat.values()) + 0.05
    idx = {n: i for i, n in enumerate(nodes)}
    fig, ax = plt.subplots(figsize=(13, 11))
    plt.rcParams.update({"font.size": 15})
    la.plot(ax=ax, facecolor="#eef0ec", edgecolor="#b9b9b9", lw=0.7)
    asc.plot(ax=ax, facecolor="#fffdf8", edgecolor="black", lw=2.2, alpha=0.5)
    ne = 0
    for nd in keepn:
        for nu in keepn:
            if An[idx[nd], idx[nu]] <= 0.02:
                continue
            x0, y0, x1, y1 = lon[nu], lat[nu], lon[nd], lat[nd]
            cross = basin(nu) != basin(nd)
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.2,
                                         color="#d62728" if cross else "#3a6ea5",
                                         alpha=0.7 if cross else 0.4, zorder=3 if cross else 2, shrinkA=6, shrinkB=8))
            ne += 1
    for n in keepn:
        ax.scatter(lon[n], lat[n], s=130, c=BASIN_COL.get(n[:2], "#555"), edgecolors="black", lw=0.9, zorder=6)
    ax.set_xlim(lo0, lo1)
    ax.set_ylim(la0, la1)
    ax.set_aspect("auto")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"DEM routing STRICTLY confined by basin  —  {ne} edges\n(blue = within-basin, red = anti-isolation cross-basin fallback)", fontsize=16)
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=BASIN_COL[b], markeredgecolor="black", ms=14, label=BASIN_NM[b]) for b in ["BC", "BM", "MB", "HB"]]
    leg += [Line2D([0], [0], color="#3a6ea5", lw=2.4, label="within-basin DEM flow"),
            Line2D([0], [0], color="#d62728", lw=2.4, label="cross-basin fallback (isolated gauges)")]
    ax.legend(handles=leg, loc="lower right", framealpha=0.95, title="Drainage basin")
    fig.tight_layout()
    fig.savefig(f"{FIG}/DEM basin-confined graph.png", dpi=150)
    plt.close()
    print(f"-> wrote {FIG}/DEM basin-confined graph.png  ({ne} edges drawn)")
except Exception as e:
    print("plot skipped:", e)
