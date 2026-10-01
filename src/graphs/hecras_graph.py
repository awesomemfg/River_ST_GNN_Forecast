"""Build the PHYSICS GNN graph from the EBR HEC-RAS simulation (real hydraulics, not geometry, not obs).
1. read the 69 gauge-cell Water-Surface time series from the EBR PostProcessing.hdf (lazy slice)
2. cross-correlate every mesh-connected pair (<= MAXMI floodplain mi) of simulated WSE -> which gauge's
   wave LEADS (=upstream) + lag (=travel time) + peak corr (=edge strength)
3. keep corr>=CORR_THR & |lag|>=MINLAG; A[d,u]=corr; row-normalize -> gnn_graph_physics.npz (+ pairs csv)
Same node order as the deployed graph so features align. Mirrors build_gnn_graph.py's edge rule but the
direction/lag/strength come from SIMULATED hydraulics."""
import numpy as np, pandas as pd, h5py
from scipy.signal import correlate, correlation_lags
GNN = "project/hpc/Training/GNN"
PP = "project/HECRAS_Files/New_Files/PostProcessing.hdf"
WSEPATH = "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D_REG_01/Water Surface"
RG = "project/physical_graphs"
OUT = "gnn_graph_physics.npz"
CORR_THR = 0.40
MINLAG = 2
MAXLAG = 720        # steps searched each side
MAXMI = 20.0        # only consider mesh-connected pairs within this floodplain distance

ref = np.load(f"{GNN}/gnn_graph.npz", allow_pickle=True)
nodes = [str(x) for x in ref["nodes"]]
N = len(nodes)
ix = {n: i for i, n in enumerate(nodes)}

rg = pd.read_csv(f"{RG}/gauge_nodes_upstream.csv").dropna(subset=["cell"])
rg["cell"] = rg["cell"].astype(int)
cellof = {r["gauge"]: int(r["cell"]) for _, r in rg.iterrows()}
snapof = {r["gauge"]: float(r["snap_ft"]) for _, r in rg.iterrows()}

# network distance (mesh) to limit candidate pairs
D = pd.read_csv(f"{RG}/gauge_network_distance_miles.csv", index_col=0)

# read 69 gauge-cell WSE series (lazy)
have = [n for n in nodes if n in cellof and snapof.get(n, 9e9) < 5280*5]   # within 5 mi snap = on mesh
cells = np.array([cellof[n] for n in have])
uniq = np.unique(cells)                      # sorted unique (h5py needs increasing, no dups)
with h5py.File(PP, "r") as f:
    d = f[WSEPATH]
    T = d.shape[0]
    Wuni = d[:, uniq]                        # (T, n_unique)
col = {c: j for j, c in enumerate(uniq)}
WS = np.stack([Wuni[:, col[cellof[n]]] for n in have], axis=1).astype("float32")  # (T, len(have))
print(f"[physics] read WSE ({T} steps) for {len(have)} on-mesh gauges; {len(nodes)-len(have)} off-mesh skipped")


def zc(x):
    x = x.astype("float64")
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x*0


pairs = []
for a in range(len(have)):
    for b in range(len(have)):
        if a == b:
            continue
        ga, gb = have[a], have[b]
        try:
            nd = float(D.loc[ga, gb])
        except Exception:
            nd = np.nan
        if not np.isfinite(nd) or nd > MAXMI:
            continue
        xa, xb = WS[:, a], WS[:, b]
        m = np.isfinite(xa) & np.isfinite(xb)
        if m.sum() < 100 or np.nanstd(xa[m]) < 1e-6 or np.nanstd(xb[m]) < 1e-6:
            continue
        za, zb = zc(xa[m]), zc(xb[m])
        cc = correlate(za, zb, mode="full")
        lags = correlation_lags(len(za), len(zb), mode="full")
        sel = np.abs(lags) <= MAXLAG
        cc, lags = cc[sel]/len(za), lags[sel]
        k = int(np.argmax(cc))
        lag, corr = int(lags[k]), float(cc[k])
        # lag>0 means za leads zb (a leads b) -> a upstream of b
        if corr >= CORR_THR and abs(lag) >= MINLAG:
            if lag > 0:
                pairs.append((gb, ga, abs(lag), corr))   # target=b(downstream), feeder=a(upstream)
            else:
                pairs.append((ga, gb, abs(lag), corr))
P = pd.DataFrame(pairs, columns=["target", "feeder", "lag_steps", "corr"]).drop_duplicates(["target", "feeder"])
P.to_csv(f"{RG}/physics_pairs.csv", index=False)

A = np.zeros((N, N), "float32")
LAG = np.zeros((N, N), "int32")
for r in P.itertuples():
    if r.target in ix and r.feeder in ix:
        A[ix[r.target], ix[r.feeder]] = max(A[ix[r.target], ix[r.feeder]], r.corr)
        LAG[ix[r.target], ix[r.feeder]] = r.lag_steps
rs = A.sum(1, keepdims=True)
A_norm = np.where(rs > 0, A/np.where(rs == 0, 1, rs), 0.0).astype("float32")
deg = (A > 0).sum(1)
np.savez(f"{GNN}/{OUT}", nodes=np.array(nodes), A=A, A_norm=A_norm, LAG=LAG)
print(f"[physics] edges={int((A>0).sum())} connected={int((deg>0).sum())}/{N} isolated={int((deg==0).sum())}")
print(f"  saved {GNN}/{OUT} + {RG}/physics_pairs.csv")
print(f"  vs deployed xcorr-obs graph (497 edges) / hecras-geometry graph (366 edges)")
