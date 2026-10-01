"""Build a GNN routing graph from the EBR BATHY TIFF (high-res DEM), via least-cost downhill flow
routing — vs the original BEAST graph which used the coarse mesh-TIN elevation in the .hdf.

Method:
  1. read the 5ft bathy TIFF downsampled to ~500ft (tractable), CRS EPSG:3452
  2. directed grid graph (8-conn): cost(i->j) = dist + BIGPEN*max(0, elev_j-elev_i)  -> downhill cheap,
     uphill ~blocked, so least-cost paths follow channels/drainage
  3. snap 69 gauges to the lowest cell in a small window (put them in-channel); take bathy elevation there
  4. Dijkstra from each gauge -> cost to every other gauge; u feeds d (u upstream) if reachable downhill
     (finite cost u->d) AND elev(u)>elev(d); keep the K nearest-by-cost upstream feeders per gauge
  5. A[d,u]=1/geodist_mi, row-normalize -> gnn_graph_tiff.npz (+ tiff_pairs.csv)
Same node order as deployed graph."""
import numpy as np, pandas as pd, rasterio
from rasterio.enums import Resampling
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
GNN = "project/hpc/Training/GNN"
RG = "project/physical_graphs"
TIF = "project/HECRAS_Files/New_Files/EBR_Regional_wBathymetry_v4_CRD.EBR_Regional_wBathymetry_v4.Resampled.tif"
FACTOR = 100          # 5ft * 100 = ~500ft cells
BIGPEN = 500.0        # cost per ft of uphill climb (effectively blocks uphill)
MAXMI = 20.0
KFEED = 6             # max upstream feeders kept per gauge
SNAPWIN = 6           # cells radius to snap a gauge to the local channel (lowest cell)

ref = np.load(f"{GNN}/gnn_graph.npz", allow_pickle=True)
nodes = [str(x) for x in ref["nodes"]]
N = len(nodes)
ix = {n: i for i, n in enumerate(nodes)}
coords = pd.read_csv(f"{RG}/gauge_nodes_upstream.csv").set_index("gauge")[["lat", "lon"]].to_dict("index")

with rasterio.open(TIF) as r:
    W, H = r.width//FACTOR, r.height//FACTOR
    dem = r.read(1, out_shape=(1, H, W), resampling=Resampling.average).astype("float64")
    tr = r.transform*r.transform.scale(r.width/W, r.height/H)
    nod = r.nodata
    px, py = abs(tr.a), abs(tr.e)
    x0, y0 = tr.c, tr.f
print(f"[tiff] DEM {W}x{H} ~{px:.0f}ft, valid%={100*np.mean(dem!=nod):.1f}")
valid = dem != nod
elev = np.where(valid, dem, np.inf)

# ---- directed grid graph ----
idx = np.arange(H*W).reshape(H, W)
src, dst, cst = [], [], []
for dr in (-1, 0, 1):
    for dc in (-1, 0, 1):
        if dr == 0 and dc == 0:
            continue
        dist = np.hypot(dr*py, dc*px)
        a = elev[max(0, -dr):H-max(0, dr), max(0, -dc):W-max(0, dc)]
        b = elev[max(0, dr):H-max(0, -dr), max(0, dc):W-max(0, -dc)]
        ia = idx[max(0, -dr):H-max(0, dr), max(0, -dc):W-max(0, dc)]
        ib = idx[max(0, dr):H-max(0, -dr), max(0, dc):W-max(0, -dc)]
        ok = np.isfinite(a) & np.isfinite(b)
        climb = np.maximum(0.0, b-a)
        c = dist + BIGPEN*climb
        src.append(ia[ok]); dst.append(ib[ok]); cst.append(c[ok])
src, dst, cst = np.concatenate(src), np.concatenate(dst), np.concatenate(cst)
G = csr_matrix((cst, (src, dst)), shape=(H*W, H*W))
print(f"[tiff] grid graph: {H*W} cells, {len(cst)} directed edges")

# ---- snap gauges to local channel (lowest cell in window) ----
trf = Transformer.from_crs(4326, 3452, always_xy=True)
gcell, gelev, gxy = {}, {}, {}
for n in nodes:
    if n not in coords:
        continue
    x, y = trf.transform(coords[n]["lon"], coords[n]["lat"])
    c = int((x-x0)/px); rr = int((y0-y)/py)
    if not (0 <= rr < H and 0 <= c < W):
        continue
    r0, r1, c0, c1 = max(0, rr-SNAPWIN), min(H, rr+SNAPWIN+1), max(0, c-SNAPWIN), min(W, c+SNAPWIN+1)
    win = elev[r0:r1, c0:c1]
    if not np.isfinite(win).any():
        continue
    fr, fc = np.unravel_index(np.nanargmin(np.where(np.isfinite(win), win, np.inf)), win.shape)
    rr2, cc2 = r0+fr, c0+fc
    gcell[n] = rr2*W+cc2; gelev[n] = float(elev[rr2, cc2]); gxy[n] = (x, y)
have = list(gcell)
print(f"[tiff] snapped {len(have)} gauges to channel cells")

# ---- cost from each gauge to all cells -> gauge-to-gauge ----
CM = np.full((N, N), np.inf)
for u in have:
    d = dijkstra(G, directed=True, indices=gcell[u])
    for b in have:
        CM[ix[u], ix[b]] = d[gcell[b]]

def geomi(a, b):
    return np.hypot(gxy[a][0]-gxy[b][0], gxy[a][1]-gxy[b][1])/5280.0

# ---- edges: keep K nearest-by-cost downhill feeders per downstream gauge ----
pairs = []
for d in have:
    cands = []
    for u in have:
        if u == d:
            continue
        if np.isfinite(CM[ix[u], ix[d]]) and gelev[u] > gelev[d]+0.1 and geomi(u, d) <= MAXMI:
            cands.append((CM[ix[u], ix[d]], u))
    cands.sort()
    for cost, u in cands[:KFEED]:
        pairs.append((d, u, round(geomi(u, d), 2), round(gelev[u]-gelev[d], 2)))
P = pd.DataFrame(pairs, columns=["target", "feeder", "geo_mi", "delev_ft"])
P.to_csv(f"{RG}/tiff_pairs.csv", index=False)

A = np.zeros((N, N), "float32")
for r_ in P.itertuples():
    A[ix[r_.target], ix[r_.feeder]] = 1.0/max(r_.geo_mi, 0.1)
rs = A.sum(1, keepdims=True)
A_norm = np.where(rs > 0, A/np.where(rs == 0, 1, rs), 0.0).astype("float32")
deg = (A > 0).sum(1)
np.savez(f"{GNN}/gnn_graph_tiff.npz", nodes=np.array(nodes), A=A, A_norm=A_norm, LAG=np.zeros((N, N), "int32"))
print(f"[tiff] edges={int((A>0).sum())} connected={int((deg>0).sum())}/{N} isolated={int((deg==0).sum())}")
print(f"  saved {GNN}/gnn_graph_tiff.npz + {RG}/tiff_pairs.csv")
print(f"  (BEAST geometry-hdf=366 edges, xcorr-obs=497, physics-sim=963)")
