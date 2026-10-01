"""REFORECAST 2026H1 trainer — EXPERIMENT, NOT PRODUCTION (never uploaded to <SERVER>).

Copy of the deployed DQ-P80 trainer (gnn_train_featDQ.py) with exactly three changes, for the
paper's leakage-safe retrospective rolling-origin reforecast over 2026-01-01..2026-06-30:
  1. TRAIN_END = "2025-12-31 23:45"  — the matrix is sliced BEFORE any statistic is computed, so
     per-node mu/sd, channel z-stats, and all training windows see NOTHING after 2025.
  2. OUTD = Model_Weights_GNN_featDQ_reforecast_trainend20251231 (+ "_nograph" in identity mode).
  3. GRAPH_MODE: "obs" = the deployed 68-node observation lead-lag graph (default);
     "identity" = A := I (no cross-gauge message passing) — the clean no-graph ST-GNN control.
Everything else (features, architecture, quantiles, loss, epochs, freshness guard) is identical.
Evaluated by project/Experiments/REFORECAST_2026H1/<SERVER>/
eval_reforecast_2026H1_perfectQPF.py (runs on <SERVER> via the project_hpc mount).

history = [stage_z, rain6_z, rain24_z, API24_z, API48_z, API72_z, tend_z, RE_z, raccel_z]  (9)
forcing = [rain6_z, rain24_z, API24_z, API48_z, API72_z, RE_z, raccel_z, sin, cos]          (9)
Output: residual_from_t0 at each quantile. Meta feat_version=DQ + quantiles list.
NOTE: all config is HARDCODED below (no environment variables) — container env is unreliable.
"""
import os, sys, json, numpy as np, pandas as pd, tensorflow as tf
try:
    import numpy.core as _npc, numpy.core.numeric as _npn, numpy.core.multiarray as _npm
    for _k, _v in [("numpy._core", _npc), ("numpy._core.numeric", _npn), ("numpy._core.multiarray", _npm)]:
        sys.modules.setdefault(_k, _v)
except Exception:
    pass

# =============================================================================
# HARDCODED OPERATIONAL CONFIG (2026-06-25) — NO ENVIRONMENT VARIABLES.
# This trainer runs on a PUBLIC, SHARED HPC inside an apptainer container, where env vars are NOT a reliable
# transport (cleanenv, container passthrough, other users/contexts). Every operational value below is hardcoded
# so the nightly retrain ALWAYS loads the freshly-downloaded daily all-stations pickle and the correct config,
# with zero chance of an env-miss silently falling back to a stale pickle. Do NOT reintroduce os.environ here.
# =============================================================================
SMOKE = False
DIR = "project/hpc/Training/GNN"
# FRESH daily all-stations pickle, rewritten every night by download_and_prepare_matrix.py (NOT the stale FULLDATA_QC one-off):
QC = "project/hpc/Training/Prepared_Global_Matrix/global_features_all_stations_feature_engineered.pkl"
# GRAPH_MODE: "obs" = deployed observation lead-lag graph; "identity" = no-graph control (A = I).
GRAPH_MODE = "obs"
OUTD = "project/hpc/Training/GNN/Model_Weights_GNN_featDQ_reforecast_trainend20251231"
if GRAPH_MODE == "identity":
    OUTD = OUTD + "_nograph"
QUANTILES = [0.5, 0.6, 0.7, 0.8, 0.9]
TEND_LAG = 24
HL = {"api24": 96.0, "api48": 192.0, "api72": 288.0}
IN_W, LB_W, H = 288, 96, 64
IN_CH, F_CH = 9, 9
NQ = len(QUANTILES)
EPOCHS = 40
STRIDE = 4
# Hard freshness limit: refuse to train if the loaded pickle is older than this many hours (hardcoded, always on).
FRESH_MAX_AGE_H = 48.0
os.makedirs(OUTD, exist_ok=True)

g = np.load(os.path.join(DIR, "gnn_graph.npz"), allow_pickle=True)
nodes = [str(x) for x in g["nodes"]]
A = g["A_norm"].astype("float32")
N = len(nodes)
if GRAPH_MODE == "identity":
    A = np.eye(N, dtype="float32")
    print(f"[graph] IDENTITY mode — no cross-gauge message passing (no-graph ST-GNN control)")
elif GRAPH_MODE != "obs":
    raise SystemExit(f"[graph][FATAL] unknown GRAPH_MODE={GRAPH_MODE!r} (use 'obs' or 'identity')")
print(f"[graph] N={N} edges={int((A>0).sum())} mode={GRAPH_MODE} feat_version=DQ quantiles={QUANTILES}")

# FRESH-DATA AUDIT + GUARD (2026-06-25): log exactly which pickle is loaded, its mtime, and its row count, so
# every training run is auditable for data freshness. ALWAYS hard-fail (no env toggle) if the loaded pickle is
# older than FRESH_MAX_AGE_H hours, instead of silently training on stale data.
import time as _time
from datetime import datetime as _dt, timezone as _tz
_qc_mtime = os.path.getmtime(QC)
_qc_age_h = (_time.time() - _qc_mtime) / 3600.0
print(f"[freshness] loading pickle: {QC}")
print(f"[freshness] pickle mtime  : {_dt.fromtimestamp(_qc_mtime, _tz.utc).isoformat()} (age {_qc_age_h:.1f}h)")
if _qc_age_h > FRESH_MAX_AGE_H:
    raise SystemExit(f"[freshness][FATAL] pickle {QC} is {_qc_age_h:.1f}h old > {FRESH_MAX_AGE_H}h limit. "
                     f"Refusing to train on stale data. Re-run download_and_prepare_matrix.py (check PYTHONLOG_DOWNLOAD.log).")
df = pd.read_pickle(QC).sort_index()
print(f"[freshness] loaded rows={len(df)} cols={df.shape[1]} range={df.index.min()} -> {df.index.max()}")
# LEAKAGE-SAFE CUTOFF: slice BEFORE any statistic (mu/sd, channel z-stats, windows) is computed.
TRAIN_END = "2025-12-31 23:45"
if TRAIN_END:
    df = df[df.index <= pd.Timestamp(TRAIN_END).tz_localize("UTC")]
    print(f"[split] {TRAIN_END} -> {len(df)} rows")
    if len(df) < 10000:
        raise SystemExit(f"[split][FATAL] only {len(df)} rows <= {TRAIN_END} — wrong pickle?")
if SMOKE:
    df = df.iloc[:8000]
T = len(df)
idxt = df.index

S = np.full((T, N), np.nan, "float32")
M = np.zeros((T, N), bool)
MASK_NPZ = ""
if MASK_NPZ:
    mz = np.load(MASK_NPZ, allow_pickle=True)
    M = pd.DataFrame(mz["mask"], index=pd.to_datetime(mz["grid"], utc=True)).reindex(idxt).fillna(False).values.astype(bool)
    for i, n in enumerate(nodes):
        c = f"{n}_stage_ft"
        if c in df.columns:
            S[:, i] = pd.to_numeric(df[c], errors="coerce").values
    print(f"[mask] coverage {M.mean()*100:.1f}%")
else:
    for i, n in enumerate(nodes):
        c = f"{n}_stage_ft"
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            s = s.mask(s.rolling(96, center=True, min_periods=96).std() < 1e-6)
            S[:, i] = s.values
            M[:, i] = np.isfinite(s.values)

m2r = pd.read_csv(os.path.join(DIR, "stage_to_rain_gauge.csv")).set_index("stage")["nearest_rain"].to_dict()
def raincol(suf):
    cols = [c for c in df.columns if c.endswith(suf)]
    gm = pd.to_numeric(df[cols].mean(axis=1), errors="coerce").fillna(0).values if cols else np.zeros(T)
    G = np.zeros((T, N), "float32")
    for i, n in enumerate(nodes):
        rc = f"{m2r.get(n,'')}{suf}"
        G[:, i] = pd.to_numeric(df[rc], errors="coerce").fillna(0).values if rc in df.columns else gm
    return G
R6 = raincol("_rain_in_accum_6h")
R24 = raincol("_rain_in_accum_24h")
RSTEP = raincol("_rain_in")
def api(hl):
    k = float(0.5 ** (1.0 / hl))
    out = np.zeros((T, N), "float32")
    acc = np.zeros(N, "float32")
    for t in range(T):
        acc = k * acc + RSTEP[t]
        out[t] = acc
    return out
API24, API48, API72 = api(HL["api24"]), api(HL["api48"]), api(HL["api72"])
RE = R6 * API48
RACCEL = np.zeros((T, N), "float32")
RACCEL[4:] = RSTEP[4:] - RSTEP[:-4]

mu = np.zeros(N, "float32")
sd = np.ones(N, "float32")
for i in range(N):
    r = S[M[:, i], i]
    r = r[np.isfinite(r)]
    if len(r) < 10:
        r = S[np.isfinite(S[:, i]), i]
    if len(r):
        mu[i] = r.mean()
        sd[i] = r.std() + 1e-6
Z = np.nan_to_num((S - mu) / sd)
TEND = np.zeros((T, N), "float32")
TEND[TEND_LAG:] = Z[TEND_LAG:] - Z[:-TEND_LAG]
def zc(X):
    m = float(X.mean())
    s = float(X.std() + 1e-6)
    return (X - m) / s, m, s
R6z, r6m, r6s = zc(R6)
R24z, r24m, r24s = zc(R24)
A24z, a24m, a24s = zc(API24)
A48z, a48m, a48s = zc(API48)
A72z, a72m, a72s = zc(API72)
REz, rem, res_ = zc(RE)
RAz, ram, ras = zc(RACCEL)
TENDz, tendm, tends = zc(TEND)
ang = 2 * np.pi * (idxt.hour + idxt.minute / 60.0) / 24
TS = np.sin(ang).astype("float32")
TC = np.cos(ang).astype("float32")

json.dump({"nodes": nodes, "mu": mu.tolist(), "sd": sd.tolist(), "r6": [r6m, r6s], "r24": [r24m, r24s],
           "api24": [a24m, a24s], "api48": [a48m, a48s], "api72": [a72m, a72s], "re": [rem, res_],
           "raccel": [ram, ras], "tend": [tendm, tends], "halflives": HL, "tend_lag": TEND_LAG,
           "feat_version": "DQ", "quantiles": QUANTILES, "IN_CH": IN_CH, "F_CH": F_CH,
           "IN_W": IN_W, "LB_W": LB_W, "H": H,
           "train_end": TRAIN_END, "graph_mode": GRAPH_MODE,
           "experiment": "REFORECAST_2026H1 (leakage-safe cutoff trainer; NOT production)"},
          open(os.path.join(OUTD, "gnn_meta.json"), "w"))

ten = {k: tf.constant(v) for k, v in dict(Z=Z, R6=R6z, R24=R24z, A24=A24z, A48=A48z, A72=A72z,
                                          RE=REz, RA=RAz, TEND=TENDz, TS=TS, TC=TC).items()}
MT = tf.constant(M.astype("float32"))
starts = np.arange(0, T - IN_W - LB_W, STRIDE, dtype=np.int64)
np.random.default_rng(0).shuffle(starts)
def make(i):
    h = tf.stack([ten["Z"][i:i+IN_W], ten["R6"][i:i+IN_W], ten["R24"][i:i+IN_W], ten["A24"][i:i+IN_W],
                  ten["A48"][i:i+IN_W], ten["A72"][i:i+IN_W], ten["TEND"][i:i+IN_W], ten["RE"][i:i+IN_W], ten["RA"][i:i+IN_W]], -1)
    f0 = i + IN_W
    t0 = f0 - 1
    fts = tf.tile(ten["TS"][f0:f0+LB_W, None], [1, N])
    ftc = tf.tile(ten["TC"][f0:f0+LB_W, None], [1, N])
    forcing = tf.stack([ten["R6"][f0:f0+LB_W], ten["R24"][f0:f0+LB_W], ten["A24"][f0:f0+LB_W], ten["A48"][f0:f0+LB_W],
                        ten["A72"][f0:f0+LB_W], ten["RE"][f0:f0+LB_W], ten["RA"][f0:f0+LB_W], fts, ftc], -1)
    z0 = ten["Z"][t0]
    tgt = ten["Z"][f0:f0+LB_W] - z0[None, :]
    w = MT[f0:f0+LB_W] * MT[t0][None, :]
    return (h, forcing, z0), tgt, w
ds = tf.data.Dataset.from_tensor_slices(starts).map(make, num_parallel_calls=tf.data.AUTOTUNE)
nval = max(1, int(len(starts) * 0.1))
BATCH = 8 if SMOKE else 16
val = ds.take(nval).batch(BATCH).prefetch(2)
trn = ds.skip(nval).batch(BATCH).shuffle(256).prefetch(2)
print(f"[data] T={T} windows={len(starts)} batch={BATCH} NQ={NQ}")

TAUS = tf.constant(QUANTILES, dtype=tf.float32)
def multi_pinball(y, yh):
    # y [B,LB,N]; yh [B,LB,N,NQ]; return [B,LB,N] (mean over quantiles) so sample_weight [B,LB,N] applies
    e = y[..., None] - yh
    pl = tf.maximum(TAUS * e, (TAUS - 1.0) * e)
    return tf.reduce_mean(pl, axis=-1)

class STGNNQ(tf.keras.Model):
    def __init__(s):
        super().__init__()
        s.A = tf.constant(A)
        s.enc = tf.keras.layers.GRU(H)
        s.cell = tf.keras.layers.GRUCell(H)
        s.out = tf.keras.layers.Dense(NQ)
    def call(s, inp, training=False):
        h, forcing, z0 = inp
        B = tf.shape(h)[0]
        hs = tf.reshape(s.enc(tf.reshape(tf.transpose(h, [0, 2, 1, 3]), [B*N, IN_W, IN_CH])), [B, N, H])
        outs = tf.TensorArray(tf.float32, size=LB_W)
        for t in range(LB_W):
            msg = tf.einsum('du,buh->bdh', s.A, hs)
            inp_t = tf.concat([forcing[:, t], msg], -1)
            h2, _ = s.cell(tf.reshape(inp_t, [B*N, F_CH+H]), [tf.reshape(hs, [B*N, H])])
            hs = tf.reshape(h2, [B, N, H])
            outs = outs.write(t, tf.reshape(s.out(hs), [B, N, NQ]))
        return tf.transpose(outs.stack(), [1, 0, 2, 3])    # [B,LB,N,NQ]
model = STGNNQ()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0), loss=multi_pinball)
cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)]
model.fit(trn, validation_data=val, epochs=EPOCHS, callbacks=cb, verbose=2)
model.save_weights(os.path.join(OUTD, "gnn.weights.h5"))
print("[SAVE]", os.path.join(OUTD, "gnn.weights.h5"))
print("GNN_TRAIN_DONE")
