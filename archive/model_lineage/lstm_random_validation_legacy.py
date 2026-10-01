"""Train the ST-GNN and the gauge-by-gauge LSTM once, on the same frozen matrix, for the revised paper.

EXPERIMENT, NOT PRODUCTION. Nothing here is uploaded to the forecast server.

Protocol = the paper's frozen reforecast trainer
(archive/model_lineage/stgnn_train_reforecast_2026.py),
unchanged except for four things:
  1. It reads ONE frozen copy of the training matrix (frozen_inputs/, sha256 logged), so the ST-GNN and
     the LSTM see bit-identical data. The nightly matrix is rebuilt every night and the Parish backfills
     telemetry, so models trained from different days are not a matched pair.
  2. model kind is chosen on the command line: "stgnn" (the paper's model, deployed 68-node graph) or
     "lstm" (shared LSTM, no exchange between gauges). Both use the identical 9 history + 9 forcing
     channels, target, quantiles, loss, optimizer, epochs, early stopping and train/validation split.
  3. an explicit seed (101, 202 or 303) sets the weight initialization and batch order. The window
     shuffle that decides the 90/10 split keeps the original fixed generator (seed 0), so every model
     and seed shares the SAME training and early-stopping windows.
  4. the per-epoch loss history is written next to the weights.

Training data: 2023-01-01 00:00 to 2025-12-31 23:45 UTC (the matrix is sliced before any statistic).

Usage (inside the TF container on mike):  python -u train.py stgnn 101
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import numpy.core as _npc
    import numpy.core.numeric as _npn
    import numpy.core.multiarray as _npm
    sys.modules.setdefault("numpy._core", _npc)
    sys.modules.setdefault("numpy._core.numeric", _npn)
    sys.modules.setdefault("numpy._core.multiarray", _npm)
except Exception:
    pass

EXP = "project/hpc/Experiments/HRRR_FORCING_AND_LSTM_20260911"
sys.path.insert(0, EXP)
from lstm_models import build_model  # noqa: E402

MODEL_KIND = sys.argv[1]
SEED = int(sys.argv[2])
if MODEL_KIND not in ("stgnn", "lstm"):
    raise SystemExit("[FATAL] model kind must be stgnn or lstm")

PICKLE = os.path.join(EXP, "frozen_inputs", "global_features_all_stations_feature_engineered_20260911.pkl")
GRAPH = os.path.join(EXP, "frozen_inputs", "gnn_graph.npz")
STAGE_TO_RAIN = os.path.join(EXP, "frozen_inputs", "stage_to_rain_gauge.csv")
OUTD = os.path.join(EXP, "weights", f"{MODEL_KIND}_seed{SEED}")
TRAIN_END = "2025-12-31 23:45"
QUANTILES = [0.5, 0.6, 0.7, 0.8, 0.9]
TEND_LAG = 24
HL = {"api24": 96.0, "api48": 192.0, "api72": 288.0}
IN_W = 288
LB_W = 96
H = 64
IN_CH = 9
F_CH = 9
NQ = len(QUANTILES)
EPOCHS = 40
STRIDE = 4
BATCH = 16
os.makedirs(OUTD, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)
print(f"[config] kind={MODEL_KIND} seed={SEED} out={OUTD}", flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


pickle_sha = sha256_file(PICKLE)
graph_sha = sha256_file(GRAPH)
print(f"[frozen] pickle {PICKLE} sha256 {pickle_sha}", flush=True)
print(f"[frozen] graph  {GRAPH} sha256 {graph_sha}", flush=True)

g = np.load(GRAPH, allow_pickle=True)
nodes = [str(x) for x in g["nodes"]]
N = len(nodes)
A = g["A_norm"].astype("float32")
if MODEL_KIND == "lstm":
    A = np.eye(N, dtype="float32")
print(f"[graph] N={N} edges={int((g['A_norm'] > 0).sum())} (used only by the ST-GNN)", flush=True)

df = pd.read_pickle(PICKLE).sort_index()
print(f"[data] loaded rows={len(df)} cols={df.shape[1]} range={df.index.min()} -> {df.index.max()}", flush=True)
df = df[df.index <= pd.Timestamp(TRAIN_END).tz_localize("UTC")]
print(f"[split] <= {TRAIN_END} -> {len(df)} rows", flush=True)
if len(df) < 10000:
    raise SystemExit("[FATAL] too few rows before the cutoff")
T = len(df)
idxt = df.index

S = np.full((T, N), np.nan, "float32")
M = np.zeros((T, N), bool)
for i, n in enumerate(nodes):
    c = f"{n}_stage_ft"
    if c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        s = s.mask(s.rolling(96, center=True, min_periods=96).std() < 1e-6)
        S[:, i] = s.values
        M[:, i] = np.isfinite(s.values)

m2r = pd.read_csv(STAGE_TO_RAIN).set_index("stage")["nearest_rain"].to_dict()


def raincol(suffix):
    cols = [c for c in df.columns if c.endswith(suffix)]
    if cols:
        gm = pd.to_numeric(df[cols].mean(axis=1), errors="coerce").fillna(0).values
    else:
        gm = np.zeros(T)
    G = np.zeros((T, N), "float32")
    for i, n in enumerate(nodes):
        rc = f"{m2r.get(n, '')}{suffix}"
        if rc in df.columns:
            G[:, i] = pd.to_numeric(df[rc], errors="coerce").fillna(0).values
        else:
            G[:, i] = gm
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


API24 = api(HL["api24"])
API48 = api(HL["api48"])
API72 = api(HL["api72"])
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

meta = {"nodes": nodes, "mu": mu.tolist(), "sd": sd.tolist(), "r6": [r6m, r6s], "r24": [r24m, r24s],
        "api24": [a24m, a24s], "api48": [a48m, a48s], "api72": [a72m, a72s], "re": [rem, res_],
        "raccel": [ram, ras], "tend": [tendm, tends], "halflives": HL, "tend_lag": TEND_LAG,
        "feat_version": "DQ", "quantiles": QUANTILES, "IN_CH": IN_CH, "F_CH": F_CH,
        "IN_W": IN_W, "LB_W": LB_W, "H": H, "train_end": TRAIN_END,
        "graph_mode": "obs" if MODEL_KIND == "stgnn" else "none_nodewise",
        "model_kind": MODEL_KIND, "seed": SEED, "pickle": PICKLE, "pickle_sha256": pickle_sha,
        "graph": GRAPH, "graph_sha256": graph_sha,
        "experiment": "HRRR_FORCING_AND_LSTM_20260911 (frozen once-trained models; NOT production)"}
json.dump(meta, open(os.path.join(OUTD, "gnn_meta.json"), "w"))

ten = {k: tf.constant(v) for k, v in dict(Z=Z, R6=R6z, R24=R24z, A24=A24z, A48=A48z, A72=A72z,
                                          RE=REz, RA=RAz, TEND=TENDz, TS=TS, TC=TC).items()}
MT = tf.constant(M.astype("float32"))
starts = np.arange(0, T - IN_W - LB_W, STRIDE, dtype=np.int64)
np.random.default_rng(0).shuffle(starts)


def make(i):
    h = tf.stack([ten["Z"][i:i + IN_W], ten["R6"][i:i + IN_W], ten["R24"][i:i + IN_W], ten["A24"][i:i + IN_W],
                  ten["A48"][i:i + IN_W], ten["A72"][i:i + IN_W], ten["TEND"][i:i + IN_W],
                  ten["RE"][i:i + IN_W], ten["RA"][i:i + IN_W]], -1)
    f0 = i + IN_W
    t0 = f0 - 1
    fts = tf.tile(ten["TS"][f0:f0 + LB_W, None], [1, N])
    ftc = tf.tile(ten["TC"][f0:f0 + LB_W, None], [1, N])
    forcing = tf.stack([ten["R6"][f0:f0 + LB_W], ten["R24"][f0:f0 + LB_W], ten["A24"][f0:f0 + LB_W],
                        ten["A48"][f0:f0 + LB_W], ten["A72"][f0:f0 + LB_W], ten["RE"][f0:f0 + LB_W],
                        ten["RA"][f0:f0 + LB_W], fts, ftc], -1)
    z0 = ten["Z"][t0]
    tgt = ten["Z"][f0:f0 + LB_W] - z0[None, :]
    w = MT[f0:f0 + LB_W] * MT[t0][None, :]
    return (h, forcing, z0), tgt, w


ds = tf.data.Dataset.from_tensor_slices(starts).map(make, num_parallel_calls=tf.data.AUTOTUNE)
nval = max(1, int(len(starts) * 0.1))
val = ds.take(nval).batch(BATCH).prefetch(2)
trn = ds.skip(nval).batch(BATCH).shuffle(256, seed=SEED).prefetch(2)
print(f"[data] T={T} windows={len(starts)} validation windows={nval} batch={BATCH}", flush=True)

TAUS = tf.constant(QUANTILES, dtype=tf.float32)


def multi_pinball(y, yh):
    e = y[..., None] - yh
    pl = tf.maximum(TAUS * e, (TAUS - 1.0) * e)
    return tf.reduce_mean(pl, axis=-1)


model = build_model(MODEL_KIND, A, N, IN_W, LB_W, IN_CH, F_CH, H, NQ)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0), loss=multi_pinball)
cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)]
hist = model.fit(trn, validation_data=val, epochs=EPOCHS, callbacks=cb, verbose=2)
model.save_weights(os.path.join(OUTD, "gnn.weights.h5"))
history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
best_epoch = int(np.argmin(history["val_loss"])) + 1
json.dump({"history": history, "epochs_run": len(history["val_loss"]), "best_epoch": best_epoch,
           "best_val_loss": float(min(history["val_loss"])),
           "n_params": int(model.count_params())},
          open(os.path.join(OUTD, "train_history.json"), "w"), indent=1)
print(f"[SAVE] {OUTD}/gnn.weights.h5  epochs={len(history['val_loss'])} best={best_epoch} "
      f"params={model.count_params()}", flush=True)
print("FORECAST_MODEL_TRAIN_DONE", flush=True)
