"""Run a frozen System B model at every January-June 2026 hourly origin.

Forcing over the 24 h horizon is either
  perfect   observed rain (the paper's hindcast, an upper bound), or
  <npz>     a per-origin rain forecast file: the HRRR history built by hrrr_issue_time_forcing.py,
            or the rain the live system shipped (03_validate_hrrr_vs_shipped.py). Before t0 every input
            stays observed; rolling rain sums that straddle t0 mix observed and forecast steps exactly,
            as in project/Experiments/REFORECAST_2026H1/<SERVER>/
            eval_reforecast_2026H1_perfectQPF.py (hybrid_forcing), except that the forecast now changes
            with each origin.

System B uses causal, trailing-only flatline detection. The historical evaluator used a centered
window that could inspect future stage values. System B prohibits postprocessing so every model is
compared using raw output.

Output npz: origins_utc, nodes, quantiles_saved (0.5, 0.8, 0.9), pred_ft [O, 96, N, 3] (float32),
issue_stage_ft [O, N] (raw stage at t0, for persistence), obs_ft [O, 96, N] (only with --save-obs),
plus run metadata. Units are feet (the model's native units); scoring converts to meters.

Example:
  conda run -n operational python -u lstm_inference_before_fix.py --kind stgnn \
     --model-dir project/hpc_home/.../weights/stgnn_seed101 \
     --forcing project/home/.../data/forcing/hrrr_issue_time_stitch_L2.npz --out ../runs/stgnn_seed101_hrrr.npz
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
XUE_EXPERIMENT = "project/hpc/Experiments/HRRR_FORCING_AND_LSTM_20260911"
XUE_FROZEN_INPUTS = os.path.join(XUE_EXPERIMENT, "frozen_inputs")

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", required=True)
parser.add_argument("--kind", choices=["stgnn", "lstm"], required=True)
parser.add_argument("--forcing", default="perfect")
parser.add_argument("--origin-start", default="2026-01-01 00:00")
parser.add_argument("--origin-end", default="2026-06-30 23:00")
parser.add_argument("--postproc", choices=["off", "on"], default="off")
parser.add_argument("--out", required=True)
parser.add_argument("--pickle", default=os.path.join(XUE_FROZEN_INPUTS, "global_features_all_stations_feature_engineered_20260911.pkl"))
parser.add_argument("--graph", default=os.path.join(EXPERIMENT_ROOT, "frozen_assets", "graph_obs_pre2026_471edges.npz"))
parser.add_argument("--stage-to-rain", default=os.path.join(EXPERIMENT_ROOT, "frozen_assets", "stage_to_rain_gauge.csv"))
parser.add_argument("--caps-json", default="config/rain_rise_caps.json")
parser.add_argument("--threads", type=int, default=16)
parser.add_argument("--save-obs", action="store_true")
args = parser.parse_args()
if args.postproc != "off":
    raise ValueError("System B prohibits postprocessing. Use --postproc off.")

import tensorflow as tf  # noqa: E402

tf.config.threading.set_intra_op_parallelism_threads(args.threads)
tf.config.threading.set_inter_op_parallelism_threads(2)
sys.path.insert(0, SCRIPT_DIRECTORY)
from stgnn_models import build_model  # noqa: E402

t_begin = time.time()
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)


def log(*parts):
    print("[eval]", *parts, flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


with open(os.path.join(args.model_dir, "gnn_meta.json"), "r", encoding="utf-8") as metadata_handle:
    meta = json.load(metadata_handle)
if str(meta.get("system")) != "System B" or str(meta.get("stage")) != "final":
    raise ValueError("The evaluator requires final System B weights.")
if str(meta.get("model_kind")) != str(args.kind):
    raise ValueError(
        "The requested evaluator kind does not match the fitted model metadata. "
        + "Requested: "
        + str(args.kind)
        + "; metadata: "
        + str(meta.get("model_kind"))
    )
if str(meta.get("feat_version")) != "DQ" or int(meta["IN_CH"]) != 9 or int(meta["F_CH"]) != 9:
    raise ValueError("System B requires the nine-channel DQ recipe.")
if int(meta.get("rain6_steps", -1)) != 6 or float(meta.get("rain6_hours", -1.0)) != 1.5:
    raise ValueError("System B rain6 must remain a 6-step, 1.5-hour accumulation.")
if int(meta.get("rain24_steps", -1)) != 24 or float(meta.get("rain24_hours", -1.0)) != 6.0:
    raise ValueError("System B rain24 must remain a 24-step, 6-hour accumulation.")
if int(meta.get("decoder_steps", -1)) != 96 or float(meta.get("decoder_hours", -1.0)) != 24.0:
    raise ValueError("System B must retain the 96-step, 24-hour decoder trajectory.")
nodes = [str(x) for x in meta["nodes"]]
quantiles = [float(q) for q in meta["quantiles"]]
q_keep = [quantiles.index(q) for q in (0.5, 0.8, 0.9)]
IN_W = int(meta["IN_W"])
LB_W = int(meta["LB_W"])
H = int(meta["H"])
IN_CH = int(meta["IN_CH"])
F_CH = int(meta["F_CH"])
N = len(nodes)
mu = np.asarray(meta["mu"], np.float32)
sd = np.asarray(meta["sd"], np.float32)
r6m, r6s = meta["r6"]
r24m, r24s = meta["r24"]
a24m, a24s = meta["api24"]
a48m, a48s = meta["api48"]
a72m, a72s = meta["api72"]
rem, res = meta["re"]
ram, ras = meta["raccel"]
tendm, tends = meta["tend"]
tend_lag = int(meta["tend_lag"])
halflives = {k: float(v) for k, v in meta["halflives"].items()}

graph = np.load(args.graph, allow_pickle=True)
if [str(x) for x in graph["nodes"]] != nodes:
    raise ValueError("Graph node order differs from the model metadata.")
graph_sha256 = sha256_file(args.graph)
if graph_sha256 != str(meta.get("graph_sha256")):
    raise ValueError("Evaluation graph hash differs from the graph used for final fitting.")
A = graph["A_norm"].astype("float32")
if args.kind == "stgnn":
    actual_graph_nonzero_count = int(np.count_nonzero(A))
    expected_graph_nonzero_count = int(meta.get("graph_nonzero_count", -1))
    if expected_graph_nonzero_count <= 0:
        raise ValueError("System B ST-GNN metadata does not declare a positive graph edge count.")
    if actual_graph_nonzero_count != expected_graph_nonzero_count:
        raise ValueError(
            "Evaluation graph edge count differs from the graph used for final fitting. "
            + "Expected: "
            + str(expected_graph_nonzero_count)
            + "; found: "
            + str(actual_graph_nonzero_count)
        )
if args.kind == "lstm":
    A = np.eye(N, dtype="float32")

df = pd.read_pickle(args.pickle).sort_index()
if df.index.tz is not None:
    df.index = df.index.tz_convert("UTC").tz_localize(None)
T = len(df)
idxt = pd.DatetimeIndex(df.index)
log("matrix", args.pickle, df.shape, idxt.min(), "->", idxt.max())

S = np.full((T, N), np.nan, np.float32)
S_RAW = np.full((T, N), np.nan, np.float32)
for i, node in enumerate(nodes):
    column = node + "_stage_ft"
    if column in df.columns:
        series = pd.to_numeric(df[column], errors="coerce")
        S_RAW[:, i] = series.values
        trailing_standard_deviation = series.rolling(
            window=96,
            center=False,
            min_periods=96,
        ).std()
        series = series.mask(trailing_standard_deviation < 1e-6)
        S[:, i] = series.values

stage_to_rain = pd.read_csv(args.stage_to_rain).set_index("stage")["nearest_rain"].to_dict()


def raincol(suffix):
    cols = [c for c in df.columns if c.endswith(suffix)]
    if cols:
        gm = pd.to_numeric(df[cols].mean(axis=1), errors="coerce").fillna(0).values.astype(np.float32)
    else:
        gm = np.zeros(T, np.float32)
    values = np.zeros((T, N), np.float32)
    for i, node in enumerate(nodes):
        column = str(stage_to_rain.get(node, "")) + suffix
        if column in df.columns:
            values[:, i] = pd.to_numeric(df[column], errors="coerce").fillna(0).values.astype(np.float32)
        else:
            values[:, i] = gm
    return values


def api_from_rain(rain_step, halflife):
    k = float(0.5 ** (1.0 / halflife))
    values = np.zeros((T, N), np.float32)
    acc = np.zeros(N, np.float32)
    for row in range(T):
        acc = k * acc + rain_step[row]
        values[row] = acc
    return values


R6 = raincol("_rain_in_accum_6h")
R24 = raincol("_rain_in_accum_24h")
RSTEP = raincol("_rain_in")
API24 = api_from_rain(RSTEP, halflives["api24"])
API48 = api_from_rain(RSTEP, halflives["api48"])
API72 = api_from_rain(RSTEP, halflives["api72"])
RE = R6 * API48
RACCEL = np.zeros((T, N), np.float32)
RACCEL[4:] = RSTEP[4:] - RSTEP[:-4]
Z = np.nan_to_num((S - mu[None, :]) / sd[None, :]).astype(np.float32)
TEND = np.zeros((T, N), np.float32)
TEND[tend_lag:] = Z[tend_lag:] - Z[:-tend_lag]


def zs(values, mean_value, std_value):
    return ((values - float(mean_value)) / float(std_value)).astype(np.float32)


R6z = zs(R6, r6m, r6s)
R24z = zs(R24, r24m, r24s)
A24z = zs(API24, a24m, a24s)
A48z = zs(API48, a48m, a48s)
A72z = zs(API72, a72m, a72s)
REz = zs(RE, rem, res)
RAz = zs(RACCEL, ram, ras)
TENDz = zs(TEND, tendm, tends)
hour_of_day = idxt.hour.to_numpy(np.float32) + idxt.minute.to_numpy(np.float32) / 60.0
TS = np.sin(2 * np.pi * hour_of_day / 24).astype(np.float32)
TC = np.cos(2 * np.pi * hour_of_day / 24).astype(np.float32)
CUMO = np.zeros((T + 1, N), np.float64)
CUMO[1:] = np.cumsum(RSTEP, axis=0)

# ---- origins ----
wanted = pd.date_range(args.origin_start, args.origin_end, freq="h")
FORECAST = None
if args.forcing != "perfect":
    fz = np.load(args.forcing, allow_pickle=True)
    f_origins = pd.to_datetime(fz["origins_utc"])
    if f_origins.tz is not None:
        f_origins = f_origins.tz_convert("UTC").tz_localize(None)
    f_pos = {t: i for i, t in enumerate(f_origins)}
    wanted = pd.DatetimeIndex([t for t in wanted if t in f_pos])
    codes = [str(c) for c in fz["rain_codes"]]
    code_pos = {c: i for i, c in enumerate(codes)}
    rain_all = fz["rain_in"].astype(np.float32)
    FORECAST = {"rain": rain_all, "pos": f_pos, "node_code": [code_pos.get(str(stage_to_rain.get(n, "")), -1) for n in nodes]}
    log("forcing", args.forcing, "origins in file", len(f_origins), "nodes without a mapped code",
        sum(1 for c in FORECAST["node_code"] if c < 0))
row_of = {t: i for i, t in enumerate(idxt)}
origins = [t for t in wanted if t in row_of and row_of[t] - IN_W + 1 >= 0 and row_of[t] + LB_W < T]
log("origins", len(origins), origins[0], "->", origins[-1])
if args.origin_start == "2026-01-01 00:00" and args.origin_end == "2026-06-30 23:00":
    if len(origins) != 4344:
        raise ValueError("System B requires exactly 4,344 January-June 2026 origins.")


def node_forecast_rain(o_time):
    block = FORECAST["rain"][FORECAST["pos"][o_time]]
    step_mean = np.nanmean(block, axis=1)
    step_mean = np.nan_to_num(step_mean)
    q = np.zeros((LB_W, N), np.float32)
    for i, c in enumerate(FORECAST["node_code"]):
        if c >= 0:
            column = block[:, c]
            q[:, i] = np.where(np.isfinite(column), column, step_mean)
        else:
            q[:, i] = step_mean
    return np.clip(q, 0.0, None)


def forecast_forcing(p, qo):
    t0 = p
    f0 = p + 1
    lead = np.arange(LB_W)
    cq = np.zeros((LB_W + 1, N), np.float64)
    cq[1:] = np.cumsum(qo, axis=0)

    def hybrid_sum(window):
        n_q = np.minimum(lead + 1, window)
        n_o = window - n_q
        q_part = cq[lead + 1] - cq[lead + 1 - n_q]
        o_part = CUMO[t0 + 1] - CUMO[t0 + 1 - n_o]
        return (q_part + o_part).astype(np.float32)

    r6 = hybrid_sum(6)
    r24 = hybrid_sum(24)
    api = {}
    for name, value_at_t0 in (("api24", API24[t0]), ("api48", API48[t0]), ("api72", API72[t0])):
        decay = float(0.5 ** (1.0 / halflives[name]))
        acc = value_at_t0.astype(np.float64).copy()
        track = np.zeros((LB_W, N), np.float32)
        for k in range(LB_W):
            acc = decay * acc + qo[k]
            track[k] = acc
        api[name] = track
    re_ = r6 * api["api48"]
    back = np.zeros((LB_W, N), np.float32)
    back[4:] = qo[:-4]
    back[:4] = RSTEP[f0 - 4:f0]
    raccel = qo - back
    fts = np.tile(TS[f0:f0 + LB_W, None], (1, N))
    ftc = np.tile(TC[f0:f0 + LB_W, None], (1, N))
    return np.stack([(r6 - r6m) / r6s, (r24 - r24m) / r24s, (api["api24"] - a24m) / a24s,
                     (api["api48"] - a48m) / a48s, (api["api72"] - a72m) / a72s, (re_ - rem) / res,
                     (raccel - ram) / ras, fts, ftc], axis=-1).astype(np.float32)


# ---- postprocessing (copied from eval_reforecast_2026H1_perfectQPF.py) ----
POST = args.postproc == "on"
FG_TRIG = 12
FG_NBACK = 8
FG_DAMP = 0.5
ANCH_STEPS = 12
CAP_RELAX_TRIG = 1.0
CAP_RELAX_MULT = 1.5
CAP_FLOOR = 0.5
CAPS = None
FROZEN_RUN = None
if POST:
    CAPS = json.load(open(args.caps_json))
    FROZEN_RUN = np.zeros((T, N), np.int32)
    for i in range(N):
        column = S[:, i]
        count = 0
        for t in range(T):
            if t > 0 and np.isfinite(column[t]) and np.isfinite(column[t - 1]) and abs(column[t] - column[t - 1]) < 1e-6:
                count += 1
            else:
                count = 0
            FROZEN_RUN[t, i] = count


def repair_history(start, s_hist):
    t0 = start + IN_W - 1
    fixed = 0
    for i in range(N):
        c = int(FROZEN_RUN[t0, i]) + 1
        if c < FG_TRIG:
            continue
        base_idx = t0 - c
        if base_idx < 0 or base_idx - FG_NBACK < 0:
            continue
        base = S[base_idx, i]
        prior = S[base_idx - FG_NBACK, i]
        if not (np.isfinite(base) and np.isfinite(prior)):
            continue
        slope = (base - prior) / float(FG_NBACK)
        if abs(slope) < 1e-9:
            continue
        lo = max(base_idx + 1 - start, 0)
        offsets = np.arange(start + lo, start + IN_W) - base_idx
        s_hist[lo:IN_W, i] = (base + FG_DAMP * slope * offsets).astype(np.float32)
        fixed += 1
    return fixed


model = build_model(args.kind, A, N, IN_W, LB_W, IN_CH, F_CH, H, len(quantiles))
model([np.zeros((1, IN_W, N, IN_CH), np.float32), np.zeros((1, LB_W, N, F_CH), np.float32),
       np.zeros((1, N), np.float32)], training=False)
model.load_weights(os.path.join(args.model_dir, "gnn.weights.h5"))
log("weights loaded", args.model_dir, "kind", args.kind)

O = len(origins)
pred = np.zeros((O, LB_W, N, 3), np.float32)
obs = np.zeros((O, LB_W, N), np.float32) if args.save_obs else None
issue_stage = np.zeros((O, N), np.float32)
BATCH = 32
n_repair = 0
n_cap = 0
for b0 in range(0, O, BATCH):
    batch_times = origins[b0:b0 + BATCH]
    B = len(batch_times)
    hist = np.zeros((B, IN_W, N, IN_CH), np.float32)
    forc = np.zeros((B, LB_W, N, F_CH), np.float32)
    z0 = np.zeros((B, N), np.float32)
    horizon_rain = np.zeros((B, N), np.float32)
    for r, t in enumerate(batch_times):
        p = row_of[t]
        start = p - IN_W + 1
        z_win = Z[start:p + 1]
        tendz_win = TENDz[start:p + 1]
        if POST:
            s_hist = S[start:p + 1].copy()
            fixed = repair_history(start, s_hist)
            if fixed:
                n_repair += fixed
                z_win = np.nan_to_num((s_hist - mu[None, :]) / sd[None, :]).astype(np.float32)
                tend_win = np.zeros((IN_W, N), np.float32)
                tend_win[tend_lag:] = z_win[tend_lag:] - z_win[:-tend_lag]
                tend_win[:tend_lag] = z_win[:tend_lag] - Z[start - tend_lag:start]
                tendz_win = ((tend_win - float(tendm)) / float(tends)).astype(np.float32)
        hist[r] = np.stack([z_win, R6z[start:p + 1], R24z[start:p + 1], A24z[start:p + 1], A48z[start:p + 1],
                            A72z[start:p + 1], tendz_win, REz[start:p + 1], RAz[start:p + 1]], axis=-1)
        if FORECAST is None:
            f0 = p + 1
            fts = np.tile(TS[f0:f0 + LB_W, None], (1, N))
            ftc = np.tile(TC[f0:f0 + LB_W, None], (1, N))
            forc[r] = np.stack([R6z[f0:f0 + LB_W], R24z[f0:f0 + LB_W], A24z[f0:f0 + LB_W], A48z[f0:f0 + LB_W],
                                A72z[f0:f0 + LB_W], REz[f0:f0 + LB_W], RAz[f0:f0 + LB_W], fts, ftc], axis=-1)
            horizon_rain[r] = RSTEP[f0:f0 + LB_W].sum(0)
        else:
            qo = node_forecast_rain(t)
            forc[r] = forecast_forcing(p, qo)
            horizon_rain[r] = qo.sum(0)
        z0[r] = z_win[-1]
        issue_stage[b0 + r] = S_RAW[p]
        if obs is not None:
            obs[b0 + r] = S[p + 1:p + 1 + LB_W]
    out_q = model([hist, forc, z0], training=False).numpy()
    y = (out_q + z0[:, None, :, None]) * sd[None, None, :, None] + mu[None, None, :, None]
    y = y[..., q_keep]
    if POST:
        obs0 = np.stack([S[row_of[t]] for t in batch_times])
        w = np.clip(1.0 - np.arange(LB_W) / ANCH_STEPS, 0.0, 1.0).astype(np.float32)[None, :, None]
        fin = np.isfinite(obs0)[:, None, :]
        wa = np.where(fin, w, 0.0)[..., None]
        y = np.where(fin[..., None], wa * np.nan_to_num(obs0)[:, None, :, None] + (1.0 - wa) * y, y)
        for r, t in enumerate(batch_times):
            p = row_of[t]
            for i, node in enumerate(nodes):
                c = CAPS.get(node)
                if c is None:
                    continue
                allow = np.asarray([np.inf if v is None else v for v in c["allow"]], "float64")
                if not np.all(np.isfinite(allow)):
                    continue
                edges = np.asarray(c["edges"], "float64")
                dry = np.inf if c["dry"] is None else float(c["dry"])
                for k in range(3):
                    anchor_value = float(y[r, 0, i, k])
                    rise = float(np.nanmax(y[r, :, i, k])) - anchor_value
                    if not np.isfinite(rise) or rise <= 0:
                        continue
                    base = max(dry, float(np.interp(horizon_rain[r, i], edges, allow, left=allow[0], right=allow[-1])))
                    base = max(base, CAP_FLOOR)
                    own6 = float(S[p, i] - S[p - 24, i]) if p - 24 >= 0 else np.nan
                    if np.isfinite(own6) and own6 >= CAP_RELAX_TRIG:
                        base = base * CAP_RELAX_MULT
                    if rise > base:
                        y[r, :, i, k] = anchor_value + (y[r, :, i, k] - anchor_value) * (base / rise)
                        n_cap += 1
    pred[b0:b0 + B] = y.astype(np.float32)
    if (b0 // BATCH) % 20 == 0:
        log(f"origins {b0 + B}/{O}  elapsed {time.time() - t_begin:.0f} s")

save = {"origins_utc": np.asarray([t.isoformat() for t in origins]), "nodes": np.asarray(nodes),
        "quantiles_saved": np.asarray([0.5, 0.8, 0.9]), "pred_ft": pred, "issue_stage_ft": issue_stage,
        "run_meta": json.dumps({"system": "System B", "model_dir": args.model_dir, "kind": args.kind, "forcing": args.forcing,
                                "forcing_sha256": sha256_file(args.forcing) if args.forcing != "perfect" else "",
                                "postproc": args.postproc, "pickle": args.pickle,
                                "pickle_sha256": sha256_file(args.pickle), "graph": args.graph,
                                "graph_sha256": graph_sha256,
                                "flatline_detection": "causal trailing 96-step standard deviation",
                                "n_repair_gauge_origins": n_repair, "n_cap_applied": n_cap,
                                "seconds": time.time() - t_begin})}
if obs is not None:
    save["obs_ft"] = obs
np.savez_compressed(args.out, **save)
log("wrote", args.out, f"{time.time() - t_begin:.0f} s")
print("EVAL_DONE", flush=True)
