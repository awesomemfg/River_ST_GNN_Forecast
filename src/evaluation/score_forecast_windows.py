"""Score one causal System B forecast run against observed stage.

Run sources
  04 output     pred_ft [O, 96, N, 3]; the P80 line (the paper's scored output) is used.
  05 output     predictions_ft [O, 96, G] (the Bi-LSTM's single trajectory).
  persistence   the last reported stage at t0 held for all 96 steps.

Observations use the same causal trailing flatline rule as System B training and inference.

Writes to --out-dir the files that evaluator writes (same names, columns and formulas, lines 888-979
and 1042-1206), so the manuscript's figure recipes can read a new run by changing only a path:
  reforecast_2026H1_lead_metrics.csv
  reforecast_2026H1_metrics.csv
  reforecast_2026H1_origin_window_metrics.csv
  reforecast_2026H1_h96_timeseries.csv
MBSA4570 (a duplicate of MBSA5881) is dropped from the lead, gauge and h+24 files, as in the evaluator.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
XUE_FROZEN_INPUTS = "project/hpc/Experiments/HRRR_FORCING_AND_LSTM_20260911/frozen_inputs"
FT2M = 0.3048
LB_W = 96
EXCLUDED_GAUGES = {"MBSA4570"}

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, help="04 or 05 output npz, or the word persistence")
parser.add_argument("--out-dir", required=True)
parser.add_argument("--pickle", default=os.path.join(XUE_FROZEN_INPUTS, "global_features_all_stations_feature_engineered_20260911.pkl"))
parser.add_argument("--graph", default=os.path.join(EXPERIMENT_ROOT, "frozen_assets", "graph_obs_pre2026_471edges.npz"))
parser.add_argument("--origin-start", default="2026-01-01 00:00")
parser.add_argument("--origin-end", default="2026-06-30 23:00")
parser.add_argument("--quantile", type=float, default=0.8)
args = parser.parse_args()
t_begin = time.time()
os.makedirs(args.out_dir, exist_ok=True)


def log(*parts):
    print("[score]", *parts, flush=True)


df = pd.read_pickle(args.pickle).sort_index()
if df.index.tz is not None:
    df.index = df.index.tz_convert("UTC").tz_localize(None)
idxt = pd.DatetimeIndex(df.index)
row_of = {t: i for i, t in enumerate(idxt)}

if args.run == "persistence":
    gauges = [str(n) for n in np.load(args.graph, allow_pickle=True)["nodes"]]
    origins = pd.date_range(args.origin_start, args.origin_end, freq="h")
    quantile_label = np.nan
    run_meta = {"source": "persistence: last reported stage at t0"}
else:
    z = np.load(args.run, allow_pickle=True)
    origins = pd.to_datetime(z["origins_utc"])
    if origins.tz is not None:
        origins = origins.tz_convert("UTC").tz_localize(None)
    if "pred_ft" in z:
        gauges = [str(n) for n in z["nodes"]]
        q_index = [float(q) for q in z["quantiles_saved"]].index(args.quantile)
        pred_all = z["pred_ft"][..., q_index].astype(np.float32)
        quantile_label = args.quantile
    else:
        gauges = [str(n) for n in z["target_gauges"]]
        pred_all = z["predictions_ft"].astype(np.float32)
        quantile_label = np.nan
    run_meta = json.loads(str(z["run_meta"])) if "run_meta" in z else {}
    if str(run_meta.get("system")) != "System B":
        raise ValueError("The scorer requires a forecast archive produced by System B.")
    if str(run_meta.get("postproc")) != "off":
        raise ValueError("System B scores must use raw forecasts with postprocessing disabled.")
    if str(run_meta.get("flatline_detection")) != "causal trailing 96-step standard deviation":
        raise ValueError("The System B forecast archive does not declare the causal flatline rule.")
    keep_origin = (origins >= pd.Timestamp(args.origin_start)) & (origins <= pd.Timestamp(args.origin_end))
    origins = origins[keep_origin]
    pred_all = pred_all[np.asarray(keep_origin)]

if args.origin_start == "2026-01-01 00:00" and args.origin_end == "2026-06-30 23:00":
    if len(origins) != 4344:
        raise ValueError("System B scoring requires exactly 4,344 January-June 2026 origins.")

N = len(gauges)
S = np.full((len(df), N), np.nan, np.float32)
S_RAW = np.full((len(df), N), np.nan, np.float32)
for i, gauge in enumerate(gauges):
    column = gauge + "_stage_ft"
    if column in df.columns:
        series = pd.to_numeric(df[column], errors="coerce")
        S_RAW[:, i] = series.values
        trailing_standard_deviation = series.rolling(
            window=96,
            center=False,
            min_periods=96,
        ).std()
        S[:, i] = series.mask(trailing_standard_deviation < 1e-6).values
    else:
        log("no stage column for", gauge)
positions = np.asarray([row_of[t] for t in origins])
if np.any(positions + LB_W >= len(df)):
    raise ValueError("The evaluation matrix ends before some verifying windows close.")
steps = positions[:, None] + 1 + np.arange(LB_W)[None, :]
y_true = S[steps]
issue_stage = S[positions]
if args.run == "persistence":
    y_pred = np.repeat(S_RAW[positions][:, None, :], LB_W, axis=1)
else:
    y_pred = pred_all
O = len(origins)
log("run", args.run, "origins", O, origins[0], "->", origins[-1], "gauges", N)


def pooled_metrics(true, pred, axis):
    """RMSE, MAE, bias, NSE, Pearson r and KGE from the evaluator's running sums, pooled over axis."""
    mask = np.isfinite(true) & np.isfinite(pred)
    t = np.where(mask, true, 0.0).astype(np.float64)
    p = np.where(mask, pred, 0.0).astype(np.float64)
    n = mask.sum(axis=axis).astype(np.float64)
    diff = p - t
    sse = (diff ** 2).sum(axis=axis)
    sae = np.abs(diff).sum(axis=axis)
    sum_y = t.sum(axis=axis)
    sum_p = p.sum(axis=axis)
    sum_y2 = (t ** 2).sum(axis=axis)
    sum_p2 = (p ** 2).sum(axis=axis)
    sum_yp = (t * p).sum(axis=axis)
    out = {
        key: np.full(n.shape, np.nan)
        for key in ("rmse", "mae", "bias", "ubrmse", "nse", "corr", "kge", "alpha", "beta")
    }
    valid = n > 0
    out["rmse"][valid] = np.sqrt(sse[valid] / n[valid])
    out["mae"][valid] = sae[valid] / n[valid]
    out["bias"][valid] = (sum_p[valid] - sum_y[valid]) / n[valid]
    out["ubrmse"][valid] = np.sqrt(
        np.maximum(out["rmse"][valid] ** 2 - out["bias"][valid] ** 2, 0.0)
    )
    den_y = sum_y2 - sum_y ** 2 / np.maximum(n, 1.0)
    den_p = sum_p2 - sum_p ** 2 / np.maximum(n, 1.0)
    nse_valid = valid & (den_y > 0)
    corr_valid = valid & (den_y > 0) & (den_p > 0)
    numerator = sum_yp - sum_y * sum_p / np.maximum(n, 1.0)
    out["nse"][nse_valid] = 1.0 - sse[nse_valid] / den_y[nse_valid]
    out["corr"][corr_valid] = numerator[corr_valid] / np.sqrt(den_y[corr_valid] * den_p[corr_valid])
    mean_y = np.where(valid, sum_y / np.maximum(n, 1.0), np.nan)
    mean_p = np.where(valid, sum_p / np.maximum(n, 1.0), np.nan)
    std_y = np.where(nse_valid, np.sqrt(np.maximum(den_y, 0.0) / np.maximum(n, 1.0)), np.nan)
    std_p = np.where(corr_valid, np.sqrt(np.maximum(den_p, 0.0) / np.maximum(n, 1.0)), np.nan)
    kge_valid = corr_valid & (np.abs(mean_y) > 1e-9) & (std_y > 0)
    alpha = std_p[kge_valid] / std_y[kge_valid]
    beta = mean_p[kge_valid] / mean_y[kge_valid]
    out["alpha"][kge_valid] = alpha
    out["beta"][kge_valid] = beta
    out["kge"][kge_valid] = 1.0 - np.sqrt((out["corr"][kge_valid] - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    out["n"] = n
    return out


keep = [i for i, g in enumerate(gauges) if g not in EXCLUDED_GAUGES]

# ---- per lead and gauge ----
lead = pooled_metrics(y_true, y_pred, axis=0)
lead_rows = []
for k in range(LB_W):
    for i in keep:
        lead_rows.append({"lead_step": k + 1, "lead_hours": (k + 1) * 0.25, "node": gauges[i],
                          "target": gauges[i] + "_stage_m", "source_stage_column": gauges[i] + "_stage_ft",
                          "quantile": quantile_label, "n": int(lead["n"][k, i]),
                          "RMSE_m": lead["rmse"][k, i] * FT2M, "MAE_m": lead["mae"][k, i] * FT2M,
                          "bias_m": lead["bias"][k, i] * FT2M,
                          "ubRMSE_m": lead["ubrmse"][k, i] * FT2M,
                          "NSE": lead["nse"][k, i], "Pearson_r": lead["corr"][k, i],
                          "KGE": lead["kge"][k, i], "KGE_r": lead["corr"][k, i],
                          "KGE_alpha": lead["alpha"][k, i], "KGE_beta": lead["beta"][k, i]})
pd.DataFrame(lead_rows).to_csv(os.path.join(args.out_dir, "reforecast_2026H1_lead_metrics.csv"), index=False)

# ---- per gauge: all leads pooled, and h+24 ----
pooled_all = pooled_metrics(y_true.reshape(-1, N), y_pred.reshape(-1, N), axis=0)
pooled_h96 = pooled_metrics(y_true[:, -1, :], y_pred[:, -1, :], axis=0)
metric_rows = []
for i in keep:
    metric_rows.append({"node": gauges[i], "target": gauges[i] + "_stage_m", "source_stage_column": gauges[i] + "_stage_ft",
                        "run": args.run, "quantile": quantile_label,
                        "n_all": int(pooled_all["n"][i]), "RMSE_all_m": pooled_all["rmse"][i] * FT2M,
                        "MAE_all_m": pooled_all["mae"][i] * FT2M, "bias_all_m": pooled_all["bias"][i] * FT2M,
                        "ubRMSE_all_m": pooled_all["ubrmse"][i] * FT2M,
                        "NSE_all": pooled_all["nse"][i], "Pearson_r_all": pooled_all["corr"][i],
                        "KGE_all": pooled_all["kge"][i], "KGE_r_all": pooled_all["corr"][i],
                        "KGE_alpha_all": pooled_all["alpha"][i], "KGE_beta_all": pooled_all["beta"][i],
                        "n_h96": int(pooled_h96["n"][i]), "RMSE_h96_m": pooled_h96["rmse"][i] * FT2M,
                        "MAE_h96_m": pooled_h96["mae"][i] * FT2M, "bias_h96_m": pooled_h96["bias"][i] * FT2M,
                        "ubRMSE_h96_m": pooled_h96["ubrmse"][i] * FT2M,
                        "NSE_h96": pooled_h96["nse"][i], "Pearson_r_h96": pooled_h96["corr"][i],
                        "KGE_h96": pooled_h96["kge"][i], "KGE_r_h96": pooled_h96["corr"][i],
                        "KGE_alpha_h96": pooled_h96["alpha"][i], "KGE_beta_h96": pooled_h96["beta"][i]})
pd.DataFrame(metric_rows).to_csv(os.path.join(args.out_dir, "reforecast_2026H1_metrics.csv"), index=False)

# ---- per origin, gauge and window ----
frames = []
for window_hours, window_steps in [(3, 12), (6, 24), (12, 48), (24, 96)]:
    wt = y_true[:, :window_steps, :]
    wp = y_pred[:, :window_steps, :]
    valid = np.isfinite(wt) & np.isfinite(wp)
    count = valid.sum(axis=1)
    err = np.where(valid, wp - wt, 0.0)
    rmse = np.full(count.shape, np.nan)
    mae = np.full(count.shape, np.nan)
    positive = count > 0
    rmse[positive] = np.sqrt((err ** 2).sum(axis=1)[positive] / count[positive]) * FT2M
    mae[positive] = np.abs(err).sum(axis=1)[positive] / count[positive] * FT2M
    data = {"t0_utc": np.repeat(origins.to_numpy(), N), "gauge": np.tile(np.asarray(gauges, dtype=object), O),
            "window_hours": window_hours, "n": count.reshape(-1), "RMSE_m": rmse.reshape(-1), "MAE_m": mae.reshape(-1)}
    if window_hours == 24:
        scorable = (count >= 4) & np.isfinite(issue_stage)
        obs_masked = np.where(valid, wt, -np.inf)
        obs_peak_index = np.argmax(obs_masked, axis=1)
        obs_peak = np.take_along_axis(obs_masked, obs_peak_index[:, None, :], axis=1)[:, 0, :]
        lead_index = np.arange(window_steps)[None, :, None]
        near = np.abs(lead_index - obs_peak_index[:, None, :]) <= 16
        pred_masked = np.where(valid & near, wp, -np.inf)
        pred_peak_index = np.argmax(pred_masked, axis=1)
        pred_peak = np.take_along_axis(pred_masked, pred_peak_index[:, None, :], axis=1)[:, 0, :]
        scorable = scorable & np.isfinite(pred_peak)
        observed_rise = np.full(scorable.shape, np.nan, dtype=np.float64)
        peak_bias = np.full(scorable.shape, np.nan, dtype=np.float64)
        observed_rise[scorable] = (obs_peak[scorable] - issue_stage[scorable]) * FT2M
        peak_bias[scorable] = (pred_peak[scorable] - obs_peak[scorable]) * FT2M
        data["observed_rise_m"] = observed_rise.reshape(-1)
        data["peak_absolute_error_m"] = np.abs(peak_bias).reshape(-1)
        data["peak_bias_m"] = peak_bias.reshape(-1)
        data["peak_timing_error_hours"] = np.where(scorable, (pred_peak_index - obs_peak_index) * 0.25, np.nan).reshape(-1)
    frames.append(pd.DataFrame(data))
pd.concat(frames, ignore_index=True).to_csv(os.path.join(args.out_dir, "reforecast_2026H1_origin_window_metrics.csv"), index=False)

# ---- h+24 time series ----
h96_rows = []
verify_times = origins + pd.Timedelta(hours=24)
for i in keep:
    h96_rows.append(pd.DataFrame({"timestamp_utc": verify_times, "node": gauges[i], "target": gauges[i] + "_stage_m",
                                  "source_stage_column": gauges[i] + "_stage_ft", "lead_step": LB_W, "lead_hours": 24.0,
                                  "quantile": quantile_label, "observed_stage_m": y_true[:, -1, i] * FT2M,
                                  "predicted_stage_m": y_pred[:, -1, i] * FT2M}))
pd.concat(h96_rows, ignore_index=True).to_csv(os.path.join(args.out_dir, "reforecast_2026H1_h96_timeseries.csv"), index=False)

json.dump({"run": args.run, "origins": O, "first_origin": str(origins[0]), "last_origin": str(origins[-1]),
           "gauges": N, "quantile": quantile_label, "run_meta": run_meta, "pickle": args.pickle,
           "seconds": time.time() - t_begin}, open(os.path.join(args.out_dir, "score_meta.json"), "w"), indent=1, default=str)
log("wrote", args.out_dir, f"{time.time() - t_begin:.0f} s")
print("SCORE_DONE", flush=True)
