"""Event-conditioned versions of the rainfall-forcing results: Table 6, Figure 13 and Figure 14.

Section 4.1 asks how sensitive the ST-GNN is to the rainfall forecast. The published version answers
it over all 4,344 January to June origins. This script answers the same question on event origins
only, using the event definitions agreed for this experiment:

  all            every origin
  pump_at_issue  any pump in the parish already running at issue time
  stage_p80      per gauge, observed stage in the 24 h window reaches that gauge's 80th percentile
  stage_p90      the same at the 90th percentile

Both stage thresholds come from the record through 31 December 2025, so the event definition never
sees the scored period, following Nearing et al. (2022).

No model is run again. Every rainfall forcing was already scored per origin, and this reads those
scores:

  reforecast_2026H1_h96_timeseries.csv        per origin, per gauge: observed and predicted stage at
                                              the fixed h+24 lead. The file is stamped with the
                                              verifying time, so the origin is that minus 24 h.
  reforecast_2026H1_origin_window_metrics.csv per origin, per gauge, for the 24 h window: the observed
                                              rise, the peak absolute error and the peak bias.

Outputs, all written next to this experiment:
  event_table6_skill_by_forcing.csv   Table 6 on events: h+24 RMSE, correlation, NSE and KGE
  event_fig13_per_gauge_h24.csv       per-gauge values behind the Figure 13 boxes
  event_fig14_crest_by_forcing.csv    Figure 14 on events: crest errors during rising stage
  EVENT_RAINFALL_REPORT.txt

Aggregation follows the manuscript's locked order: average each gauge over the three seeds first,
then take the median across gauges. Crest statistics are computed per seed and then averaged, which
is what 06_aggregate.py does for the published figure.

Usage:
  conda run -n operational python -u rainfall_forcing_skill.py
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
OUTPUT_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "outputs")

SYSTEM_B_SCORED = (
    "project/Experiments/"
    "SYSTEM_B_471_CHRONOLOGICAL_20260913/scored"
)
ENSEMBLE_SCORED = (
    "project/Experiments/"
    "SYSTEM_B_ENSEMBLE_FORCING_20260914/scored"
)
MATRIX = (
    "project/hpc/Experiments/"
    "HRRR_FORCING_AND_LSTM_20260911/frozen_inputs/"
    "global_features_all_stations_feature_engineered_20260911.pkl"
)
INPARISH_MANIFEST = (
    "project/Experiments/INPARISH51_20260813/"
    "outputs/in_parish_gauges.json"
)

SEEDS = [101, 202, 303]
LB_W = 96
EXCLUDED_GAUGES = {"MBSA4570"}
THRESHOLD_RECORD_END = "2025-12-31 23:45"
EVENT_PERCENTILES = [80.0, 90.0]
MINIMUM_SAMPLES = 30

# Rise and crest conventions copied from 06_aggregate.py so the event figure matches the published one.
RISE_THRESHOLD_M = 0.1524
STRONG_RISE_THRESHOLD_M = 0.6096
UNDERCALL_M = -0.1524
RISE_BANDS = [
    ("0.15-0.30 m", 0.1524, 0.3048),
    ("0.30-0.61 m", 0.3048, 0.6096),
    ("above 0.61 m", 0.6096, np.inf),
]

# label -> (scored root, directory template). The order is the published table's order.
FORCINGS = [
    ("Observed rain", SYSTEM_B_SCORED, "stgnn_obs_pre2026_seed{seed}_observed"),
    ("Ensemble mean, 6", ENSEMBLE_SCORED, "ens6_mean_seed{seed}"),
    ("Ensemble median, 6", ENSEMBLE_SCORED, "ens6_median_seed{seed}"),
    ("Ensemble mean, 8", ENSEMBLE_SCORED, "ens8_mean_seed{seed}"),
    ("JMA", ENSEMBLE_SCORED, "jma_seamless_seed{seed}"),
    ("NBM", ENSEMBLE_SCORED, "ncep_nbm_conus_seed{seed}"),
    ("GEM", ENSEMBLE_SCORED, "gem_global_seed{seed}"),
    ("GEFS mean", ENSEMBLE_SCORED, "gefs_geavg_seed{seed}"),
    ("ECMWF IFS", ENSEMBLE_SCORED, "ecmwf_ifs025_seed{seed}"),
    ("GFS", ENSEMBLE_SCORED, "gfs_global_seed{seed}"),
    ("HRRR", ENSEMBLE_SCORED, "gfs_hrrr_seed{seed}"),
    ("GEFS pmm", ENSEMBLE_SCORED, "gefs_pmm_seed{seed}"),
    ("HRRR, issue time", SYSTEM_B_SCORED, "stgnn_obs_pre2026_seed{seed}_hrrr"),
]

# The score directories above cover January to June. A longer period needs its own scored tables, so
# the root is an argument. Without it this script silently reports January to June crest counts for
# whatever period it is asked about. The defaults reproduce the January to June result exactly.
parser = argparse.ArgumentParser()
parser.add_argument("--scored-root", default="", help="score directories for one period")
parser.add_argument("--members-root", default="",
                    help="score directories of the individual weather models for the same period")
parser.add_argument("--graphs-root", default="",
                    help="score directories of the six graphs, all with observed rainfall")
parser.add_argument("--tag", default="", help="suffix for the output files")
arguments = parser.parse_args()
TAG = arguments.tag

# Section 3.1 ends by saying that the selected graph does not lead every peak measure: the HEC-RAS
# graph had the lower mean peak error for rises of at least 0.15 m, and missed the crest less often.
# That claim compares graphs, not rainfall forcings, and all six graphs run with observed rainfall.
# The labels are the ones Table 3 prints, so the two can be read together.
GRAPH_LABELS = [
    ("Graph: observation lead-lag (selected)", "gObsLagLead"),
    ("Graph: observation, within-basin", "gObsBasin"),
    ("Graph: DEM downslope", "gDem"),
    ("Graph: DEM, within-basin", "gDemBasin"),
    ("Graph: HEC-RAS hydraulic", "gHecras"),
    ("Graph: identity", "gIdentity"),
]

# Section 4.1 also states the crest share of each weather-model member. Those members are scored
# under their own names in a separate root, so they are added only when that root is given.
#
# gfs_hrrr is Open-Meteo's blended GFS and HRRR product. It is a different forcing from the native
# issue-time HRRR stitch, and the two do not score alike, 0.1851 m against 0.1907 m at h+24. It is
# never labelled plain "HRRR" here, because the manuscript uses that word for the native forcing.
MEMBER_LABELS = [
    ("JMA", "jma_seamless"),
    ("NBM", "ncep_nbm_conus"),
    ("GEM", "gem_global"),
    ("GEFS mean", "gefs_geavg"),
    ("ECMWF IFS", "ecmwf_ifs025"),
    ("GFS", "gfs_global"),
    ("GFS-HRRR blend", "gfs_hrrr"),
    ("GEFS pmm", "gefs_pmm"),
    ("Ensemble median, 6", "ens6_median"),
    ("Ensemble mean, 8", "ens8_mean"),
]

if arguments.scored_root:
    FORCINGS = [
        ("Observed rain", arguments.scored_root, "observed_seed{seed}"),
        ("HRRR, issue time", arguments.scored_root, "hrrr_seed{seed}"),
        ("Ensemble mean, 6", arguments.scored_root, "ens6_mean_seed{seed}"),
    ]
    if arguments.members_root:
        FORCINGS += [(label, arguments.members_root, name + "_seed{seed}")
                     for label, name in MEMBER_LABELS]
    if arguments.graphs_root:
        FORCINGS += [(label, arguments.graphs_root, name + "_seed{seed}")
                     for label, name in GRAPH_LABELS]

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def log(*parts):
    print("[event-rain]", *parts, flush=True)


frame = pd.read_pickle(MATRIX).sort_index()
if frame.index.tz is not None:
    frame.index = frame.index.tz_convert("UTC").tz_localize(None)
index_times = pd.DatetimeIndex(frame.index)
row_of = {time_point: position for position, time_point in enumerate(index_times)}
log("matrix", frame.shape)

with open(INPARISH_MANIFEST, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
if isinstance(manifest, dict):
    for key in ("inparish51", "gauges", "nodes", "stations"):
        if key in manifest:
            scored_gauges = [str(value) for value in manifest[key]]
            break
    else:
        scored_gauges = [str(value) for value in list(manifest.values())[0]]
else:
    scored_gauges = [str(value) for value in manifest]
scored_gauges = [gauge for gauge in scored_gauges if gauge not in EXCLUDED_GAUGES]
gauge_position = {gauge: position for position, gauge in enumerate(scored_gauges)}
log("scored gauges", len(scored_gauges))

stage_columns = []
for gauge in scored_gauges:
    series = pd.to_numeric(frame[gauge + "_stage_ft"], errors="coerce")
    trailing = series.rolling(window=96, center=False, min_periods=96).std()
    stage_columns.append(series.mask(trailing < 1e-6))
S = np.column_stack([column.values for column in stage_columns]).astype(np.float32)

threshold_slice = frame.index <= pd.Timestamp(THRESHOLD_RECORD_END)
thresholds = {}
for percentile in EVENT_PERCENTILES:
    values = np.full(len(scored_gauges), np.nan)
    for position, column in enumerate(stage_columns):
        finite = column.values[threshold_slice]
        finite = finite[np.isfinite(finite)]
        if len(finite):
            values[position] = float(np.percentile(finite, percentile))
    thresholds[percentile] = values

pump_columns = sorted(str(name) for name in frame.columns if re.search(r"PD\d+_status$", str(name)))
pump_on = np.zeros(len(frame), dtype=bool)
for column in pump_columns:
    pump_on = pump_on | (pd.to_numeric(frame[column], errors="coerce").fillna(0.0).values > 0.5)
log("pump columns", len(pump_columns))


def build_masks(origins):
    """Per (origin, gauge) event flags, as a tidy frame keyed on t0 and gauge."""
    positions = np.asarray([row_of[time_point] for time_point in origins])
    steps = positions[:, None] + 1 + np.arange(LB_W)[None, :]
    observed = S[steps]
    peak = np.nanmax(np.where(np.isfinite(observed), observed, -np.inf), axis=1)
    pump_at_issue = pump_on[positions]
    records = {
        "t0": np.repeat(origins.to_numpy(), len(scored_gauges)),
        "gauge": np.tile(np.asarray(scored_gauges, dtype=object), len(origins)),
        "pump_at_issue": np.repeat(pump_at_issue, len(scored_gauges)),
    }
    for percentile in EVENT_PERCENTILES:
        records[f"stage_p{int(percentile)}"] = (peak >= thresholds[percentile][None, :]).reshape(-1)
    return pd.DataFrame(records)


def metrics(true_values, predicted_values):
    mask = np.isfinite(true_values) & np.isfinite(predicted_values)
    count = int(mask.sum())
    result = {"n": count, "RMSE_m": np.nan, "Pearson_r": np.nan, "NSE": np.nan, "KGE": np.nan}
    if count < MINIMUM_SAMPLES:
        return result
    true_selected = true_values[mask].astype(np.float64)
    predicted_selected = predicted_values[mask].astype(np.float64)
    difference = predicted_selected - true_selected
    result["RMSE_m"] = float(np.sqrt(np.mean(difference ** 2)))
    variance_true = float(np.var(true_selected))
    variance_predicted = float(np.var(predicted_selected))
    if variance_true <= 0.0:
        return result
    result["NSE"] = float(
        1.0 - np.sum(difference ** 2) / np.sum((true_selected - true_selected.mean()) ** 2)
    )
    if variance_predicted <= 0.0:
        return result
    correlation = float(np.corrcoef(true_selected, predicted_selected)[0, 1])
    result["Pearson_r"] = correlation
    mean_true = float(true_selected.mean())
    if abs(mean_true) <= 1e-9:
        return result
    alpha = float(np.sqrt(variance_predicted) / np.sqrt(variance_true))
    beta = float(predicted_selected.mean() / mean_true)
    result["KGE"] = float(
        1.0 - np.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    )
    return result


ORIGIN_SETS = ["all", "pump_at_issue", "stage_p80", "stage_p90"]

per_gauge_rows = []
crest_rows = []
mask_frame = None

for label, root, template in FORCINGS:
    for seed in SEEDS:
        directory = os.path.join(root, template.format(seed=seed))
        skill_path = os.path.join(directory, "reforecast_2026H1_h96_timeseries.csv")
        window_path = os.path.join(directory, "reforecast_2026H1_origin_window_metrics.csv")
        if not (os.path.exists(skill_path) and os.path.exists(window_path)):
            log("missing scores for", label, "seed", seed, directory)
            continue

        skill = pd.read_csv(
            skill_path,
            usecols=["timestamp_utc", "node", "observed_stage_m", "predicted_stage_m"],
            parse_dates=["timestamp_utc"],
        )
        skill = skill[skill["node"].isin(gauge_position)].copy()
        skill["t0"] = skill["timestamp_utc"] - pd.Timedelta(hours=24)
        skill = skill.rename(columns={"node": "gauge"})

        if mask_frame is None:
            origins = pd.DatetimeIndex(sorted(skill["t0"].unique()))
            origins = pd.DatetimeIndex([t for t in origins if t in row_of])
            mask_frame = build_masks(origins)
            log("origins", len(origins), origins.min(), "->", origins.max())

        merged = skill.merge(mask_frame, on=["t0", "gauge"], how="inner")
        for origin_set in ORIGIN_SETS:
            selection = (
                np.ones(len(merged), dtype=bool) if origin_set == "all" else merged[origin_set].to_numpy()
            )
            block = merged[selection]
            for gauge, gauge_block in block.groupby("gauge", sort=False):
                values = metrics(
                    gauge_block["observed_stage_m"].to_numpy(dtype=float),
                    gauge_block["predicted_stage_m"].to_numpy(dtype=float),
                )
                values.update(
                    {"forcing": label, "seed": seed, "origin_set": origin_set, "gauge": gauge}
                )
                per_gauge_rows.append(values)

        window = pd.read_csv(
            window_path,
            usecols=[
                "t0_utc",
                "gauge",
                "window_hours",
                "observed_rise_m",
                "peak_absolute_error_m",
                "peak_bias_m",
            ],
            parse_dates=["t0_utc"],
        )
        window = window[window["window_hours"] == 24].copy()
        window = window.rename(columns={"t0_utc": "t0"})
        window = window[window["gauge"].isin(gauge_position)]
        window = window.dropna(subset=["peak_absolute_error_m", "peak_bias_m"])
        window = window.merge(mask_frame, on=["t0", "gauge"], how="inner")
        rises = window[window["observed_rise_m"] >= RISE_THRESHOLD_M]
        for origin_set in ORIGIN_SETS:
            selection = (
                np.ones(len(rises), dtype=bool) if origin_set == "all" else rises[origin_set].to_numpy()
            )
            block = rises[selection]
            strong = block[block["observed_rise_m"] >= STRONG_RISE_THRESHOLD_M]
            record = {
                "forcing": label,
                "seed": seed,
                "origin_set": origin_set,
                "n_rises": int(len(block)),
                "mean_peak_abs_error_m": float(block["peak_absolute_error_m"].mean())
                if len(block)
                else np.nan,
                "missed_crest_pct": float(100.0 * (block["peak_bias_m"] < UNDERCALL_M).mean())
                if len(block)
                else np.nan,
                "n_rises_strong": int(len(strong)),
                "mean_peak_abs_error_m_strong": float(strong["peak_absolute_error_m"].mean())
                if len(strong)
                else np.nan,
                "missed_crest_pct_strong": float(
                    100.0 * (strong["peak_bias_m"] < UNDERCALL_M).mean()
                )
                if len(strong)
                else np.nan,
            }
            # The three rise bands of the published crest figure, so the event version can be drawn
            # with the same bars() layout instead of a new one.
            for band_label, low, high in RISE_BANDS:
                band = block[
                    (block["observed_rise_m"] >= low) & (block["observed_rise_m"] < high)
                ]
                record["n_rises_band_" + band_label] = int(len(band))
                record["mean_peak_abs_error_m_band_" + band_label] = (
                    float(band["peak_absolute_error_m"].mean()) if len(band) else np.nan
                )
                record["missed_crest_pct_band_" + band_label] = (
                    float(100.0 * (band["peak_bias_m"] < UNDERCALL_M).mean())
                    if len(band)
                    else np.nan
                )
            crest_rows.append(record)
        log(label, "seed", seed, "done")

per_gauge = pd.DataFrame(per_gauge_rows)
per_gauge_path = os.path.join(OUTPUT_DIRECTORY, f"event_fig13_per_gauge_h24{TAG}.csv")
per_gauge.to_csv(per_gauge_path, index=False)
log("wrote", per_gauge_path, per_gauge.shape)

metric_names = ["RMSE_m", "Pearson_r", "NSE", "KGE"]
seed_mean = (
    per_gauge.groupby(["forcing", "origin_set", "gauge"], as_index=False)[metric_names].mean()
)
table6 = (
    seed_mean.groupby(["forcing", "origin_set"], as_index=False)[metric_names].median()
)
sample = per_gauge.groupby(["forcing", "origin_set"], as_index=False)["n"].median()
table6 = table6.merge(sample, on=["forcing", "origin_set"], how="left")
table6_path = os.path.join(OUTPUT_DIRECTORY, f"event_table6_skill_by_forcing{TAG}.csv")
table6.to_csv(table6_path, index=False)
log("wrote", table6_path)

crest = pd.DataFrame(crest_rows)
# Average every value column over the seeds. A hand-written column list silently dropped the rise
# bands when they were added, so the columns are taken from the frame itself instead.
crest_value_columns = [
    column for column in crest.columns if column not in ("forcing", "seed", "origin_set")
]
crest_mean = crest.groupby(["forcing", "origin_set"], as_index=False)[crest_value_columns].mean()
crest_path = os.path.join(OUTPUT_DIRECTORY, f"event_fig14_crest_by_forcing{TAG}.csv")
crest_mean.to_csv(crest_path, index=False)
log("wrote", crest_path)

report_path = os.path.join(OUTPUT_DIRECTORY, f"EVENT_RAINFALL_REPORT{TAG}.txt")
with open(report_path, "w", encoding="utf-8") as handle:
    handle.write("Section 4.1 on event origins: rainfall-forcing sensitivity\n")
    handle.write("1 January to 30 June 2026, 51 gauges, three seeds, fixed h+24 lead\n")
    handle.write(
        "stage thresholds from the record through "
        f"{THRESHOLD_RECORD_END}; rises are at least {RISE_THRESHOLD_M:.4f} m\n\n"
    )
    for origin_set in ORIGIN_SETS:
        handle.write(f"=== Table 6, origin set: {origin_set} ===\n")
        block = table6[table6["origin_set"] == origin_set].sort_values("RMSE_m")
        handle.write(block.to_string(index=False))
        handle.write("\n\n")
    for origin_set in ORIGIN_SETS:
        handle.write(f"=== Figure 14 crest errors, origin set: {origin_set} ===\n")
        block = crest_mean[crest_mean["origin_set"] == origin_set].sort_values(
            "mean_peak_abs_error_m"
        )
        handle.write(block.to_string(index=False))
        handle.write("\n\n")
log("wrote", report_path)
print("EVENT_RAINFALL_DONE", flush=True)
