"""Event hydrographs: observed stage against the ST-GNN, the LSTM and the GRU, at the h+24 lead.

Two snapshots, one for each event definition agreed for this experiment:

  pump      the longest continuous period with any pump in the parish running
  stage_p90 the hour with the most gauges above their own 90th percentile

Revision 14 style (Farid, 2026-09-25): fonts and layout follow Figure 13 (plot_fig13_h24.py through the contact-sheet
recipe): serif Nimbus Roman or Times New Roman, bold 19 pt title, legend under the title at 13 pt, bold 12.2 pt panel
titles, 12.2 pt tick labels, 14 pt "Stage (m)" and bold 14 pt "Verifying time (UTC)", 300 dpi, the same panel spacing,
and date labels rotated 28 degrees. Gauge selection, event windows, data and line colors are unchanged.

Each figure is a 4-by-4 sheet of gauges drawn in the manuscript's established contact-sheet style
(project/manuscript/20260826_Final_V2/
figure_revision_scripts/fig08_11_contact_sheets_larger_fonts.py): same observed color, same grid, the
same three-tick y axis and the same bold panel titles. The recipe draws one forecast line, so the two
extra model lines are added in colors that do not collide with it, and the event window is shaded.

The plotted line is the fixed h+24 lead, stamped at its verifying time, which is how the published
contact sheets are drawn. Predictions are averaged over the three training seeds, so a single seed's
noise does not decide what the reader sees.

No model is run again. This reads the January to August archives already in ../runs.

Usage:
  conda run -n operational python -u build_event_hydrographs.py
"""
import json
import os
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# Figure 13's type and frame, copied from the contact-sheet recipe
# (20260826_Final_V2/figure_revision_scripts/fig08_11_contact_sheets_larger_fonts.py, lines 236-252).
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Nimbus Roman",
            "Times New Roman",
            "DejaVu Serif",
        ],
        "axes.edgecolor": "#3D3D3D",
        "axes.linewidth": 0.65,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "path.simplify": True,
        "path.simplify_threshold": 0.15,
    }
)

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
RUN_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "paper_view", "runs")
FIGURE_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "figures")
OUTPUT_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "outputs")

MATRIX = (
    "project/Experiments/"
    "NIGHTLY_VS_FROZEN_20260911/frozen_inputs/"
    "global_features_all_stations_feature_engineered_20260911.pkl"
)
INPARISH_MANIFEST = (
    "project/Experiments/INPARISH51_20260813/"
    "outputs/inparish51_manifest.json"
)

SEEDS = [101, 202, 303]
MODELS = [
    ("ST-GNN", "stgnn_seed{seed}_observed_jan_aug.npz", "#D95F70"),
    ("LSTM", "lstm_seed{seed}_observed_jan_aug.npz", "#2C6E9B"),
    ("GRU", "gru_seed{seed}_observed_jan_aug.npz", "#3E8E4F"),
]
OBSERVED_COLOR = "#171717"
GRID_COLOR = "#9A9A9A"
EVENT_SHADE = "#F2C14E"

FT2M = 0.3048
LB_W = 96
QUANTILE = 0.8
EXCLUDED_GAUGES = {"MBSA4570"}
THRESHOLD_RECORD_END = "2025-12-31 23:45"
EVENT_PERCENTILE = 90.0
PANEL_ROWS = 4
PANEL_COLUMNS = 4
PAD_HOURS = 18

BASIN_NAMES = {
    "BC": "Bayou Conway",
    "BM": "Bayou Manchac",
    "MB": "Marvin Braud",
    "HB": "Henderson Bayou",
}

os.makedirs(FIGURE_DIRECTORY, exist_ok=True)
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def log(*parts):
    print("[hydrograph]", *parts, flush=True)


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
log("scored gauges", len(scored_gauges))

stage_series = {}
for gauge in scored_gauges:
    series = pd.to_numeric(frame[gauge + "_stage_ft"], errors="coerce")
    trailing = series.rolling(window=96, center=False, min_periods=96).std()
    stage_series[gauge] = series.mask(trailing < 1e-6)
S = np.column_stack([stage_series[gauge].values for gauge in scored_gauges]).astype(np.float32)

threshold_slice = frame.index <= pd.Timestamp(THRESHOLD_RECORD_END)
thresholds = np.full(len(scored_gauges), np.nan)
for position, gauge in enumerate(scored_gauges):
    values = stage_series[gauge].values[threshold_slice]
    values = values[np.isfinite(values)]
    if len(values):
        thresholds[position] = float(np.percentile(values, EVENT_PERCENTILE))

pump_columns = sorted(str(name) for name in frame.columns if re.search(r"PD\d+_status$", str(name)))
pump_on = np.zeros(len(frame), dtype=bool)
for column in pump_columns:
    pump_on = pump_on | (pd.to_numeric(frame[column], errors="coerce").fillna(0.0).values > 0.5)

# ---- forecasts at the fixed h+24 lead, averaged over seeds ----
predictions = {}
origins = None
for label, pattern, _ in MODELS:
    stack = []
    for seed in SEEDS:
        path = os.path.join(RUN_DIRECTORY, pattern.format(seed=seed))
        if not os.path.exists(path):
            raise SystemExit("[FATAL] missing archive: " + path)
        archive = np.load(path, allow_pickle=True)
        archive_origins = pd.to_datetime(archive["origins_utc"])
        if archive_origins.tz is not None:
            archive_origins = archive_origins.tz_convert("UTC").tz_localize(None)
        nodes = [str(value) for value in archive["nodes"]]
        node_position = {node: position for position, node in enumerate(nodes)}
        columns = [node_position[gauge] for gauge in scored_gauges]
        quantile_index = [float(q) for q in archive["quantiles_saved"]].index(QUANTILE)
        stack.append(
            archive["pred_ft"][:, LB_W - 1, :, quantile_index][:, columns].astype(np.float32)
        )
        if origins is None:
            origins = archive_origins
        archive.close()
    predictions[label] = np.mean(np.stack(stack), axis=0)
    log(label, "loaded", predictions[label].shape)

verifying_times = origins + pd.Timedelta(hours=24)
origin_positions = np.asarray([row_of[time_point] for time_point in origins])
observed_at_verifying = S[origin_positions + LB_W]


def longest_true_run(flags):
    best_start = 0
    best_length = 0
    start = None
    for position, value in enumerate(flags):
        if value and start is None:
            start = position
        elif not value and start is not None:
            if position - start > best_length:
                best_length = position - start
                best_start = start
            start = None
    if start is not None and len(flags) - start > best_length:
        best_length = len(flags) - start
        best_start = start
    return best_start, best_length


# ---- case A: the longest continuous period with a pump running ----
pump_at_origin = pump_on[origin_positions]
pump_start, pump_length = longest_true_run(pump_at_origin)
pump_window = (pump_start, pump_start + pump_length)
log("pump period", verifying_times[pump_window[0]], "->", verifying_times[pump_window[1] - 1],
    "hours", pump_length)

# ---- case B: the hour with the most gauges above their 90th percentile ----
above = observed_at_verifying >= thresholds[None, :]
above_count = np.nansum(above, axis=1)
peak_index = int(np.argmax(above_count))
log("p90 peak", verifying_times[peak_index], "gauges above", int(above_count[peak_index]))

CASES = [
    {
        "key": "pump",
        "title": "Pump event: the longest continuous period with pumps running",
        "center": (pump_window[0] + pump_window[1]) // 2,
        "half_width": max(pump_length // 2 + PAD_HOURS, 36),
        "shade": (pump_window[0], pump_window[1]),
        "rank": pump_at_origin.astype(float),
    },
    {
        "key": "stage_p90",
        "title": "High-stage event: the most gauges above their 90th percentile",
        "center": peak_index,
        "half_width": 48,
        "shade": None,
        "rank": above_count.astype(float),
    },
]

summary_rows = []
for case in CASES:
    low = max(case["center"] - case["half_width"], 0)
    high = min(case["center"] + case["half_width"], len(origins) - 1)
    window = slice(low, high + 1)
    window_times = verifying_times[window]

    observed_window = observed_at_verifying[window]
    finite_counts = np.isfinite(observed_window).sum(axis=0)
    rise = np.full(len(scored_gauges), np.nan)
    chatter = np.full(len(scored_gauges), np.nan)
    for position in range(len(scored_gauges)):
        column = observed_window[:, position]
        column = column[np.isfinite(column)]
        if len(column) < 8:
            continue
        rise[position] = float(column.max() - column.min())
        # A median is the wrong summary here. A gauge that sits flat for most of the window and then
        # cycles violently still has a near-zero median step, so it passes a median test. The 90th
        # percentile of the step size catches that intermittent cycling, which is what a pump sump
        # does to its own stage record.
        chatter[position] = float(np.percentile(np.abs(np.diff(column)), 90))
    # A gauge that reverses direction on nearly every step is reporting sensor chatter, not a
    # hydrograph. Ranking by range alone rewards exactly those gauges, so they are excluded here.
    steady = np.isfinite(rise) & (rise > 0.05) & (chatter <= 0.25 * rise)
    if case["key"] == "pump":
        base_score = rise
    else:
        base_score = np.nansum(above[window], axis=0).astype(float)
    selection_score = np.where(steady, base_score, -np.inf)
    order = [
        position
        for position in np.argsort(-selection_score)
        if np.isfinite(selection_score[position]) and finite_counts[position] > 4
    ]
    chosen = order[: PANEL_ROWS * PANEL_COLUMNS]
    if len(chosen) < PANEL_ROWS * PANEL_COLUMNS:
        raise SystemExit("[FATAL] not enough gauges with data in the window for " + case["key"])

    figure, axes = plt.subplots(
        PANEL_ROWS,
        PANEL_COLUMNS,
        figsize=(11.7, 8.3),
        sharex=True,
    )
    for axis, gauge_position in zip(axes.ravel(), chosen):
        gauge = scored_gauges[gauge_position]
        observed_line = observed_at_verifying[window, gauge_position] * FT2M
        if case["shade"] is not None:
            axis.axvspan(
                verifying_times[case["shade"][0]],
                verifying_times[min(case["shade"][1], len(verifying_times) - 1)],
                color=EVENT_SHADE,
                alpha=0.22,
                zorder=0,
            )
        else:
            event_flags = above[window, gauge_position]
            if event_flags.any():
                axis.fill_between(
                    window_times,
                    0,
                    1,
                    where=event_flags,
                    transform=axis.get_xaxis_transform(),
                    color=EVENT_SHADE,
                    alpha=0.22,
                    zorder=0,
                )
        # Line widths of Figure 13's panels: observed 1.10, forecast 1.00.
        axis.plot(
            window_times,
            observed_line,
            color=OBSERVED_COLOR,
            linewidth=1.10,
            alpha=0.96,
            label="Observed stage",
            zorder=5,
        )
        for label, _, color in MODELS:
            axis.plot(
                window_times,
                predictions[label][window, gauge_position] * FT2M,
                color=color,
                linewidth=1.00,
                alpha=0.90,
                label=label + " h+24",
                zorder=4,
            )
        axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
        axis.xaxis.set_major_locator(mdates.DayLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        axis.grid(True, color=GRID_COLOR, linewidth=0.50, alpha=0.24, zorder=0)
        axis.tick_params(axis="both", labelsize=12.2, length=2.6, width=0.6, pad=1.6)
        basin = BASIN_NAMES.get(gauge[:2], "unrecognized basin")
        axis.set_title(gauge + " | " + basin, fontsize=12.2, fontweight="bold", pad=1.8)

        for label, _, _ in MODELS:
            model_line = predictions[label][window, gauge_position] * FT2M
            mask = np.isfinite(observed_line) & np.isfinite(model_line)
            if mask.sum() > 4:
                summary_rows.append(
                    {
                        "case": case["key"],
                        "gauge": gauge,
                        "model": label,
                        "n": int(mask.sum()),
                        "RMSE_m": float(
                            np.sqrt(np.mean((model_line[mask] - observed_line[mask]) ** 2))
                        ),
                    }
                )

    for axis in axes[-1]:
        plt.setp(axis.get_xticklabels(), rotation=28, horizontalalignment="right")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    # Title, legend, axis labels and spacing as Figure 13 draws them.
    figure.suptitle(case["title"], fontsize=19.0, fontweight="bold", y=0.997)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.974),
        ncol=4,
        frameon=False,
        fontsize=13.0,
    )
    figure.text(
        0.012,
        0.50,
        "Stage (m)",
        rotation=90,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=14.0,
    )
    figure.text(
        0.5,
        0.008,
        "Verifying time (UTC)",
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=14.0,
        fontweight="bold",
        color="#222222",
    )
    figure.subplots_adjust(
        left=0.060,
        right=0.996,
        top=0.872,
        bottom=0.110,
        hspace=0.48,
        wspace=0.18,
    )
    for extension in ("png",):
        path = os.path.join(FIGURE_DIRECTORY, f"event_hydrographs_{case['key']}.{extension}")
        figure.savefig(path, dpi=300)
        log("wrote", path)
    plt.close(figure)

summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUTPUT_DIRECTORY, "event_hydrograph_window_rmse.csv")
summary.to_csv(summary_path, index=False)
log("wrote", summary_path)
if len(summary):
    log("window median RMSE by case and model")
    log(
        summary.groupby(["case", "model"])["RMSE_m"].median().round(4).to_string()
    )
print("EVENT_HYDROGRAPHS_DONE", flush=True)
plt.show()
