"""
Freeze the primary gauge cohort for the chronological HESS experiments.

Membership is determined only from stage values through 31 December 2025.
The script reproduces the deployed DQ stage mask, verifies exact duplicate
series, checks persistent abrupt-step behavior, and writes an auditable cohort
table. It never modifies source data, graphs, models, or forecasts.
"""

import hashlib
import os

import numpy as np
import pandas as pd


OUTPUT_DIRECTORY = (
    "project/Experiments/"
    "EVENT_SPLIT_INVENTORY_HESS_20260910"
)
OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIRECTORY, "GAUGE_COHORT_PRE2026.csv")
OUTPUT_REPORT_PATH = os.path.join(OUTPUT_DIRECTORY, "GAUGE_COHORT_PRE2026.md")

MATRIX_PATH = (
    "project/hpc/Training/"
    "Prepared_Global_Matrix/global_features_all_stations_feature_engineered.pkl"
)
GRAPH_PATH = (
    "project/hpc/Training/GNN/"
    "Experiments/REFORECAST_GRAPH_RANKING_2026H1/graphs/graph_obs_pre2026.npz"
)
TRAINER_PATH = (
    "project/hpc/Training/GNN/"
    "stgnn_train_reforecast_2026.py"
)
STATION_METADATA_PATH = (
    "project/Data_Share_Sidney_20260824/"
    "outputs/ascension_stage_stations_full.csv"
)

PRE2026_END = pd.Timestamp("2025-12-31 23:45:00", tz="UTC")
DQ_FLATLINE_WINDOW_STEPS = 96
DQ_FLATLINE_STANDARD_DEVIATION_FT = 1.0e-6
TARGET_STEPS = 96
CADENCE_MINUTES = 15
MINIMUM_DQ_WEIGHTED_TARGET_FRACTION = 0.90
ABRUPT_STEP_THRESHOLD_FT = 0.50
MAXIMUM_ABRUPT_STEP_FRACTION = 0.01
EXACT_DUPLICATE_MINIMUM_OVERLAP = 1000

SPLITS = {
    "training": (
        pd.Timestamp("2023-01-04 00:00:00", tz="UTC"),
        pd.Timestamp("2024-12-31 23:00:00", tz="UTC"),
    ),
    "validation": (
        pd.Timestamp("2025-01-05 00:00:00", tz="UTC"),
        pd.Timestamp("2025-10-31 23:00:00", tz="UTC"),
    ),
    "calibration": (
        pd.Timestamp("2025-11-05 00:00:00", tz="UTC"),
        pd.Timestamp("2025-12-27 23:00:00", tz="UTC"),
    ),
}


def calculate_sha256(path):
    print(f"[HASH] Calculating SHA256: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        while True:
            block = input_file.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    result = digest.hexdigest()
    print(f"[HASH] SHA256: {result}")
    return result


def format_timestamp(value):
    if pd.isna(value):
        return ""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")


def longest_true_run(boolean_values):
    maximum = 0
    current = 0
    for value in boolean_values:
        if bool(value):
            current += 1
            if current > maximum:
                maximum = current
        else:
            current = 0
    return maximum


def markdown_table(frame):
    if frame.empty:
        return "No rows."
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].astype(str).str.replace("|", "\\|", regex=False)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = []
    for _, row in display.iterrows():
        rows.append("| " + " | ".join(row.tolist()) + " |")
    return "\n".join([header, separator] + rows)


print("[START] Freezing the pre-2026 gauge cohort.")
print(f"[START] Working directory: {os.getcwd()}")
print(f"[START] Output directory: {OUTPUT_DIRECTORY}")
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

required_source_paths = [
    MATRIX_PATH,
    GRAPH_PATH,
    TRAINER_PATH,
    STATION_METADATA_PATH,
]
for source_path in required_source_paths:
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Required source file does not exist: {source_path}")
    print(f"[SOURCE] Found: {source_path}")

matrix_sha256 = calculate_sha256(MATRIX_PATH)
graph_sha256 = calculate_sha256(GRAPH_PATH)
trainer_sha256 = calculate_sha256(TRAINER_PATH)
station_metadata_sha256 = calculate_sha256(STATION_METADATA_PATH)

print("[VERIFY] Confirming the trainer contains the expected DQ rule.")
with open(TRAINER_PATH, "r", encoding="utf-8") as trainer_file:
    trainer_text = trainer_file.read()
required_trainer_fragments = [
    "s.rolling(96, center=True, min_periods=96).std() < 1e-6",
    "w = MT[f0:f0+LB_W] * MT[t0][None, :]",
]
for fragment in required_trainer_fragments:
    if fragment not in trainer_text:
        raise AssertionError(
            "Trainer DQ implementation has changed. Missing required fragment: "
            f"{fragment}"
        )
print("[VERIFY] Trainer DQ rule matches the implemented cohort audit rule.")

print("[LOAD] Loading canonical graph node order.")
graph_archive = np.load(GRAPH_PATH, allow_pickle=True)
if "nodes" not in graph_archive.files:
    raise KeyError(f"Graph archive lacks nodes. Keys: {graph_archive.files}")
nodes = [str(value) for value in graph_archive["nodes"].tolist()]
if len(nodes) != 68:
    raise ValueError(f"Expected 68 canonical nodes, found {len(nodes)}")
if len(set(nodes)) != len(nodes):
    raise ValueError("Canonical graph node names are not unique.")
node_order = {node: index for index, node in enumerate(nodes)}
print(f"[LOAD] Canonical node count: {len(nodes)}")

print("[LOAD] Loading station metadata for basin labels and duplicate cross-checking.")
station_metadata = pd.read_csv(STATION_METADATA_PATH)
required_metadata_columns = {
    "stage_name_legacy",
    "drainage_area",
    "duplicate_of",
}
missing_metadata_columns = required_metadata_columns.difference(station_metadata.columns)
if missing_metadata_columns:
    raise KeyError(f"Station metadata lacks columns: {sorted(missing_metadata_columns)}")
station_metadata = station_metadata.drop_duplicates(
    subset=["stage_name_legacy"],
    keep="last",
).set_index("stage_name_legacy")
missing_metadata_nodes = [node for node in nodes if node not in station_metadata.index]
if missing_metadata_nodes:
    raise ValueError(f"Canonical nodes absent from station metadata: {missing_metadata_nodes}")

print("[LOAD] Loading the engineered matrix and slicing before any calculation.")
matrix = pd.read_pickle(MATRIX_PATH)
if not isinstance(matrix.index, pd.DatetimeIndex):
    raise TypeError(f"Expected DatetimeIndex, found {type(matrix.index)}")
if matrix.index.tz is None:
    matrix.index = matrix.index.tz_localize("UTC")
else:
    matrix.index = matrix.index.tz_convert("UTC")
matrix = matrix.sort_index()
duplicate_timestamp_count = int(matrix.index.duplicated().sum())
if duplicate_timestamp_count > 0:
    raise ValueError(f"Matrix contains {duplicate_timestamp_count} duplicate timestamps.")
matrix = matrix.loc[matrix.index <= PRE2026_END].copy()
if matrix.empty:
    raise ValueError("Pre-2026 matrix slice is empty.")

expected_index = pd.date_range(
    matrix.index.min(),
    PRE2026_END,
    freq=f"{CADENCE_MINUTES}min",
)
missing_matrix_timestamps = expected_index.difference(matrix.index)
if len(missing_matrix_timestamps) > 0:
    raise ValueError(
        f"Pre-2026 matrix is missing {len(missing_matrix_timestamps)} 15-minute timestamps."
    )
print(
    f"[LOAD] Pre-2026 matrix: {len(matrix):,} rows from {matrix.index.min()} "
    f"through {matrix.index.max()}."
)

stage_columns = [f"{node}_stage_ft" for node in nodes]
missing_stage_columns = [column for column in stage_columns if column not in matrix.columns]
if missing_stage_columns:
    raise KeyError(f"Matrix lacks canonical stage columns: {missing_stage_columns}")
stage = matrix[stage_columns].apply(pd.to_numeric, errors="coerce")

print("[DQ] Reproducing the deployed centered 96-step flatline mask.")
dq_stage = pd.DataFrame(index=stage.index, columns=stage.columns, dtype="float32")
flatline_mask = pd.DataFrame(index=stage.index, columns=stage.columns, dtype=bool)
for stage_column in stage_columns:
    stage_series = stage[stage_column]
    current_flatline_mask = stage_series.rolling(
        DQ_FLATLINE_WINDOW_STEPS,
        center=True,
        min_periods=DQ_FLATLINE_WINDOW_STEPS,
    ).std().lt(DQ_FLATLINE_STANDARD_DEVIATION_FT)
    flatline_mask[stage_column] = current_flatline_mask
    dq_stage[stage_column] = stage_series.mask(current_flatline_mask).astype("float32")

dq_valid = dq_stage.notna()
future_valid_target_count = dq_valid.shift(-1).iloc[::-1].rolling(
    window=TARGET_STEPS,
    min_periods=1,
).sum().iloc[::-1]

toy_mask = pd.DataFrame({"stage": [True] * 200})
toy_future_count = toy_mask.shift(-1).iloc[::-1].rolling(
    window=TARGET_STEPS,
    min_periods=1,
).sum().iloc[::-1]
if int(toy_future_count.iloc[0, 0]) != TARGET_STEPS:
    raise AssertionError("Forward target-validity calculation failed its deterministic toy test.")
print("[DQ] Forward target-validity calculation passed its deterministic toy test.")

print("[DUPLICATE] Detecting bit-identical candidate series using only pre-2026 values.")
preliminary_metrics = {}
for node in nodes:
    stage_column = f"{node}_stage_ft"
    node_metrics = {}
    for split_name, split_bounds in SPLITS.items():
        origins = pd.date_range(split_bounds[0], split_bounds[1], freq="1h")
        t0_valid = dq_valid.reindex(origins)[stage_column].fillna(False)
        target_count = future_valid_target_count.reindex(origins)[stage_column].fillna(0.0)
        weighted_target_fraction = float(
            (target_count * t0_valid.astype(int)).sum()
            / (len(origins) * TARGET_STEPS)
        )
        full_window_fraction = float(
            (t0_valid & target_count.eq(TARGET_STEPS)).mean()
        )
        node_metrics[f"{split_name}_origin_count"] = int(len(origins))
        node_metrics[f"{split_name}_t0_dq_valid_fraction"] = float(t0_valid.mean())
        node_metrics[f"{split_name}_dq_weighted_target_fraction"] = weighted_target_fraction
        node_metrics[f"{split_name}_fully_dq_valid_24h_window_fraction"] = full_window_fraction
    preliminary_metrics[node] = node_metrics

coverage_candidate_nodes = []
for node in nodes:
    node_metrics = preliminary_metrics[node]
    training_pass = (
        node_metrics["training_dq_weighted_target_fraction"]
        >= MINIMUM_DQ_WEIGHTED_TARGET_FRACTION
    )
    validation_pass = (
        node_metrics["validation_dq_weighted_target_fraction"]
        >= MINIMUM_DQ_WEIGHTED_TARGET_FRACTION
    )
    if training_pass and validation_pass:
        coverage_candidate_nodes.append(node)
print(f"[DUPLICATE] Coverage candidates before duplicate checks: {len(coverage_candidate_nodes)}")

duplicate_representative = {node: "" for node in nodes}
duplicate_overlap_count = {node: 0 for node in nodes}
detected_duplicate_pairs = []
for first_index, first_node in enumerate(coverage_candidate_nodes):
    first_values = stage[f"{first_node}_stage_ft"].to_numpy(dtype=float)
    for second_node in coverage_candidate_nodes[first_index + 1:]:
        second_values = stage[f"{second_node}_stage_ft"].to_numpy(dtype=float)
        both_finite = np.isfinite(first_values) & np.isfinite(second_values)
        overlap_count = int(both_finite.sum())
        if overlap_count < EXACT_DUPLICATE_MINIMUM_OVERLAP:
            continue
        missing_patterns_equal = np.array_equal(
            np.isfinite(first_values),
            np.isfinite(second_values),
        )
        finite_values_equal = np.array_equal(
            first_values[both_finite],
            second_values[both_finite],
        )
        if missing_patterns_equal and finite_values_equal:
            representative = first_node
            if duplicate_representative[first_node]:
                representative = duplicate_representative[first_node]
            duplicate_representative[second_node] = representative
            duplicate_overlap_count[second_node] = overlap_count
            detected_duplicate_pairs.append(
                {
                    "representative": representative,
                    "duplicate": second_node,
                    "overlap_points": overlap_count,
                }
            )
            print(
                f"[DUPLICATE] {second_node} is bit-identical to {representative} "
                f"over {overlap_count:,} pre-2026 finite points."
            )

expected_duplicate_map = {}
for node in nodes:
    metadata_duplicate = station_metadata.loc[node, "duplicate_of"]
    if pd.notna(metadata_duplicate) and str(metadata_duplicate).strip():
        expected_duplicate_map[node] = str(metadata_duplicate).strip()

for duplicate_node, expected_representative in expected_duplicate_map.items():
    detected_representative = duplicate_representative.get(duplicate_node, "")
    if detected_representative != expected_representative:
        raise AssertionError(
            f"Duplicate cross-check failed for {duplicate_node}: metadata says "
            f"{expected_representative}, pre-2026 values detected {detected_representative or 'none'}"
        )
print(
    f"[DUPLICATE] Cross-checked {len(expected_duplicate_map)} documented duplicate feeds."
)

print("[QUALITY] Calculating abrupt-step and flatline diagnostics.")
cohort_rows = []
for node in nodes:
    stage_column = f"{node}_stage_ft"
    raw_series = stage[stage_column]
    dq_series = dq_stage[stage_column]
    raw_finite = raw_series.notna()
    dq_finite = dq_series.notna()
    adjacent_finite = raw_finite & raw_series.shift(1).notna()
    absolute_step = raw_series.diff().abs()
    abrupt_step_mask = adjacent_finite & absolute_step.gt(ABRUPT_STEP_THRESHOLD_FT)
    adjacent_finite_count = int(adjacent_finite.sum())
    abrupt_step_count = int(abrupt_step_mask.sum())
    if adjacent_finite_count > 0:
        abrupt_step_fraction = abrupt_step_count / adjacent_finite_count
    else:
        abrupt_step_fraction = np.nan

    training_coverage = preliminary_metrics[node][
        "training_dq_weighted_target_fraction"
    ]
    validation_coverage = preliminary_metrics[node][
        "validation_dq_weighted_target_fraction"
    ]
    training_coverage_pass = bool(
        training_coverage >= MINIMUM_DQ_WEIGHTED_TARGET_FRACTION
    )
    validation_coverage_pass = bool(
        validation_coverage >= MINIMUM_DQ_WEIGHTED_TARGET_FRACTION
    )
    duplicate_of = duplicate_representative[node]
    duplicate_pass = not bool(duplicate_of)
    discontinuity_pass = bool(
        np.isfinite(abrupt_step_fraction)
        and abrupt_step_fraction <= MAXIMUM_ABRUPT_STEP_FRACTION
    )

    exclusion_reasons = []
    if not training_coverage_pass:
        exclusion_reasons.append(
            "training DQ-weighted target fraction below 0.90"
        )
    if not validation_coverage_pass:
        exclusion_reasons.append(
            "validation DQ-weighted target fraction below 0.90"
        )
    if not duplicate_pass:
        exclusion_reasons.append(
            f"bit-identical second feed of {duplicate_of}"
        )
    if not discontinuity_pass:
        exclusion_reasons.append(
            "more than 1% of finite adjacent pre-2026 steps change by over 0.50 ft"
        )

    include_primary = bool(
        training_coverage_pass
        and validation_coverage_pass
        and duplicate_pass
        and discontinuity_pass
    )
    if include_primary:
        cohort_status = "INCLUDE_PRIMARY"
        exclusion_reason = ""
        severity = "none"
    else:
        exclusion_reason = " | ".join(exclusion_reasons)
        if not training_coverage_pass or not validation_coverage_pass:
            cohort_status = "EXCLUDE_DQ_COVERAGE"
            severity = "high"
        elif not duplicate_pass:
            cohort_status = "EXCLUDE_DUPLICATE_SECOND_FEED"
            severity = "high"
        else:
            cohort_status = "EXCLUDE_UNRESOLVED_DISCONTINUITY"
            severity = "high"

    unchanged_step_mask = adjacent_finite & absolute_step.le(1.0e-8)
    metadata_duplicate = station_metadata.loc[node, "duplicate_of"]
    if pd.isna(metadata_duplicate):
        metadata_duplicate_text = ""
    else:
        metadata_duplicate_text = str(metadata_duplicate).strip()

    row = {
        "canonical_graph_order": int(node_order[node]),
        "selected_cohort_order": np.nan,
        "gauge": node,
        "stage_column": stage_column,
        "basin": str(station_metadata.loc[node, "drainage_area"]),
        "include_primary": include_primary,
        "cohort_status": cohort_status,
        "severity": severity,
        "exclusion_reason": exclusion_reason,
        "training_coverage_pass": training_coverage_pass,
        "validation_coverage_pass": validation_coverage_pass,
        "duplicate_pass": duplicate_pass,
        "discontinuity_pass": discontinuity_pass,
        "detected_duplicate_of": duplicate_of,
        "documented_duplicate_of": metadata_duplicate_text,
        "duplicate_finite_overlap_points": int(duplicate_overlap_count[node]),
        "first_raw_finite_utc": format_timestamp(raw_series.first_valid_index()),
        "last_raw_finite_pre2026_utc": format_timestamp(raw_series.last_valid_index()),
        "first_dq_valid_utc": format_timestamp(dq_series.first_valid_index()),
        "last_dq_valid_pre2026_utc": format_timestamp(dq_series.last_valid_index()),
        "pre2026_raw_finite_fraction": float(raw_finite.mean()),
        "pre2026_dq_valid_fraction": float(dq_finite.mean()),
        "pre2026_flatline_masked_fraction": float(flatline_mask[stage_column].mean()),
        "pre2026_flatline_masked_days": float(
            flatline_mask[stage_column].sum() * CADENCE_MINUTES / 60.0 / 24.0
        ),
        "longest_raw_unchanged_run_hours": float(
            longest_true_run(unchanged_step_mask.to_numpy()) * CADENCE_MINUTES / 60.0
        ),
        "finite_adjacent_step_count": adjacent_finite_count,
        "abrupt_step_gt_0_50ft_count": abrupt_step_count,
        "abrupt_step_gt_0_50ft_fraction": float(abrupt_step_fraction),
        "abs_step_gt_1ft_count": int((adjacent_finite & absolute_step.gt(1.0)).sum()),
        "abs_step_gt_3ft_count": int((adjacent_finite & absolute_step.gt(3.0)).sum()),
        "maximum_abs_15min_step_ft": float(absolute_step.max()),
        "pre2026_stage_min_ft": float(raw_series.min()),
        "pre2026_stage_p01_ft": float(raw_series.quantile(0.01)),
        "pre2026_stage_median_ft": float(raw_series.median()),
        "pre2026_stage_p99_ft": float(raw_series.quantile(0.99)),
        "pre2026_stage_max_ft": float(raw_series.max()),
        "selection_uses_2026_data": False,
        "selection_rule": (
            "training and validation DQ-weighted target fractions >=0.90; retain only the "
            "first canonical feed from any bit-identical pre-2026 duplicate group; exclude "
            "series where >1% of finite adjacent steps change by >0.50 ft"
        ),
        "dq_rule": (
            "mask stage where centered 96-step rolling standard deviation <1e-6; "
            "origin-lead weight equals valid(target lead) times valid(origin)"
        ),
    }
    for split_name in SPLITS:
        row.update(
            {
                f"{split_name}_origin_count": preliminary_metrics[node][
                    f"{split_name}_origin_count"
                ],
                f"{split_name}_t0_dq_valid_fraction": preliminary_metrics[node][
                    f"{split_name}_t0_dq_valid_fraction"
                ],
                f"{split_name}_dq_weighted_target_fraction": preliminary_metrics[node][
                    f"{split_name}_dq_weighted_target_fraction"
                ],
                f"{split_name}_fully_dq_valid_24h_window_fraction": preliminary_metrics[node][
                    f"{split_name}_fully_dq_valid_24h_window_fraction"
                ],
            }
        )
    cohort_rows.append(row)

cohort = pd.DataFrame(cohort_rows).sort_values("canonical_graph_order").reset_index(drop=True)
selected_indexes = cohort.index[cohort["include_primary"]].tolist()
for selected_order, row_index in enumerate(selected_indexes):
    cohort.loc[row_index, "selected_cohort_order"] = int(selected_order)
cohort["selected_cohort_order"] = cohort["selected_cohort_order"].astype("Int64")

selected_nodes = cohort.loc[cohort["include_primary"], "gauge"].tolist()
selected_node_payload = "\n".join(selected_nodes).encode("utf-8")
cohort_id = hashlib.sha256(selected_node_payload).hexdigest()
cohort.insert(0, "cohort_id_sha256", cohort_id)

expected_selected_count = 30
if len(selected_nodes) != expected_selected_count:
    raise AssertionError(
        f"Expected {expected_selected_count} selected gauges under the frozen rule, "
        f"found {len(selected_nodes)}. Source data or logic changed and requires review."
    )

detected_duplicate_table = pd.DataFrame(detected_duplicate_pairs)
if len(detected_duplicate_table) != 5:
    raise AssertionError(
        f"Expected five documented bit-identical pairs, found {len(detected_duplicate_table)}"
    )

status_counts = cohort["cohort_status"].value_counts().to_dict()
expected_status_counts = {
    "EXCLUDE_DQ_COVERAGE": 30,
    "INCLUDE_PRIMARY": 30,
    "EXCLUDE_DUPLICATE_SECOND_FEED": 5,
    "EXCLUDE_UNRESOLVED_DISCONTINUITY": 3,
}
if status_counts != expected_status_counts:
    raise AssertionError(
        f"Unexpected cohort status counts: {status_counts}; expected {expected_status_counts}"
    )

print("[WRITE] Writing GAUGE_COHORT_PRE2026.csv.")
cohort.to_csv(OUTPUT_CSV_PATH, index=False)

selected_display = cohort.loc[
    cohort["include_primary"],
    [
        "selected_cohort_order",
        "gauge",
        "basin",
        "training_dq_weighted_target_fraction",
        "validation_dq_weighted_target_fraction",
    ],
].copy()
selected_display["training_dq_weighted_target_fraction"] = selected_display[
    "training_dq_weighted_target_fraction"
].map(lambda value: f"{100.0 * value:.2f}%")
selected_display["validation_dq_weighted_target_fraction"] = selected_display[
    "validation_dq_weighted_target_fraction"
].map(lambda value: f"{100.0 * value:.2f}%")

excluded_display = cohort.loc[
    ~cohort["include_primary"],
    [
        "gauge",
        "cohort_status",
        "training_dq_weighted_target_fraction",
        "validation_dq_weighted_target_fraction",
        "detected_duplicate_of",
        "abrupt_step_gt_0_50ft_fraction",
        "exclusion_reason",
    ],
].copy()
excluded_display["training_dq_weighted_target_fraction"] = excluded_display[
    "training_dq_weighted_target_fraction"
].map(lambda value: f"{100.0 * value:.2f}%")
excluded_display["validation_dq_weighted_target_fraction"] = excluded_display[
    "validation_dq_weighted_target_fraction"
].map(lambda value: f"{100.0 * value:.2f}%")
excluded_display["abrupt_step_gt_0_50ft_fraction"] = excluded_display[
    "abrupt_step_gt_0_50ft_fraction"
].map(lambda value: f"{100.0 * value:.3f}%" if np.isfinite(value) else "")

duplicate_display = detected_duplicate_table.copy()
duplicate_display["decision"] = duplicate_display.apply(
    lambda row: f"keep {row['representative']}; exclude {row['duplicate']}",
    axis=1,
)

discontinuity_display = cohort.loc[
    cohort["cohort_status"].eq("EXCLUDE_UNRESOLVED_DISCONTINUITY"),
    [
        "gauge",
        "abrupt_step_gt_0_50ft_count",
        "abrupt_step_gt_0_50ft_fraction",
        "abs_step_gt_1ft_count",
        "abs_step_gt_3ft_count",
        "maximum_abs_15min_step_ft",
    ],
].copy()
discontinuity_display["abrupt_step_gt_0_50ft_fraction"] = discontinuity_display[
    "abrupt_step_gt_0_50ft_fraction"
].map(lambda value: f"{100.0 * value:.3f}%")
discontinuity_display["maximum_abs_15min_step_ft"] = discontinuity_display[
    "maximum_abs_15min_step_ft"
].map(lambda value: f"{value:.3f}")

basin_counts = selected_display["basin"].value_counts().rename_axis("basin").reset_index(
    name="selected_gauges"
)

report_lines = [
    "# Frozen pre-2026 gauge cohort",
    "",
    "## Decision",
    "",
    f"Freeze a {len(selected_nodes)}-gauge primary scoring cohort for the clean chronological experiments. ",
    "Keep the 68-node graph. Use the 30 gauges, in the same order, as the evaluation mask ",
    "for the observation-derived ST-GNN, identity-graph ST-GNN, and nodewise LSTM. ",
    "Membership was determined without reading 2026 values.",
    "",
    f"Cohort ID, SHA256 of the ordered gauge names: `{cohort_id}`",
    "",
    "## Frozen selection rule",
    "",
    "A gauge is included only when all three conditions pass:",
    "",
    "1. Its DQ-weighted target fraction is at least 90% in the proposed training origins.",
    "2. Its DQ-weighted target fraction is at least 90% in the proposed validation origins.",
    "3. It is not a bit-identical second feed and no more than 1% of its finite adjacent ",
    "   pre-2026 values change by more than 0.50 ft in 15 minutes.",
    "",
    "The DQ mask exactly reproduces the current trainer: mask a stage value when its centered ",
    "96-step rolling standard deviation is below 1e-6. An origin-lead target receives weight only ",
    "when both the origin and that target value are valid. Calibration coverage is reported but ",
    "does not select the model cohort. No test-period availability or stage behavior is used.",
    "",
    "## Outcome",
    "",
    markdown_table(
        pd.DataFrame(
            [
                {"status": status, "gauges": count}
                for status, count in sorted(status_counts.items())
            ]
        )
    ),
    "",
    markdown_table(basin_counts),
    "",
    "## Selected gauges",
    "",
    markdown_table(selected_display),
    "",
    "## Exact duplicate feeds removed",
    "",
    "These pairs are bit-identical over their complete pre-2026 finite overlap. The first gauge in ",
    "canonical graph order is retained, except when that representative independently fails another ",
    "quality rule.",
    "",
    markdown_table(duplicate_display),
    "",
    "## Unresolved discontinuity holds",
    "",
    "The production flatline mask does not detect rapidly switching or persistently noisy series. ",
    "The following coverage-qualified representatives exceed the predeclared 1% abrupt-step limit ",
    "and are held out of the primary cohort.",
    "",
    markdown_table(discontinuity_display),
    "",
    "## All exclusions",
    "",
    markdown_table(excluded_display),
    "",
    "## What this permits next",
    "",
    "Keep the 68-node observation-derived graph. Do not construct a new 30-node graph. ",
    "Use the 30 selected gauges as a scoring mask on that 68-node tensor. Construct the ",
    "identity adjacency on the same 68 nodes, in the identical canonical order. ",
    "Train all comparison models with the frozen origin blocks and DQ masks. Test-period missing ",
    "targets may be masked during scoring, but they must not change cohort membership.",
    "",
    "## Provenance and limitations",
    "",
    f"- Engineered matrix: `{MATRIX_PATH}`",
    f"- Matrix SHA256: `{matrix_sha256}`",
    f"- Canonical graph node source: `{GRAPH_PATH}`",
    f"- Graph SHA256: `{graph_sha256}`",
    f"- DQ trainer source: `{TRAINER_PATH}`",
    f"- DQ trainer SHA256: `{trainer_sha256}`",
    f"- Station metadata cross-check: `{STATION_METADATA_PATH}`",
    f"- Station metadata SHA256: `{station_metadata_sha256}`",
    f"- Data cutoff applied before all membership calculations: {format_timestamp(PRE2026_END)}",
    f"- Source duplicate timestamp count: {duplicate_timestamp_count}",
    "- The 0.50-ft step screen identifies unresolved discontinuity or noise; it does not assign a ",
    "  physical cause such as datum change, sensor error, or real rapid hydraulic response.",
    "- Historical APG values may have been backfilled. This cohort is deterministic for the recorded ",
    "  source hashes, not a reconstruction of every historical as-issued data vintage.",
    "",
    "## Read-only guarantee",
    "",
    "This script read the source files above and wrote only GAUGE_COHORT_PRE2026.csv and ",
    "GAUGE_COHORT_PRE2026.md in the experiment inventory directory. It did not alter any source ",
    "matrix, graph, metadata, model, forecast, or production file.",
]

print("[WRITE] Writing GAUGE_COHORT_PRE2026.md.")
with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as output_file:
    output_file.write("\n".join(report_lines))
    output_file.write("\n")

print("[VERIFY] Re-reading and validating generated cohort artifacts.")
verified_cohort = pd.read_csv(OUTPUT_CSV_PATH)
with open(OUTPUT_REPORT_PATH, "r", encoding="utf-8") as report_file:
    verified_report = report_file.read()

if len(verified_cohort) != len(nodes):
    raise AssertionError(
        f"Cohort CSV has {len(verified_cohort)} rows, expected {len(nodes)}"
    )
if verified_cohort["gauge"].duplicated().any():
    raise AssertionError("Cohort CSV contains duplicate gauge rows.")
if int(verified_cohort["include_primary"].sum()) != expected_selected_count:
    raise AssertionError("Cohort CSV selected-count verification failed.")
verified_orders = verified_cohort.loc[
    verified_cohort["include_primary"],
    "selected_cohort_order",
].astype(int).tolist()
if verified_orders != list(range(expected_selected_count)):
    raise AssertionError(f"Selected cohort order is not contiguous: {verified_orders}")
if cohort_id not in verified_report:
    raise AssertionError("Cohort report does not contain the cohort ID.")
if "2026" in " ".join(
    [
        column
        for column in verified_cohort.columns
        if column.endswith("_dq_weighted_target_fraction")
    ]
):
    raise AssertionError("A 2026 metric unexpectedly appears in the membership metric columns.")
if "\u2014" in verified_report:
    raise AssertionError("Cohort report contains a prohibited em dash character.")

print(f"[VERIFY] Cohort rows: {len(verified_cohort)}")
print(f"[VERIFY] Selected primary gauges: {int(verified_cohort['include_primary'].sum())}")
print(f"[VERIFY] Excluded gauges: {int((~verified_cohort['include_primary']).sum())}")
print(f"[VERIFY] Cohort ID: {cohort_id}")
print(f"[DONE] {OUTPUT_CSV_PATH}")
print(f"[DONE] {OUTPUT_REPORT_PATH}")
print("[DONE] Pre-2026 gauge cohort frozen successfully.")
