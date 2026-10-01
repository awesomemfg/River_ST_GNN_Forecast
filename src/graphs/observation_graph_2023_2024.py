"""
Build leakage-safe 68-node graph and scoring-mask assets for the HESS experiments.

The observation-derived adjacency is discovered only from stage observations through
31 December 2024, which is the end of the proposed training period. The graph retains
the complete operational 68-node order. The separately stored 30-gauge mask controls
primary evaluation only and does not remove nodes or targets from model training.

All source files are read-only. Every generated artifact is written beneath the
EVENT_SPLIT_INVENTORY_HESS_20260910 experiment directory.
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags


EXPERIMENT_DIRECTORY = (
    "project/Experiments/"
    "EVENT_SPLIT_INVENTORY_HESS_20260910"
)
OUTPUT_DIRECTORY = os.path.join(
    EXPERIMENT_DIRECTORY,
    "GRAPH_ASSETS_TRAIN_THROUGH_2024",
)

MATRIX_PATH = (
    "project/hpc/Training/"
    "Prepared_Global_Matrix/global_features_all_stations_feature_engineered.pkl"
)
CANONICAL_NODE_SOURCE_PATH = (
    "project/hpc/Training/GNN/"
    "Experiments/REFORECAST_GRAPH_RANKING_2026H1/graphs/graph_obs_pre2026.npz"
)
ROUTING_NODES_PATH = "project/physical_graphs/gauge_nodes_upstream.csv"
NETWORK_DISTANCE_PATH = "project/physical_graphs/gauge_network_distance_miles.csv"
COHORT_CSV_PATH = os.path.join(EXPERIMENT_DIRECTORY, "GAUGE_COHORT_PRE2026.csv")
PROPOSED_SPLIT_PATH = os.path.join(EXPERIMENT_DIRECTORY, "PROPOSED_SPLIT.md")

OBSERVATION_GRAPH_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "graph_obs_train_through_2024_68nodes.npz",
)
IDENTITY_GRAPH_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "graph_identity_68nodes.npz",
)
SCORING_MASK_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "primary_scoring_mask_pre2026_68nodes.npz",
)
PAIR_TABLE_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "xcorr_pairs_train_through_2024.csv",
)
MANIFEST_PATH = os.path.join(OUTPUT_DIRECTORY, "graph_asset_manifest.json")
REPORT_PATH = os.path.join(OUTPUT_DIRECTORY, "GRAPH_PROTOCOL.md")

GRAPH_DISCOVERY_START_UTC = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
GRAPH_DISCOVERY_END_UTC = pd.Timestamp("2024-12-31 23:45:00", tz="UTC")
EXPECTED_NODE_COUNT = 68
EXPECTED_PRIMARY_SCORING_COUNT = 30

MAX_NETWORK_DISTANCE_MILES = 20.0
MAX_LAG_HOURS = 24
STEP_MINUTES = 15
MAX_LAG_STEPS = int(MAX_LAG_HOURS * 60 / STEP_MINUTES)
DETREND_WINDOW_STEPS = 288
FLATLINE_WINDOW_STEPS = 96
CORRELATION_THRESHOLD = 0.40
MINIMUM_OVERLAP_STEPS = 8640
MINIMUM_LAG_STEPS = 2


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


def row_normalize(raw_adjacency):
    row_sums = raw_adjacency.sum(axis=1, keepdims=True)
    denominator = np.where(row_sums > 0.0, row_sums, 1.0)
    normalized = raw_adjacency / denominator
    return normalized.astype("float32")


def peak_lag(first_values, second_values, max_lag_steps):
    first_centered = first_values - first_values.mean()
    second_centered = second_values - second_values.mean()
    cross_correlation = correlate(
        first_centered,
        second_centered,
        mode="full",
        method="fft",
    )
    lag_values = correlation_lags(
        len(first_centered),
        len(second_centered),
        mode="full",
    )
    accepted_lag_window = np.abs(lag_values) <= max_lag_steps
    accepted_correlations = cross_correlation[accepted_lag_window]
    accepted_lags = lag_values[accepted_lag_window]
    maximum_index = int(np.argmax(accepted_correlations))
    normalization = np.sqrt(
        float(first_centered @ first_centered)
        * float(second_centered @ second_centered)
    ) + 1.0e-9
    selected_lag = int(accepted_lags[maximum_index])
    selected_correlation = float(
        accepted_correlations[maximum_index] / normalization
    )
    return selected_lag, selected_correlation


def save_graph(path, nodes, raw_adjacency, lag_matrix, graph_role):
    normalized_adjacency = row_normalize(raw_adjacency.astype("float32"))
    np.savez(
        path,
        nodes=np.asarray(nodes),
        A=raw_adjacency.astype("float32"),
        A_norm=normalized_adjacency,
        LAG=lag_matrix.astype("int32"),
        graph_role=np.asarray(graph_role),
        graph_discovery_start_utc=np.asarray(str(GRAPH_DISCOVERY_START_UTC)),
        graph_discovery_end_utc=np.asarray(str(GRAPH_DISCOVERY_END_UTC)),
    )
    print(f"[WRITE] Graph asset: {path}")


print("[START] Building training-only 68-node graph assets.")
print(f"[START] Working directory: {os.getcwd()}")
print(f"[START] Output directory: {OUTPUT_DIRECTORY}")
print(f"[START] Graph discovery start: {GRAPH_DISCOVERY_START_UTC}")
print(f"[START] Graph discovery end: {GRAPH_DISCOVERY_END_UTC}")
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

source_paths = [
    MATRIX_PATH,
    CANONICAL_NODE_SOURCE_PATH,
    ROUTING_NODES_PATH,
    NETWORK_DISTANCE_PATH,
    COHORT_CSV_PATH,
    PROPOSED_SPLIT_PATH,
]
for source_path in source_paths:
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Required source file does not exist: {source_path}")
    print(f"[SOURCE] Found: {source_path}")

source_hashes = {}
for source_path in source_paths:
    source_hashes[source_path] = calculate_sha256(source_path)

print("[LOAD] Reading canonical 68-node order.")
canonical_archive = np.load(CANONICAL_NODE_SOURCE_PATH, allow_pickle=True)
if "nodes" not in canonical_archive.files:
    raise KeyError(
        "Canonical node source does not contain a nodes array. Keys: "
        + str(canonical_archive.files)
    )
canonical_nodes = [str(value) for value in canonical_archive["nodes"].tolist()]
if len(canonical_nodes) != EXPECTED_NODE_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_NODE_COUNT} canonical nodes, found {len(canonical_nodes)}"
    )
if len(set(canonical_nodes)) != len(canonical_nodes):
    raise ValueError("Canonical node names are not unique.")
node_index = {node: index for index, node in enumerate(canonical_nodes)}
print(f"[LOAD] Canonical node count: {len(canonical_nodes)}")

print("[LOAD] Reading the frozen primary scoring cohort.")
cohort = pd.read_csv(COHORT_CSV_PATH)
required_cohort_columns = {
    "cohort_id_sha256",
    "canonical_graph_order",
    "gauge",
    "include_primary",
    "selection_uses_2026_data",
}
missing_cohort_columns = required_cohort_columns.difference(cohort.columns)
if missing_cohort_columns:
    raise KeyError(
        "Cohort CSV lacks required columns: " + str(sorted(missing_cohort_columns))
    )
cohort = cohort.sort_values("canonical_graph_order").reset_index(drop=True)
cohort_nodes = cohort["gauge"].astype(str).tolist()
if cohort_nodes != canonical_nodes:
    raise ValueError("Cohort node order does not match the canonical 68-node order.")
if not cohort["selection_uses_2026_data"].eq(False).all():
    raise ValueError("The primary scoring cohort unexpectedly uses 2026 information.")
primary_scoring_mask = cohort["include_primary"].to_numpy(dtype=bool)
primary_scoring_indices = np.flatnonzero(primary_scoring_mask).astype("int32")
primary_scoring_nodes = [canonical_nodes[index] for index in primary_scoring_indices]
if int(primary_scoring_mask.sum()) != EXPECTED_PRIMARY_SCORING_COUNT:
    raise ValueError(
        "Expected 30 primary scoring gauges, found "
        + str(int(primary_scoring_mask.sum()))
    )
cohort_ids = cohort["cohort_id_sha256"].dropna().astype(str).unique().tolist()
if len(cohort_ids) != 1:
    raise ValueError(f"Expected one cohort ID, found {cohort_ids}")
cohort_id = cohort_ids[0]
recalculated_cohort_id = hashlib.sha256(
    "\n".join(primary_scoring_nodes).encode("utf-8")
).hexdigest()
if cohort_id != recalculated_cohort_id:
    raise ValueError(
        f"Cohort ID mismatch: stored {cohort_id}, recalculated {recalculated_cohort_id}"
    )
print(f"[LOAD] Primary scoring gauge count: {int(primary_scoring_mask.sum())}")
print(f"[LOAD] Cohort ID: {cohort_id}")

print("[WRITE] Saving the 68-position primary evaluation mask.")
np.savez(
    SCORING_MASK_PATH,
    nodes=np.asarray(canonical_nodes),
    primary_scoring_mask=primary_scoring_mask,
    primary_scoring_indices=primary_scoring_indices,
    primary_scoring_nodes=np.asarray(primary_scoring_nodes),
    cohort_id_sha256=np.asarray(cohort_id),
    mask_role=np.asarray("primary_evaluation_only_not_training_loss"),
    graph_node_count=np.asarray(EXPECTED_NODE_COUNT, dtype="int32"),
    primary_scoring_count=np.asarray(
        EXPECTED_PRIMARY_SCORING_COUNT,
        dtype="int32",
    ),
)
print(f"[WRITE] Scoring mask asset: {SCORING_MASK_PATH}")

print("[LOAD] Reading and slicing the engineered matrix before graph calculations.")
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
    raise ValueError(
        f"Engineered matrix contains {duplicate_timestamp_count} duplicate timestamps."
    )
matrix = matrix.loc[
    (matrix.index >= GRAPH_DISCOVERY_START_UTC)
    & (matrix.index <= GRAPH_DISCOVERY_END_UTC)
].copy()
if matrix.empty:
    raise ValueError("Training-only graph-discovery matrix is empty.")
expected_index = pd.date_range(
    GRAPH_DISCOVERY_START_UTC,
    GRAPH_DISCOVERY_END_UTC,
    freq=f"{STEP_MINUTES}min",
)
missing_timestamps = expected_index.difference(matrix.index)
if len(missing_timestamps) > 0:
    raise ValueError(
        "Training-only graph-discovery matrix is missing "
        + str(len(missing_timestamps))
        + " expected 15-minute timestamps."
    )
print(f"[LOAD] Matrix shape after training-only slice: {matrix.shape}")
print(f"[LOAD] Matrix range: {matrix.index.min()} through {matrix.index.max()}")

stage_columns = [f"{node}_stage_ft" for node in canonical_nodes]
missing_stage_columns = [
    column for column in stage_columns if column not in matrix.columns
]
if missing_stage_columns:
    raise KeyError(
        "Engineered matrix lacks canonical stage columns: "
        + str(missing_stage_columns)
    )

print("[LOAD] Reading routing eligibility and network-distance metadata.")
routing_nodes = pd.read_csv(ROUTING_NODES_PATH)
required_routing_columns = {"gauge", "snap_ft"}
missing_routing_columns = required_routing_columns.difference(routing_nodes.columns)
if missing_routing_columns:
    raise KeyError(
        "Routing node table lacks columns: " + str(sorted(missing_routing_columns))
    )
routing_nodes = routing_nodes.set_index("gauge")
network_distances = pd.read_csv(NETWORK_DISTANCE_PATH, index_col=0)

eligible_nodes = []
eligibility_rows = []
for node in canonical_nodes:
    if node not in routing_nodes.index:
        eligibility_rows.append(
            {
                "gauge": node,
                "eligible_for_xcorr": False,
                "reason": "absent from routing metadata",
            }
        )
        print(f"[ELIGIBILITY] Isolated, absent from routing metadata: {node}")
        continue
    snap_distance_feet = float(routing_nodes.loc[node, "snap_ft"])
    if snap_distance_feet > 5280.0:
        eligibility_rows.append(
            {
                "gauge": node,
                "eligible_for_xcorr": False,
                "reason": "routing snap distance exceeds one mile",
            }
        )
        print(f"[ELIGIBILITY] Isolated, snap distance exceeds one mile: {node}")
        continue
    eligible_nodes.append(node)
    eligibility_rows.append(
        {
            "gauge": node,
            "eligible_for_xcorr": True,
            "reason": "eligible",
        }
    )
print(f"[ELIGIBILITY] Eligible nodes: {len(eligible_nodes)} of {len(canonical_nodes)}")

print("[DQ] Applying the established flatline mask and rolling-median detrending.")
detrended_series = {}
flatline_masked_days = {}
for node in eligible_nodes:
    stage_column = f"{node}_stage_ft"
    stage_series = pd.to_numeric(matrix[stage_column], errors="coerce")
    flatline_mask = stage_series.rolling(
        FLATLINE_WINDOW_STEPS,
        center=True,
        min_periods=FLATLINE_WINDOW_STEPS,
    ).std().lt(1.0e-6)
    flatline_masked_days[node] = float(
        flatline_mask.sum() * STEP_MINUTES / 60.0 / 24.0
    )
    stage_series = stage_series.mask(flatline_mask)
    rolling_median = stage_series.rolling(
        DETREND_WINDOW_STEPS,
        center=True,
        min_periods=DETREND_WINDOW_STEPS // 3,
    ).median()
    detrended_series[node] = stage_series - rolling_median
detrended = pd.DataFrame(detrended_series)
print(f"[DQ] Detrended matrix shape: {detrended.shape}")

print("[VERIFY] Calibrating the lead-lag sign convention deterministically.")
random_generator = np.random.default_rng(0)
calibration_first = random_generator.standard_normal(4000)
calibration_second = np.roll(calibration_first, 10)
calibration_lag, calibration_correlation = peak_lag(
    calibration_first,
    calibration_second,
    50,
)
if calibration_lag == 0:
    raise AssertionError("Lead-lag sign calibration unexpectedly returned zero lag.")
lag_sign = 1 if calibration_lag > 0 else -1
print(f"[VERIFY] Calibration lag: {calibration_lag}")
print(f"[VERIFY] Calibration correlation: {calibration_correlation:.6f}")
print(f"[VERIFY] Lag sign multiplier: {lag_sign}")

print("[XCORR] Calculating training-only candidate pairs.")
pair_rows = []
for first_index, first_node in enumerate(eligible_nodes):
    print(
        f"[XCORR] Source {first_index + 1} of {len(eligible_nodes)}: {first_node}"
    )
    for second_node in eligible_nodes[first_index + 1:]:
        if first_node not in network_distances.index:
            continue
        if second_node not in network_distances.columns:
            continue
        distance_miles = pd.to_numeric(
            network_distances.loc[first_node, second_node],
            errors="coerce",
        )
        if not np.isfinite(distance_miles):
            continue
        if float(distance_miles) > MAX_NETWORK_DISTANCE_MILES:
            continue
        paired = detrended[[first_node, second_node]].dropna()
        overlap_steps = int(len(paired))
        if overlap_steps < MINIMUM_OVERLAP_STEPS:
            continue
        raw_lag, correlation_value = peak_lag(
            paired[first_node].to_numpy(),
            paired[second_node].to_numpy(),
            MAX_LAG_STEPS,
        )
        signed_lead_steps = int(lag_sign * raw_lag)
        accepted_edge = bool(
            correlation_value >= CORRELATION_THRESHOLD
            and abs(signed_lead_steps) >= MINIMUM_LAG_STEPS
        )
        upstream_node = ""
        downstream_node = ""
        if accepted_edge:
            if signed_lead_steps > 0:
                upstream_node = first_node
                downstream_node = second_node
            else:
                upstream_node = second_node
                downstream_node = first_node
        pair_rows.append(
            {
                "first_gauge": first_node,
                "second_gauge": second_node,
                "network_distance_miles": float(distance_miles),
                "overlap_steps": overlap_steps,
                "overlap_days": overlap_steps * STEP_MINUTES / 60.0 / 24.0,
                "signed_lead_steps": signed_lead_steps,
                "signed_lead_minutes": signed_lead_steps * STEP_MINUTES,
                "correlation": float(correlation_value),
                "accepted_edge": accepted_edge,
                "upstream_gauge": upstream_node,
                "downstream_gauge": downstream_node,
            }
        )

pair_table = pd.DataFrame(pair_rows)
if pair_table.empty:
    raise ValueError("No candidate cross-correlation pairs were produced.")
pair_table = pair_table.sort_values(
    ["first_gauge", "second_gauge"],
).reset_index(drop=True)
pair_table.to_csv(PAIR_TABLE_PATH, index=False, float_format="%.9f")
print(f"[WRITE] Candidate pair table: {PAIR_TABLE_PATH}")
print(f"[XCORR] Candidate pair count: {len(pair_table)}")
print(f"[XCORR] Accepted directed edge count: {int(pair_table['accepted_edge'].sum())}")

node_count = len(canonical_nodes)
observation_raw = np.zeros((node_count, node_count), dtype="float32")
observation_lag = np.zeros((node_count, node_count), dtype="int32")
accepted_pairs = pair_table.loc[pair_table["accepted_edge"]].copy()
for row in accepted_pairs.itertuples(index=False):
    upstream_index = node_index[str(row.upstream_gauge)]
    downstream_index = node_index[str(row.downstream_gauge)]
    observation_raw[downstream_index, upstream_index] = float(row.correlation)
    observation_lag[downstream_index, upstream_index] = abs(
        int(row.signed_lead_steps)
    )

save_graph(
    OBSERVATION_GRAPH_PATH,
    canonical_nodes,
    observation_raw,
    observation_lag,
    "observation_lead_lag_training_only_68_nodes",
)

identity_raw = np.eye(node_count, dtype="float32")
identity_lag = np.zeros((node_count, node_count), dtype="int32")
save_graph(
    IDENTITY_GRAPH_PATH,
    canonical_nodes,
    identity_raw,
    identity_lag,
    "identity_control_68_nodes",
)

print("[VERIFY] Re-reading and validating graph assets.")
observation_archive = np.load(OBSERVATION_GRAPH_PATH, allow_pickle=True)
identity_archive = np.load(IDENTITY_GRAPH_PATH, allow_pickle=True)
mask_archive = np.load(SCORING_MASK_PATH, allow_pickle=True)

for archive_name, archive in [
    ("observation", observation_archive),
    ("identity", identity_archive),
]:
    required_graph_keys = {"nodes", "A", "A_norm", "LAG"}
    missing_graph_keys = required_graph_keys.difference(archive.files)
    if missing_graph_keys:
        raise KeyError(
            f"{archive_name} graph lacks keys: {sorted(missing_graph_keys)}"
        )
    archive_nodes = [str(value) for value in archive["nodes"].tolist()]
    if archive_nodes != canonical_nodes:
        raise ValueError(f"{archive_name} graph node order changed.")
    if archive["A"].shape != (EXPECTED_NODE_COUNT, EXPECTED_NODE_COUNT):
        raise ValueError(f"{archive_name} raw adjacency shape is incorrect.")
    if archive["A_norm"].shape != (EXPECTED_NODE_COUNT, EXPECTED_NODE_COUNT):
        raise ValueError(f"{archive_name} normalized adjacency shape is incorrect.")
    if archive["LAG"].shape != (EXPECTED_NODE_COUNT, EXPECTED_NODE_COUNT):
        raise ValueError(f"{archive_name} lag matrix shape is incorrect.")

if not np.array_equal(identity_archive["A"], np.eye(node_count, dtype="float32")):
    raise ValueError("Identity graph raw adjacency is not exactly the identity matrix.")
if not np.array_equal(
    identity_archive["A_norm"],
    np.eye(node_count, dtype="float32"),
):
    raise ValueError("Identity graph normalized adjacency is not exactly identity.")
if np.any(identity_archive["LAG"] != 0):
    raise ValueError("Identity graph lag matrix contains nonzero values.")

observation_edge_mask = observation_archive["A"] > 0.0
observation_edge_count = int(observation_edge_mask.sum())
if observation_edge_count != int(pair_table["accepted_edge"].sum()):
    raise ValueError(
        "Observation graph edge count does not match the accepted pair table."
    )
if np.any(np.diag(observation_archive["A"]) != 0.0):
    raise ValueError("Observation graph unexpectedly contains self-edges.")
if np.any(observation_archive["LAG"][observation_edge_mask] < MINIMUM_LAG_STEPS):
    raise ValueError("Observation graph contains an accepted lag below the threshold.")
if np.any(observation_archive["LAG"][observation_edge_mask] > MAX_LAG_STEPS):
    raise ValueError("Observation graph contains an accepted lag above the threshold.")
if np.any(observation_archive["A"][observation_edge_mask] < CORRELATION_THRESHOLD):
    raise ValueError("Observation graph contains a weight below the correlation threshold.")
nonisolated_rows = observation_archive["A"].sum(axis=1) > 0.0
normalized_row_sums = observation_archive["A_norm"].sum(axis=1)
if not np.allclose(normalized_row_sums[nonisolated_rows], 1.0, atol=1.0e-6):
    raise ValueError("Observation graph nonisolated rows are not row-normalized.")
if not np.allclose(normalized_row_sums[~nonisolated_rows], 0.0, atol=1.0e-6):
    raise ValueError("Observation graph isolated rows are not zero.")

mask_nodes = [str(value) for value in mask_archive["nodes"].tolist()]
if mask_nodes != canonical_nodes:
    raise ValueError("Primary scoring mask node order changed.")
verified_mask = mask_archive["primary_scoring_mask"].astype(bool)
if not np.array_equal(verified_mask, primary_scoring_mask):
    raise ValueError("Primary scoring mask changed during serialization.")
if int(verified_mask.sum()) != EXPECTED_PRIMARY_SCORING_COUNT:
    raise ValueError("Primary scoring mask count is incorrect after serialization.")
if str(mask_archive["mask_role"].item()) != "primary_evaluation_only_not_training_loss":
    raise ValueError("Primary scoring mask role is incorrect.")

isolated_node_count = int((~nonisolated_rows).sum())
held_out_to_primary_edge_count = int(
    observation_edge_mask[np.ix_(primary_scoring_mask, ~primary_scoring_mask)].sum()
)
primary_to_primary_edge_count = int(
    observation_edge_mask[np.ix_(primary_scoring_mask, primary_scoring_mask)].sum()
)

output_paths = [
    OBSERVATION_GRAPH_PATH,
    IDENTITY_GRAPH_PATH,
    SCORING_MASK_PATH,
    PAIR_TABLE_PATH,
]
output_hashes = {}
for output_path in output_paths:
    output_hashes[output_path] = calculate_sha256(output_path)

manifest = {
    "protocol": "HESS chronological graph comparison",
    "graph_node_count": EXPECTED_NODE_COUNT,
    "graph_node_role": "complete operational node tensor",
    "primary_scoring_count": EXPECTED_PRIMARY_SCORING_COUNT,
    "primary_scoring_role": "evaluation only, not a training-loss or graph-node mask",
    "cohort_id_sha256": cohort_id,
    "graph_discovery_start_utc": str(GRAPH_DISCOVERY_START_UTC),
    "graph_discovery_end_utc": str(GRAPH_DISCOVERY_END_UTC),
    "validation_start_utc": "2025-01-05 00:00:00+00:00",
    "test_start_utc": "2026-01-01 00:00:00+00:00",
    "parameters": {
        "maximum_network_distance_miles": MAX_NETWORK_DISTANCE_MILES,
        "maximum_lag_hours": MAX_LAG_HOURS,
        "step_minutes": STEP_MINUTES,
        "detrend_window_steps": DETREND_WINDOW_STEPS,
        "flatline_window_steps": FLATLINE_WINDOW_STEPS,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "minimum_overlap_steps": MINIMUM_OVERLAP_STEPS,
        "minimum_lag_steps": MINIMUM_LAG_STEPS,
    },
    "results": {
        "eligible_xcorr_nodes": len(eligible_nodes),
        "candidate_pair_count": len(pair_table),
        "accepted_directed_edge_count": observation_edge_count,
        "isolated_row_count": isolated_node_count,
        "held_out_to_primary_directed_edge_count": held_out_to_primary_edge_count,
        "primary_to_primary_directed_edge_count": primary_to_primary_edge_count,
    },
    "source_hashes": source_hashes,
    "output_hashes": output_hashes,
    "source_duplicate_timestamp_count": duplicate_timestamp_count,
    "eligible_node_audit": eligibility_rows,
    "flatline_masked_days": flatline_masked_days,
}
with open(MANIFEST_PATH, "w", encoding="utf-8") as output_file:
    json.dump(manifest, output_file, indent=2, sort_keys=True)
    output_file.write("\n")
print(f"[WRITE] Manifest: {MANIFEST_PATH}")

report_lines = [
    "# Training-only graph protocol",
    "",
    "## Frozen decision",
    "",
    "The experimental ST-GNN remains a 68-node model. The observation-derived graph was ",
    "discovered from stage data spanning 1 January 2023 through 31 December 2024 only. ",
    "Therefore, its edges and weights do not use the 2025 validation/calibration period or ",
    "the January-June 2026 test period.",
    "",
    "The 30-gauge cohort is stored as an evaluation mask on the same 68-node tensor. It is ",
    "not a graph-node deletion mask and not a training-loss mask. The identity control is ",
    "also 68 by 68 and uses the identical node order.",
    "",
    "## Asset summary",
    "",
    "| item | value |",
    "| --- | --- |",
    f"| graph nodes | {EXPECTED_NODE_COUNT} |",
    f"| primary scoring gauges | {EXPECTED_PRIMARY_SCORING_COUNT} |",
    f"| cohort ID | `{cohort_id}` |",
    f"| graph discovery start | {GRAPH_DISCOVERY_START_UTC} |",
    f"| graph discovery end | {GRAPH_DISCOVERY_END_UTC} |",
    f"| routing-eligible nodes | {len(eligible_nodes)} |",
    f"| candidate gauge pairs | {len(pair_table)} |",
    f"| accepted directed edges | {observation_edge_count} |",
    f"| isolated adjacency rows | {isolated_node_count} |",
    f"| held-out-to-primary directed edges | {held_out_to_primary_edge_count} |",
    f"| primary-to-primary directed edges | {primary_to_primary_edge_count} |",
    "",
    "## Generated assets",
    "",
    f"- Observation graph: `{OBSERVATION_GRAPH_PATH}`",
    f"- Identity graph: `{IDENTITY_GRAPH_PATH}`",
    f"- Primary scoring mask: `{SCORING_MASK_PATH}`",
    f"- Cross-correlation audit table: `{PAIR_TABLE_PATH}`",
    f"- Machine-readable manifest: `{MANIFEST_PATH}`",
    "",
    "## Use in experiments",
    "",
    "1. Feed all 68 nodes to both the observation-graph and identity-graph ST-GNN.",
    "2. Apply the existing data-quality mask to training targets across the 68-node tensor.",
    "3. Do not apply the 30-gauge primary scoring mask to training loss.",
    "4. Apply the 30-gauge mask only when producing the primary comparative metrics.",
    "5. Preserve per-gauge outputs for all 68 gauges and label held-out gauges by cohort status.",
    "",
    "## Read-only guarantee",
    "",
    "The builder read the source matrix, routing metadata, prior graph only for canonical node ",
    "order, frozen cohort, and split report. It wrote only into this graph-asset directory. ",
    "No training, inference, production, source-data, or existing graph file was modified.",
]
with open(REPORT_PATH, "w", encoding="utf-8") as output_file:
    output_file.write("\n".join(report_lines))
    output_file.write("\n")
print(f"[WRITE] Protocol report: {REPORT_PATH}")

if "\u2014" in "\n".join(report_lines):
    raise AssertionError("Protocol report contains a prohibited em dash character.")

print("[DONE] Training-only graph assets built and verified successfully.")
print(f"[DONE] Observation directed edges: {observation_edge_count}")
print(f"[DONE] Primary scoring gauges: {int(primary_scoring_mask.sum())}")
print(f"[DONE] {OBSERVATION_GRAPH_PATH}")
print(f"[DONE] {IDENTITY_GRAPH_PATH}")
print(f"[DONE] {SCORING_MASK_PATH}")
print(f"[DONE] {PAIR_TABLE_PATH}")
print(f"[DONE] {MANIFEST_PATH}")
print(f"[DONE] {REPORT_PATH}")
