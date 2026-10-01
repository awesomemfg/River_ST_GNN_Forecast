"""Build leakage-safe and matched 68-node graph assets for the 2026H1 reforecast ranking.

The primary observation lead-lag graph is discovered only from observations available through
2025-12-31 23:45 UTC. The four physical or restricted alternatives are aligned to the same
68-node order and row-normalized after MBSA4570 is removed. The deployed graph, whose discovery
record extends into 2026, is retained only as a labeled sensitivity reference.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags

LOCAL_EXPERIMENT_ROOT = "project/Experiments/REFORECAST_GRAPH_RANKING_2026H1"
MOUNT_GNN_ROOT = "project/hpc/Training/GNN"
GRAPH_OUTPUT_DIR = os.path.join(
    MOUNT_GNN_ROOT,
    "Experiments",
    "REFORECAST_GRAPH_RANKING_2026H1",
    "graphs",
)
INPUT_PICKLE = (
    "project/hpc/Training/"
    "Prepared_Global_Matrix/global_features_all_stations_feature_engineered.pkl"
)
ROUTING_ROOT = "project/physical_graphs"
ROUTING_NODES_CSV = os.path.join(ROUTING_ROOT, "gauge_nodes_upstream.csv")
NETWORK_DISTANCE_CSV = os.path.join(ROUTING_ROOT, "gauge_network_distance_miles.csv")
TRAIN_END_UTC = "2025-12-31 23:45:00+00:00"
TRAINING_SNAPSHOT_PATH = os.path.join(
    GRAPH_OUTPUT_DIR,
    "training_matrix_through_20251231.pkl",
)

BASE_GRAPH_PATH = os.path.join(MOUNT_GNN_ROOT, "gnn_graph.npz")
SOURCE_GRAPH_PATHS = {
    "hecras": os.path.join(MOUNT_GNN_ROOT, "gnn_graph_physics.npz"),
    "dem": os.path.join(MOUNT_GNN_ROOT, "gnn_graph_tiff.npz"),
    "dem_basin": os.path.join(MOUNT_GNN_ROOT, "gnn_graph_tiff_basinmask.npz"),
}

MAX_NETWORK_DISTANCE_MILES = 20.0
MAX_LAG_HOURS = 24
STEP_MINUTES = 15
MAX_LAG_STEPS = int(MAX_LAG_HOURS * 60 / STEP_MINUTES)
DETREND_WINDOW_STEPS = 288
FLATLINE_WINDOW_STEPS = 96
CORRELATION_THRESHOLD = 0.40
MINIMUM_OVERLAP_STEPS = 8640
MINIMUM_LAG_STEPS = 2
TOP_K_CROSS_BASIN = 2

os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(LOCAL_EXPERIMENT_ROOT, "results"), exist_ok=True)

print("[INFO] Current directory:", os.getcwd())
print("[INFO] Input pickle:", INPUT_PICKLE)
print("[INFO] Graph output directory:", GRAPH_OUTPUT_DIR)
print("[INFO] Leakage-safe graph discovery cutoff:", TRAIN_END_UTC)

required_paths = [
    INPUT_PICKLE,
    ROUTING_NODES_CSV,
    NETWORK_DISTANCE_CSV,
    BASE_GRAPH_PATH,
] + list(SOURCE_GRAPH_PATHS.values())
for required_path in required_paths:
    if not os.path.exists(required_path):
        raise FileNotFoundError("Required graph input is missing: " + required_path)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def row_normalize(raw_adjacency):
    row_sums = raw_adjacency.sum(axis=1, keepdims=True)
    denominator = np.where(row_sums > 0.0, row_sums, 1.0)
    return (raw_adjacency / denominator).astype("float32")


def save_graph(graph_key, nodes, raw_adjacency, lag_matrix, description, discovery_end_utc, source_paths):
    output_path = os.path.join(GRAPH_OUTPUT_DIR, "graph_" + graph_key + ".npz")
    normalized_adjacency = row_normalize(raw_adjacency.astype("float32"))
    np.savez(
        output_path,
        nodes=np.asarray(nodes),
        A=raw_adjacency.astype("float32"),
        A_norm=normalized_adjacency,
        LAG=lag_matrix.astype("int32"),
    )
    positive_edges = int((raw_adjacency > 0.0).sum())
    diagonal_edges = int((np.diag(raw_adjacency) > 0.0).sum())
    isolated_rows = int(((raw_adjacency > 0.0).sum(axis=1) == 0).sum())
    record = {
        "graph_key": graph_key,
        "description": description,
        "path": output_path,
        "sha256": sha256_file(output_path),
        "node_count": len(nodes),
        "positive_edges": positive_edges,
        "diagonal_edges": diagonal_edges,
        "isolated_rows": isolated_rows,
        "discovery_end_utc": discovery_end_utc,
        "source_paths": source_paths,
    }
    print(
        "[GRAPH]",
        graph_key,
        "nodes=",
        len(nodes),
        "edges=",
        positive_edges,
        "diagonal=",
        diagonal_edges,
        "isolated=",
        isolated_rows,
    )
    print("[SAVED]", output_path)
    return record


def peak_lag(first_values, second_values, max_lag_steps):
    first_centered = first_values - first_values.mean()
    second_centered = second_values - second_values.mean()
    cross_correlation = correlate(first_centered, second_centered, mode="full", method="fft")
    lag_values = correlation_lags(len(first_centered), len(second_centered), mode="full")
    accepted = np.abs(lag_values) <= max_lag_steps
    accepted_correlations = cross_correlation[accepted]
    accepted_lags = lag_values[accepted]
    maximum_index = int(np.argmax(accepted_correlations))
    normalization = np.sqrt(
        float(first_centered @ first_centered) * float(second_centered @ second_centered)
    ) + 1.0e-9
    selected_lag = int(accepted_lags[maximum_index])
    selected_correlation = float(accepted_correlations[maximum_index] / normalization)
    return selected_lag, selected_correlation


base_graph = np.load(BASE_GRAPH_PATH, allow_pickle=True)
canonical_nodes = [str(value) for value in base_graph["nodes"]]
if len(canonical_nodes) != 68:
    raise Exception("Expected the canonical deployed graph to contain 68 nodes, found " + str(len(canonical_nodes)))
if "MBSA4570" in canonical_nodes:
    raise Exception("MBSA4570 must not be present in the canonical 68-node graph.")
if len(set(canonical_nodes)) != len(canonical_nodes):
    raise Exception("The canonical graph contains duplicate node names.")

print("[INFO] Loading the training matrix for leakage-safe graph discovery...")
dataframe = pd.read_pickle(INPUT_PICKLE).sort_index()
print("[INFO] Full matrix shape:", dataframe.shape)
print("[INFO] Full matrix range:", dataframe.index.min(), "to", dataframe.index.max())
cutoff = pd.Timestamp(TRAIN_END_UTC)
if dataframe.index.tz is None:
    cutoff = cutoff.tz_localize(None)
else:
    cutoff = cutoff.tz_convert(dataframe.index.tz)
dataframe = dataframe[dataframe.index <= cutoff].copy()
print("[INFO] Cutoff matrix shape:", dataframe.shape)
print("[INFO] Cutoff matrix range:", dataframe.index.min(), "to", dataframe.index.max())
if len(dataframe) < 10000:
    raise Exception("The cutoff matrix is unexpectedly short: " + str(len(dataframe)) + " rows")
print("[INFO] Saving the immutable cutoff matrix used by every graph-training job...")
dataframe.to_pickle(TRAINING_SNAPSHOT_PATH)
print("[SAVED]", TRAINING_SNAPSHOT_PATH)

routing_nodes = pd.read_csv(ROUTING_NODES_CSV).set_index("gauge")
network_distances = pd.read_csv(NETWORK_DISTANCE_CSV, index_col=0)
eligible_nodes = []
for node in canonical_nodes:
    if node not in routing_nodes.index:
        print("[WARN] Node is absent from routing metadata and will be isolated in observation graph:", node)
        continue
    snap_distance_feet = float(routing_nodes.loc[node, "snap_ft"])
    if snap_distance_feet > 5280.0:
        print("[WARN] Node is more than one mile from routing network and will be isolated:", node)
        continue
    stage_column = node + "_stage_ft"
    if stage_column not in dataframe.columns:
        print("[WARN] Node stage column is absent and will be isolated:", stage_column)
        continue
    eligible_nodes.append(node)

print("[INFO] Eligible nodes for observation cross-correlation:", len(eligible_nodes))

detrended_series = {}
masked_flatline_days = {}
for node in eligible_nodes:
    stage_column = node + "_stage_ft"
    stage = pd.to_numeric(dataframe[stage_column], errors="coerce")
    flatline_mask = (
        stage.rolling(
            FLATLINE_WINDOW_STEPS,
            center=True,
            min_periods=FLATLINE_WINDOW_STEPS,
        ).std()
        < 1.0e-6
    )
    masked_flatline_days[node] = float(flatline_mask.sum()) * STEP_MINUTES / 60.0 / 24.0
    stage = stage.mask(flatline_mask)
    rolling_median = stage.rolling(
        DETREND_WINDOW_STEPS,
        center=True,
        min_periods=DETREND_WINDOW_STEPS // 3,
    ).median()
    detrended_series[node] = stage - rolling_median

detrended_dataframe = pd.DataFrame(detrended_series)
print("[INFO] Detrended observation matrix shape:", detrended_dataframe.shape)

random_generator = np.random.default_rng(0)
calibration_first = random_generator.standard_normal(4000)
calibration_second = np.roll(calibration_first, 10)
calibration_lag, calibration_correlation = peak_lag(
    calibration_first,
    calibration_second,
    50,
)
lag_sign = 1.0 if calibration_lag > 0 else -1.0
print("[VERIFY] Lag sign calibration:", calibration_lag, calibration_correlation, "sign=", lag_sign)

pair_rows = []
for first_index, first_node in enumerate(eligible_nodes):
    print(
        "[XCORR] Processing source node",
        first_index + 1,
        "of",
        len(eligible_nodes),
        first_node,
    )
    for second_node in eligible_nodes[first_index + 1 :]:
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
        paired = detrended_dataframe[[first_node, second_node]].dropna()
        if len(paired) < MINIMUM_OVERLAP_STEPS:
            continue
        raw_lag, correlation_value = peak_lag(
            paired[first_node].to_numpy(),
            paired[second_node].to_numpy(),
            MAX_LAG_STEPS,
        )
        signed_lead_steps = int(lag_sign * raw_lag)
        pair_rows.append(
            {
                "A": first_node,
                "B": second_node,
                "net_mi": round(float(distance_miles), 2),
                "lag_min": int(signed_lead_steps * STEP_MINUTES),
                "corr": round(float(correlation_value), 6),
                "days": round(len(paired) * STEP_MINUTES / 60.0 / 24.0, 1),
            }
        )

pair_table = pd.DataFrame(pair_rows)
pair_csv_path = os.path.join(GRAPH_OUTPUT_DIR, "xcorr_pairs_pre2026.csv")
pair_table.to_csv(pair_csv_path, index=False)
print("[SAVED]", pair_csv_path, "rows=", len(pair_table))

node_index = {node: index for index, node in enumerate(canonical_nodes)}
node_count = len(canonical_nodes)
observation_raw = np.zeros((node_count, node_count), dtype="float32")
observation_lag = np.zeros((node_count, node_count), dtype="int32")
accepted_pair_count = 0
for row in pair_table.itertuples():
    if float(row.corr) < CORRELATION_THRESHOLD:
        continue
    if abs(int(row.lag_min)) < MINIMUM_LAG_STEPS * STEP_MINUTES:
        continue
    if int(row.lag_min) > 0:
        upstream_node = str(row.A)
        downstream_node = str(row.B)
    else:
        upstream_node = str(row.B)
        downstream_node = str(row.A)
    if upstream_node not in node_index or downstream_node not in node_index:
        continue
    upstream_index = node_index[upstream_node]
    downstream_index = node_index[downstream_node]
    observation_raw[downstream_index, upstream_index] = float(row.corr)
    observation_lag[downstream_index, upstream_index] = abs(int(row.lag_min)) // STEP_MINUTES
    accepted_pair_count += 1

print("[INFO] Leakage-safe accepted observation pairs:", accepted_pair_count)

manifest_records = []
manifest_records.append(
    save_graph(
        "obs_pre2026",
        canonical_nodes,
        observation_raw,
        observation_lag,
        "Observation lead-lag graph discovered only from observations through 2025-12-31",
        TRAIN_END_UTC,
        [INPUT_PICKLE, ROUTING_NODES_CSV, NETWORK_DISTANCE_CSV, pair_csv_path],
    )
)

basin_keep = np.zeros((node_count, node_count), dtype=bool)
for downstream_index, downstream_node in enumerate(canonical_nodes):
    cross_basin_candidates = []
    for upstream_index, upstream_node in enumerate(canonical_nodes):
        weight = float(observation_raw[downstream_index, upstream_index])
        if weight <= 0.0:
            continue
        same_basin = downstream_node[:2] == upstream_node[:2]
        downstream_external = downstream_node[4:6] == "00"
        if same_basin or downstream_external:
            basin_keep[downstream_index, upstream_index] = True
        else:
            cross_basin_candidates.append((weight, upstream_index))
    cross_basin_candidates.sort(reverse=True)
    for weight, upstream_index in cross_basin_candidates[:TOP_K_CROSS_BASIN]:
        if weight > 0.0:
            basin_keep[downstream_index, upstream_index] = True

observation_basin_raw = np.where(basin_keep, observation_raw, 0.0).astype("float32")
observation_basin_lag = np.where(basin_keep, observation_lag, 0).astype("int32")
manifest_records.append(
    save_graph(
        "obs_pre2026_basin",
        canonical_nodes,
        observation_basin_raw,
        observation_basin_lag,
        "Within-basin plus top-two cross-basin variant of the pre-2026 observation graph",
        TRAIN_END_UTC,
        [os.path.join(GRAPH_OUTPUT_DIR, "graph_obs_pre2026.npz")],
    )
)

deployed_raw = base_graph["A"].astype("float32")
deployed_lag = base_graph["LAG"].astype("int32")
manifest_records.append(
    save_graph(
        "obs_deployed_sensitivity",
        canonical_nodes,
        deployed_raw,
        deployed_lag,
        "Deployed observation lead-lag graph retained only as a non-independent sensitivity reference",
        "2026-06-02 graph build, with discovery record extending into 2026",
        [BASE_GRAPH_PATH, os.path.join(MOUNT_GNN_ROOT, "xcorr_pairs_3yr.csv")],
    )
)

for graph_key, source_path in SOURCE_GRAPH_PATHS.items():
    source_graph = np.load(source_path, allow_pickle=True)
    source_nodes = [str(value) for value in source_graph["nodes"]]
    missing_nodes = sorted(set(canonical_nodes) - set(source_nodes))
    extra_nodes = sorted(set(source_nodes) - set(canonical_nodes))
    if missing_nodes:
        raise Exception(graph_key + " source graph is missing canonical nodes: " + str(missing_nodes))
    if extra_nodes != ["MBSA4570"]:
        raise Exception(
            graph_key
            + " source graph must differ from the canonical set only by MBSA4570. Found extra nodes: "
            + str(extra_nodes)
        )
    source_indices = [source_nodes.index(node) for node in canonical_nodes]
    trimmed_raw = source_graph["A"][np.ix_(source_indices, source_indices)].astype("float32")
    if "LAG" in source_graph.files:
        trimmed_lag = source_graph["LAG"][np.ix_(source_indices, source_indices)].astype("int32")
    else:
        trimmed_lag = np.zeros_like(trimmed_raw, dtype="int32")
    if graph_key == "hecras":
        description = "HEC-RAS simulated-hydraulic connectivity aligned to the canonical 68 nodes"
    elif graph_key == "dem":
        description = "DEM downslope connectivity aligned to the canonical 68 nodes"
    else:
        description = "Within-basin DEM connectivity aligned to the canonical 68 nodes"
    manifest_records.append(
        save_graph(
            graph_key,
            canonical_nodes,
            trimmed_raw,
            trimmed_lag,
            description,
            "Physical graph, independent of the January-June 2026 verification observations",
            [source_path],
        )
    )

identity_raw = np.eye(node_count, dtype="float32")
identity_lag = np.zeros((node_count, node_count), dtype="int32")
manifest_records.append(
    save_graph(
        "identity",
        canonical_nodes,
        identity_raw,
        identity_lag,
        "No cross-gauge message-passing control using identity adjacency",
        "Not observation-derived",
        [],
    )
)

manifest = {
    "experiment": "REFORECAST_GRAPH_RANKING_2026H1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "training_and_graph_discovery_cutoff_utc": TRAIN_END_UTC,
    "verification_window_utc": [
        "2026-01-01 00:00:00+00:00",
        "2026-06-30 23:00:00+00:00",
    ],
    "canonical_node_count": len(canonical_nodes),
    "paper_in_parish_node_count": sum(node[4:6] != "00" for node in canonical_nodes),
    "excluded_duplicate_node": "MBSA4570",
    "primary_graph_keys": [
        "obs_pre2026",
        "obs_pre2026_basin",
        "hecras",
        "dem",
        "dem_basin",
        "identity",
    ],
    "sensitivity_graph_keys": ["obs_deployed_sensitivity"],
    "source_input_pickle": INPUT_PICKLE,
    "training_snapshot_pickle": TRAINING_SNAPSHOT_PATH,
    "training_snapshot_sha256": sha256_file(TRAINING_SNAPSHOT_PATH),
    "masked_flatline_days": masked_flatline_days,
    "graph_records": manifest_records,
}
manifest_path = os.path.join(GRAPH_OUTPUT_DIR, "graph_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
print("[SAVED]", manifest_path)

local_manifest_path = os.path.join(LOCAL_EXPERIMENT_ROOT, "results", "graph_manifest.json")
with open(local_manifest_path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
print("[SAVED]", local_manifest_path)
print("GRAPH_ASSET_BUILD_DONE")
