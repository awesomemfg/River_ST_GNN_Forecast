"""Prepare the frozen graph assets for the matched System B comparison suite.

The suite preserves the 68-node order and provides one graph asset for the
pre-2025 epoch-selection step and one for the final all-pre-2026 fit. Physical
graphs and identity adjacency are time-independent, so their two frozen copies
are byte-identical. Observation graphs are rebuilt or restricted at the legal
cutoff for each step.
"""
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
OUTPUT_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "frozen_assets", "comparisons")

SOURCE_GRAPH_DIRECTORY = (
    "project/hpc/Training/GNN/"
    "Experiments/REFORECAST_GRAPH_RANKING_2026H1/graphs"
)
SOURCE_OBSERVATION_DEVELOPMENT = os.path.join(
    EXPERIMENT_ROOT,
    "frozen_assets",
    "graph_obs_development_2023_2024.npz",
)
SOURCE_OBSERVATION_FINAL = os.path.join(
    EXPERIMENT_ROOT,
    "frozen_assets",
    "graph_obs_pre2026_471edges.npz",
)

TOP_K_CROSS_BASIN = 2


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_handle:
        while True:
            block = input_handle.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def row_normalize(raw_adjacency):
    row_sums = raw_adjacency.sum(axis=1, keepdims=True)
    denominators = np.where(row_sums > 0.0, row_sums, 1.0)
    return (raw_adjacency / denominators).astype("float32")


def save_graph(destination_path, nodes, raw_adjacency, lag_matrix, role, cutoff):
    normalized_adjacency = row_normalize(raw_adjacency.astype("float32"))
    np.savez(
        destination_path,
        nodes=np.asarray(nodes),
        A=raw_adjacency.astype("float32"),
        A_norm=normalized_adjacency,
        LAG=lag_matrix.astype("int32"),
        graph_role=np.asarray(role),
        graph_discovery_end_utc=np.asarray(cutoff),
    )
    print("[SAVE]", destination_path)
    print("[VERIFY] Nodes:", len(nodes))
    print("[VERIFY] Nonzero normalized edges:", int(np.count_nonzero(normalized_adjacency)))


def build_observation_basin_graph(source_path, destination_path, role, cutoff):
    source_archive = np.load(source_path, allow_pickle=True)
    nodes = [str(value) for value in source_archive["nodes"]]
    raw_adjacency = source_archive["A"].astype("float32")
    lag_matrix = source_archive["LAG"].astype("int32")
    node_count = len(nodes)
    basin_keep = np.zeros((node_count, node_count), dtype=bool)
    for downstream_index, downstream_node in enumerate(nodes):
        cross_basin_candidates = []
        for upstream_index, upstream_node in enumerate(nodes):
            weight = float(raw_adjacency[downstream_index, upstream_index])
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
    restricted_adjacency = np.where(basin_keep, raw_adjacency, 0.0).astype("float32")
    restricted_lag = np.where(basin_keep, lag_matrix, 0).astype("int32")
    save_graph(
        destination_path,
        nodes,
        restricted_adjacency,
        restricted_lag,
        role,
        cutoff,
    )


required_source_paths = [
    SOURCE_OBSERVATION_DEVELOPMENT,
    SOURCE_OBSERVATION_FINAL,
    os.path.join(SOURCE_GRAPH_DIRECTORY, "graph_obs_pre2026_basin.npz"),
    os.path.join(SOURCE_GRAPH_DIRECTORY, "graph_hecras.npz"),
    os.path.join(SOURCE_GRAPH_DIRECTORY, "graph_dem.npz"),
    os.path.join(SOURCE_GRAPH_DIRECTORY, "graph_dem_basin.npz"),
    os.path.join(SOURCE_GRAPH_DIRECTORY, "graph_identity.npz"),
]
for required_source_path in required_source_paths:
    if not os.path.exists(required_source_path):
        raise FileNotFoundError("Required comparison graph is missing: " + required_source_path)

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

observation_development_path = os.path.join(
    OUTPUT_DIRECTORY,
    "graph_obs_pre2026_development.npz",
)
observation_final_path = os.path.join(
    OUTPUT_DIRECTORY,
    "graph_obs_pre2026_final.npz",
)
shutil.copy2(SOURCE_OBSERVATION_DEVELOPMENT, observation_development_path)
shutil.copy2(SOURCE_OBSERVATION_FINAL, observation_final_path)
print("[COPY]", SOURCE_OBSERVATION_DEVELOPMENT, "->", observation_development_path)
print("[COPY]", SOURCE_OBSERVATION_FINAL, "->", observation_final_path)

observation_basin_development_path = os.path.join(
    OUTPUT_DIRECTORY,
    "graph_obs_pre2026_basin_development.npz",
)
build_observation_basin_graph(
    SOURCE_OBSERVATION_DEVELOPMENT,
    observation_basin_development_path,
    "System B observation-basin graph for chronological epoch selection",
    "2024-12-31 23:45:00+00:00",
)

copy_pairs = []
final_source_by_key = {
    "obs_pre2026_basin": os.path.join(SOURCE_GRAPH_DIRECTORY, "graph_obs_pre2026_basin.npz"),
    "hecras": os.path.join(SOURCE_GRAPH_DIRECTORY, "graph_hecras.npz"),
    "dem": os.path.join(SOURCE_GRAPH_DIRECTORY, "graph_dem.npz"),
    "dem_basin": os.path.join(SOURCE_GRAPH_DIRECTORY, "graph_dem_basin.npz"),
    "identity": os.path.join(SOURCE_GRAPH_DIRECTORY, "graph_identity.npz"),
}

observation_basin_final_path = os.path.join(
    OUTPUT_DIRECTORY,
    "graph_obs_pre2026_basin_final.npz",
)
copy_pairs.append((final_source_by_key["obs_pre2026_basin"], observation_basin_final_path))

for graph_key in ["hecras", "dem", "dem_basin", "identity"]:
    source_path = final_source_by_key[graph_key]
    development_destination = os.path.join(
        OUTPUT_DIRECTORY,
        "graph_" + graph_key + "_development.npz",
    )
    final_destination = os.path.join(
        OUTPUT_DIRECTORY,
        "graph_" + graph_key + "_final.npz",
    )
    copy_pairs.append((source_path, development_destination))
    copy_pairs.append((source_path, final_destination))

for source_path, destination_path in copy_pairs:
    shutil.copy2(source_path, destination_path)
    print("[COPY]", source_path, "->", destination_path)

canonical_nodes = [
    str(value)
    for value in np.load(observation_final_path, allow_pickle=True)["nodes"]
]
if len(canonical_nodes) != 68:
    raise ValueError("System B comparison graphs must contain 68 nodes.")

arm_records = []
arm_definitions = [
    (
        "obs_pre2026",
        "stgnn",
        observation_development_path,
        observation_final_path,
        "Observation-derived lead-lag graph",
    ),
    (
        "obs_pre2026_basin",
        "stgnn",
        observation_basin_development_path,
        observation_basin_final_path,
        "Basin-restricted observation-derived graph",
    ),
    (
        "hecras",
        "stgnn",
        os.path.join(OUTPUT_DIRECTORY, "graph_hecras_development.npz"),
        os.path.join(OUTPUT_DIRECTORY, "graph_hecras_final.npz"),
        "HEC-RAS hydraulic connectivity",
    ),
    (
        "dem",
        "stgnn",
        os.path.join(OUTPUT_DIRECTORY, "graph_dem_development.npz"),
        os.path.join(OUTPUT_DIRECTORY, "graph_dem_final.npz"),
        "DEM downslope connectivity",
    ),
    (
        "dem_basin",
        "stgnn",
        os.path.join(OUTPUT_DIRECTORY, "graph_dem_basin_development.npz"),
        os.path.join(OUTPUT_DIRECTORY, "graph_dem_basin_final.npz"),
        "Basin-restricted DEM connectivity",
    ),
    (
        "identity",
        "stgnn",
        os.path.join(OUTPUT_DIRECTORY, "graph_identity_development.npz"),
        os.path.join(OUTPUT_DIRECTORY, "graph_identity_final.npz"),
        "Identity adjacency with no cross-gauge messages",
    ),
    (
        "nodewise",
        "lstm",
        os.path.join(OUTPUT_DIRECTORY, "graph_identity_development.npz"),
        os.path.join(OUTPUT_DIRECTORY, "graph_identity_final.npz"),
        "Shared nodewise unidirectional LSTM",
    ),
]

for graph_key, model_kind, development_path, final_path, label in arm_definitions:
    development_archive = np.load(development_path, allow_pickle=True)
    final_archive = np.load(final_path, allow_pickle=True)
    development_nodes = [str(value) for value in development_archive["nodes"]]
    final_nodes = [str(value) for value in final_archive["nodes"]]
    if development_nodes != canonical_nodes or final_nodes != canonical_nodes:
        raise ValueError("Comparison node order differs for arm: " + graph_key)
    development_edges = int(np.count_nonzero(development_archive["A_norm"]))
    final_edges = int(np.count_nonzero(final_archive["A_norm"]))
    arm_records.append(
        {
            "graph_key": graph_key,
            "model_kind": model_kind,
            "label": label,
            "development_graph": development_path,
            "development_graph_sha256": sha256_file(development_path),
            "development_edges": development_edges,
            "final_graph": final_path,
            "final_graph_sha256": sha256_file(final_path),
            "final_edges": final_edges,
        }
    )
    print(
        "[ARM]",
        graph_key,
        "kind=",
        model_kind,
        "development_edges=",
        development_edges,
        "final_edges=",
        final_edges,
    )

manifest = {
    "experiment": "SYSTEM_B_471_CHRONOLOGICAL_20260913",
    "suite": "matched_graph_and_nodewise_lstm_comparisons",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "node_count": 68,
    "seeds": [101, 202, 303],
    "development_training_origins_utc": [
        "2023-01-04 00:00:00",
        "2024-12-30 23:00:00",
    ],
    "development_validation_origins_utc": [
        "2025-01-04 00:00:00",
        "2025-12-30 23:00:00",
    ],
    "final_fitting_origins_utc": [
        "2023-01-04 00:00:00",
        "2025-12-30 23:00:00",
    ],
    "test_origins_utc": [
        "2026-01-01 00:00:00",
        "2026-06-30 23:00:00",
    ],
    "test_origin_count": 4344,
    "score_gauge_count": 51,
    "rain6_steps": 6,
    "rain6_hours": 1.5,
    "rain24_steps": 24,
    "rain24_hours": 6.0,
    "decoder_steps": 96,
    "decoder_hours": 24.0,
    "postprocessing": "off",
    "arms": arm_records,
    "optional_bilstm": "deferred until mandatory matched arms complete",
}
manifest_path = os.path.join(OUTPUT_DIRECTORY, "alternative_graph_manifest.json")
with open(manifest_path, "w", encoding="utf-8") as output_handle:
    json.dump(manifest, output_handle, indent=2)

print("[SAVE]", manifest_path)
print("SYSTEM_B_COMPARISON_ASSETS_READY")
