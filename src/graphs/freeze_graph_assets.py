"""Freeze the small System B assets and record all immutable source hashes.

This script copies only graphs, the stage-to-rain mapping, and the archived HRRR
forcing into the System B experiment. The large training and evaluation matrices
remain in their existing research locations and are pinned by SHA256.
"""
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone

import numpy as np

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
ASSET_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "frozen_assets")

SOURCE_DEVELOPMENT_GRAPH = (
    "project/Experiments/"
    "EVENT_SPLIT_INVENTORY_HESS_20260910/GRAPH_ASSETS_TRAIN_THROUGH_2024/"
    "graph_obs_train_through_2024_68nodes.npz"
)
SOURCE_FINAL_GRAPH = (
    "project/hpc/Training/GNN/Experiments/"
    "REFORECAST_GRAPH_RANKING_2026H1/graphs/graph_obs_pre2026.npz"
)
SOURCE_STAGE_TO_RAIN = (
    "src/data/stage_to_rain_gauge.csv"
)
SOURCE_HRRR_FORCING = (
    "project/Experiments/HRRR_FORCING_AND_LSTM_20260911/"
    "data/forcing/hrrr_issue_time_stitch_L2.npz"
)
SOURCE_TRAINING_MATRIX = (
    "project/hpc/Training/GNN/Experiments/"
    "REFORECAST_GRAPH_RANKING_2026H1/graphs/training_matrix_through_20251231.pkl"
)
SOURCE_EVALUATION_MATRIX = (
    "project/hpc/Experiments/"
    "HRRR_FORCING_AND_LSTM_20260911/frozen_inputs/"
    "global_features_all_stations_feature_engineered_20260911.pkl"
)
SOURCE_INPARISH_MANIFEST = (
    "project/Experiments/INPARISH51_20260813/"
    "outputs/in_parish_gauges.json"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_handle:
        while True:
            block = input_handle.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


source_paths = [
    SOURCE_DEVELOPMENT_GRAPH,
    SOURCE_FINAL_GRAPH,
    SOURCE_STAGE_TO_RAIN,
    SOURCE_HRRR_FORCING,
    SOURCE_TRAINING_MATRIX,
    SOURCE_EVALUATION_MATRIX,
    SOURCE_INPARISH_MANIFEST,
]
for source_path in source_paths:
    if not os.path.exists(source_path):
        raise FileNotFoundError("Required System B source does not exist: " + source_path)

os.makedirs(ASSET_DIRECTORY, exist_ok=True)
copy_operations = [
    (
        SOURCE_DEVELOPMENT_GRAPH,
        os.path.join(ASSET_DIRECTORY, "graph_obs_development_2023_2024.npz"),
    ),
    (
        SOURCE_FINAL_GRAPH,
        os.path.join(ASSET_DIRECTORY, "graph_obs_pre2026_471edges.npz"),
    ),
    (
        SOURCE_STAGE_TO_RAIN,
        os.path.join(ASSET_DIRECTORY, "stage_to_rain_gauge.csv"),
    ),
    (
        SOURCE_HRRR_FORCING,
        os.path.join(ASSET_DIRECTORY, "hrrr_issue_time_stitch_L2.npz"),
    ),
    (
        SOURCE_INPARISH_MANIFEST,
        os.path.join(ASSET_DIRECTORY, "in_parish_gauges.json"),
    ),
]

for source_path, destination_path in copy_operations:
    shutil.copy2(source_path, destination_path)
    print("[COPY]", source_path)
    print("[COPY] ->", destination_path)

development_graph_path = copy_operations[0][1]
final_graph_path = copy_operations[1][1]
development_graph = np.load(development_graph_path, allow_pickle=True)
final_graph = np.load(final_graph_path, allow_pickle=True)
development_nodes = [str(value) for value in development_graph["nodes"]]
final_nodes = [str(value) for value in final_graph["nodes"]]
if development_nodes != final_nodes:
    raise ValueError("Development and final graphs use different node orders.")
if len(final_nodes) != 68:
    raise ValueError("System B requires 68 graph nodes. Found: " + str(len(final_nodes)))
development_edge_count = int(np.count_nonzero(development_graph["A_norm"]))
final_edge_count = int(np.count_nonzero(final_graph["A_norm"]))
if development_edge_count != 332:
    raise ValueError(
        "Expected 332 development graph edges. Found: " + str(development_edge_count)
    )
if final_edge_count != 471:
    raise ValueError("Expected 471 final graph edges. Found: " + str(final_edge_count))

asset_records = []
for source_path, destination_path in copy_operations:
    asset_records.append(
        {
            "source_path": source_path,
            "frozen_path": destination_path,
            "source_sha256": sha256_file(source_path),
            "frozen_sha256": sha256_file(destination_path),
            "size_bytes": os.path.getsize(destination_path),
        }
    )
for source_path in [SOURCE_TRAINING_MATRIX, SOURCE_EVALUATION_MATRIX]:
    asset_records.append(
        {
            "source_path": source_path,
            "frozen_path": None,
            "source_sha256": sha256_file(source_path),
            "frozen_sha256": None,
            "size_bytes": os.path.getsize(source_path),
            "note": "Large immutable research asset referenced by hash; not duplicated.",
        }
    )

for asset_record in asset_records:
    if asset_record["frozen_path"] is not None:
        if asset_record["source_sha256"] != asset_record["frozen_sha256"]:
            raise ValueError(
                "Copied asset hash differs from its source: " + asset_record["frozen_path"]
            )

manifest = {
    "experiment": "SYSTEM_B_471_CHRONOLOGICAL_20260913",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "node_count": 68,
    "development_graph_edge_count": development_edge_count,
    "final_graph_edge_count": final_edge_count,
    "development_graph_cutoff_utc": "2024-12-31 23:45:00+00:00",
    "final_graph_cutoff_utc": "2025-12-31 23:45:00+00:00",
    "assets": asset_records,
}
manifest_path = os.path.join(ASSET_DIRECTORY, "ASSET_MANIFEST.json")
with open(manifest_path, "w", encoding="utf-8") as output_handle:
    json.dump(manifest, output_handle, indent=2)

print("[VERIFY] Development graph edges:", development_edge_count)
print("[VERIFY] Final graph edges:", final_edge_count)
print("[VERIFY] Node order is identical across graphs:", True)
print("[SAVE] Asset manifest:", manifest_path)
print("SYSTEM_B_ASSETS_READY")
