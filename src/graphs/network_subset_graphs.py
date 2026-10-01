"""Create cutoff-correct System B graphs for boundary and gauge-count tests."""
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
SYSTEM_B_ROOT = os.path.join(
    os.path.dirname(EXPERIMENT_ROOT),
    "SYSTEM_B_471_CHRONOLOGICAL_20260913",
)
SYSTEM_B_ASSETS = os.path.join(SYSTEM_B_ROOT, "frozen_assets")
LEGACY_LADDER_MANIFEST = os.path.join(
    os.path.dirname(EXPERIMENT_ROOT),
    "INPARISH_LADDER_20260828",
    "outputs",
    "inparish_ladder_manifest.json",
)
OUTPUT_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "frozen_assets")
BASE_SYSTEM_B_TRAINER = os.path.join(
    SYSTEM_B_ROOT,
    "scripts",
    "train.py",
)
VARIABLE_NODE_TRAINER = os.path.join(
    SCRIPT_DIRECTORY,
    "train_network_subsets.py",
)

DEVELOPMENT_CONTROL_PATH = os.path.join(
    SYSTEM_B_ASSETS,
    "graph_obs_development_2023_2024.npz",
)
FINAL_CONTROL_PATH = os.path.join(
    SYSTEM_B_ASSETS,
    "graph_obs_pre2026_471edges.npz",
)
INPARISH_MANIFEST_PATH = os.path.join(
    SYSTEM_B_ASSETS,
    "in_parish_gauges.json",
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


def row_normalize(adjacency):
    normalized = np.zeros_like(adjacency, dtype=np.float32)
    row_sums = adjacency.sum(axis=1)
    nonzero_rows = row_sums > 0.0
    normalized[nonzero_rows] = (
        adjacency[nonzero_rows] / row_sums[nonzero_rows, None]
    )
    return normalized


def build_subset(source_archive, retained_nodes, output_path, graph_role):
    source_nodes = [str(value) for value in source_archive["nodes"]]
    node_to_index = {
        node: node_index
        for node_index, node in enumerate(source_nodes)
    }
    missing_nodes = [
        node
        for node in retained_nodes
        if node not in node_to_index
    ]
    if missing_nodes:
        raise ValueError(
            "Retained nodes are absent from the System B control graph: "
            + str(missing_nodes)
        )
    retained_set = set(retained_nodes)
    ordered_nodes = [
        node
        for node in source_nodes
        if node in retained_set
    ]
    if len(ordered_nodes) != len(retained_nodes):
        raise ValueError("The retained node list contains duplicates.")
    retained_indices = np.asarray(
        [node_to_index[node] for node in ordered_nodes],
        dtype=np.int64,
    )
    source_adjacency = source_archive["A"].astype(np.float32)
    source_lags = source_archive["LAG"].astype(np.int32)
    subset_adjacency = source_adjacency[
        np.ix_(retained_indices, retained_indices)
    ].astype(np.float32)
    subset_lags = source_lags[
        np.ix_(retained_indices, retained_indices)
    ].astype(np.int32)
    subset_normalized = row_normalize(subset_adjacency)
    row_sums = subset_normalized.sum(axis=1)
    nonzero_rows = row_sums > 0.0
    if not np.allclose(row_sums[nonzero_rows], 1.0, atol=1.0e-6):
        raise ValueError("A retained graph does not have row-normalized nonzero rows.")
    np.savez_compressed(
        output_path,
        nodes=np.asarray(ordered_nodes),
        A=subset_adjacency,
        A_norm=subset_normalized,
        LAG=subset_lags,
        graph_role=np.asarray(graph_role),
    )
    return {
        "nodes": ordered_nodes,
        "node_count": len(ordered_nodes),
        "edge_count": int(np.count_nonzero(subset_adjacency)),
        "isolated_node_count": int(
            np.count_nonzero(np.count_nonzero(subset_adjacency, axis=1) == 0)
        ),
        "path": os.path.relpath(output_path, EXPERIMENT_ROOT),
        "sha256": sha256_file(output_path),
    }


required_paths = [
    DEVELOPMENT_CONTROL_PATH,
    FINAL_CONTROL_PATH,
    INPARISH_MANIFEST_PATH,
    LEGACY_LADDER_MANIFEST,
    BASE_SYSTEM_B_TRAINER,
    VARIABLE_NODE_TRAINER,
]
for required_path in required_paths:
    if not os.path.exists(required_path):
        raise FileNotFoundError("Required source asset is missing: " + required_path)

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

with open(INPARISH_MANIFEST_PATH, "r", encoding="utf-8") as input_handle:
    inparish_manifest = json.load(input_handle)
with open(LEGACY_LADDER_MANIFEST, "r", encoding="utf-8") as input_handle:
    ladder_manifest = json.load(input_handle)

inparish_nodes = [str(node) for node in inparish_manifest["nodes"]]
if len(inparish_nodes) != 51:
    raise ValueError("The frozen System B in-parish cohort must contain 51 nodes.")
if any(node[4:6] == "00" for node in inparish_nodes):
    raise ValueError("An external supporting gauge appears in the in-parish cohort.")

arm_nodes = {
    "inparish51": inparish_nodes,
}
for size in [40, 30, 20, 10]:
    for draw in [1, 2, 3]:
        arm_key = "ipsub" + str(size) + "_d" + str(draw)
        nodes = [
            str(node)
            for node in ladder_manifest["arms"][arm_key]["nodes"]
        ]
        if len(nodes) != size:
            raise ValueError(
                arm_key
                + " contains "
                + str(len(nodes))
                + " nodes instead of "
                + str(size)
            )
        if not set(nodes).issubset(set(inparish_nodes)):
            raise ValueError(arm_key + " contains a node outside the 51-node parent.")
        arm_nodes[arm_key] = nodes

for draw in [1, 2, 3]:
    previous_nodes = set(inparish_nodes)
    for size in [40, 30, 20, 10]:
        arm_key = "ipsub" + str(size) + "_d" + str(draw)
        current_nodes = set(arm_nodes[arm_key])
        if not current_nodes < previous_nodes:
            raise ValueError("The nested-node invariant failed for " + arm_key)
        previous_nodes = current_nodes

development_control = np.load(DEVELOPMENT_CONTROL_PATH, allow_pickle=True)
final_control = np.load(FINAL_CONTROL_PATH, allow_pickle=True)
development_nodes = [str(value) for value in development_control["nodes"]]
final_nodes = [str(value) for value in final_control["nodes"]]
if development_nodes != final_nodes:
    raise ValueError("The System B development and final node orders differ.")
if int(np.count_nonzero(development_control["A"])) != 332:
    raise ValueError("The System B development control does not contain 332 edges.")
if int(np.count_nonzero(final_control["A"])) != 471:
    raise ValueError("The System B final control does not contain 471 edges.")

asset_manifest = {
    "experiment": "SYSTEM_B_BOUNDARY_INPARISH_20260914",
    "system": "System B",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_assets": {
        "development_control": {
            "path": DEVELOPMENT_CONTROL_PATH,
            "sha256": sha256_file(DEVELOPMENT_CONTROL_PATH),
            "node_count": len(development_nodes),
            "edge_count": int(np.count_nonzero(development_control["A"])),
        },
        "final_control": {
            "path": FINAL_CONTROL_PATH,
            "sha256": sha256_file(FINAL_CONTROL_PATH),
            "node_count": len(final_nodes),
            "edge_count": int(np.count_nonzero(final_control["A"])),
        },
        "inparish_manifest": {
            "path": INPARISH_MANIFEST_PATH,
            "sha256": sha256_file(INPARISH_MANIFEST_PATH),
        },
        "legacy_ladder_manifest_node_lists_only": {
            "path": LEGACY_LADDER_MANIFEST,
            "sha256": sha256_file(LEGACY_LADDER_MANIFEST),
            "note": "Only the predeclared node identities are reused. No legacy weights, scores, or graphs are used.",
        },
        "base_system_b_trainer": {
            "path": BASE_SYSTEM_B_TRAINER,
            "sha256": sha256_file(BASE_SYSTEM_B_TRAINER),
        },
        "variable_node_trainer": {
            "path": VARIABLE_NODE_TRAINER,
            "sha256": sha256_file(VARIABLE_NODE_TRAINER),
            "note": "This is the base System B trainer with only output routing, experiment labeling, and explicit expected-node-count support added.",
        },
    },
    "arms": {},
}

print("[PREPARE] Current directory:", os.getcwd())
print("[PREPARE] Experiment root:", EXPERIMENT_ROOT)
print("[PREPARE] Arm count:", len(arm_nodes))

for arm_key, retained_nodes in arm_nodes.items():
    development_output_path = os.path.join(
        OUTPUT_DIRECTORY,
        "graph_" + arm_key + "_development.npz",
    )
    final_output_path = os.path.join(
        OUTPUT_DIRECTORY,
        "graph_" + arm_key + "_final.npz",
    )
    development_record = build_subset(
        development_control,
        retained_nodes,
        development_output_path,
        "System B development graph through 2024",
    )
    final_record = build_subset(
        final_control,
        retained_nodes,
        final_output_path,
        "System B final graph through 2025",
    )
    if development_record["nodes"] != final_record["nodes"]:
        raise ValueError("Development and final node orders differ for " + arm_key)
    asset_manifest["arms"][arm_key] = {
        "node_count": len(retained_nodes),
        "draw": None if arm_key == "inparish51" else int(arm_key[-1]),
        "external_node_count": sum(
            1
            for node in retained_nodes
            if node[4:6] == "00"
        ),
        "nodes": final_record["nodes"],
        "development_graph": development_record,
        "final_graph": final_record,
    }
    print(
        "[ARM]",
        arm_key,
        "nodes=",
        len(retained_nodes),
        "development_edges=",
        development_record["edge_count"],
        "final_edges=",
        final_record["edge_count"],
    )

manifest_path = os.path.join(OUTPUT_DIRECTORY, "ASSET_MANIFEST.json")
with open(manifest_path, "w", encoding="utf-8") as output_handle:
    json.dump(asset_manifest, output_handle, indent=2)

print("[SAVED]", manifest_path)
print("SYSTEM_B_BOUNDARY_ASSETS_COMPLETE")
