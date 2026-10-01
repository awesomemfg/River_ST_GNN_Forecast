"""Build the identity graph that turns the ST-GNN into a GRU-only baseline.

Why this exists
---------------
The ST-GNN and the LSTM baseline differ in two ways at once, so the present comparison cannot say
which difference matters:

  STGNNQ         GRU encoder and GRU cell decoder, plus the graph message sum_u A[d,u] h_u
                 (stgnn_models.py lines 30, 31 and 42)
  NodewiseLSTMQ  LSTM encoder and LSTM cell decoder, with the gauge's own hidden state in place of
                 the graph message (stgnn_models.py lines 61, 62 and 73)

Setting the adjacency to the identity makes the graph message equal to the gauge's own hidden state,
which is exactly what the nodewise baseline feeds itself. The model is then a gauge-by-gauge GRU with
no exchange between gauges, identical to the ST-GNN in cell type, width, channels, loss, epoch count
and seeds. Training that alongside the existing models isolates message passing.

The real graph has a zero diagonal (471 off-diagonal weights, no self-loops, and 18 of the 68 nodes
with no edge at all), so the identity graph is a genuine no-graph control rather than a subset of the
graph now in use.

No model code changes. The trainer's 471-edge check applies only when the stage is final, the kind is
stgnn and the graph key is obs_pre2026 (train.py line 286), so a run with --graph-key
identity passes that gate by design.

Usage:
  conda run -n operational python -u identity_graph.py
"""
import hashlib
import json
import os

import numpy as np

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
ASSET_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "frozen_assets")

SOURCE_GRAPH = (
    "project/Experiments/"
    "SYSTEM_B_471_CHRONOLOGICAL_20260913/frozen_assets/graph_obs_pre2026_471edges.npz"
)
OUTPUT_GRAPH = os.path.join(ASSET_DIRECTORY, "graph_identity_68nodes.npz")

os.makedirs(ASSET_DIRECTORY, exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


source = np.load(SOURCE_GRAPH, allow_pickle=True)
nodes = [str(value) for value in source["nodes"]]
source_adjacency = source["A_norm"]
if len(nodes) != 68:
    raise ValueError("System B requires exactly 68 nodes. Found: " + str(len(nodes)))
if int(np.count_nonzero(np.diag(source_adjacency))) != 0:
    raise ValueError("The source graph carries self-loops, so the identity control is not comparable.")

identity = np.eye(len(nodes), dtype=source_adjacency.dtype)
np.savez_compressed(OUTPUT_GRAPH, nodes=np.asarray(nodes), A_norm=identity)

print("[identity] source graph:", SOURCE_GRAPH)
print("[identity] source nonzero directed weights:", int(np.count_nonzero(source_adjacency)))
print("[identity] source self-loops:", int(np.count_nonzero(np.diag(source_adjacency))))
print("[identity] wrote:", OUTPUT_GRAPH)
print("[identity] nodes:", len(nodes))
print("[identity] nonzero directed weights:", int(np.count_nonzero(identity)))
print("[identity] sha256:", sha256_file(OUTPUT_GRAPH))

manifest = {
    "purpose": "GRU-only control: the ST-GNN with message passing removed",
    "source_graph": SOURCE_GRAPH,
    "source_sha256": sha256_file(SOURCE_GRAPH),
    "output_graph": OUTPUT_GRAPH,
    "output_sha256": sha256_file(OUTPUT_GRAPH),
    "nodes": len(nodes),
    "nonzero_directed_weights": int(np.count_nonzero(identity)),
}
manifest_path = os.path.join(ASSET_DIRECTORY, "IDENTITY_GRAPH_MANIFEST.json")
with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
print("[identity] manifest:", manifest_path)
print("IDENTITY_GRAPH_READY", flush=True)
