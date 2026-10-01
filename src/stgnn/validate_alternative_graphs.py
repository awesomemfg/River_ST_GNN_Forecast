"""Validate setup or completion of the matched System B comparison suite."""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
REMOTE_EXPERIMENT_ROOT = (
    "project/hpc/Experiments/"
    "SYSTEM_B_471_CHRONOLOGICAL_20260913"
)
MANIFEST_PATH = os.path.join(
    EXPERIMENT_ROOT,
    "frozen_assets",
    "comparisons",
    "alternative_graph_manifest.json",
)
OUTPUT_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "results", "comparisons")

parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=["setup", "complete"], required=True)
arguments = parser.parse_args()

EXPECTED_GRAPH_EDGES = {
    "obs_pre2026": (332, 471),
    "obs_pre2026_basin": (196, 279),
    "hecras": (963, 963),
    "dem": (366, 366),
    "dem_basin": (223, 223),
    "identity": (68, 68),
}
COMPARISON_ARMS = [
    ("stgnn", "obs_pre2026_basin"),
    ("stgnn", "hecras"),
    ("stgnn", "dem"),
    ("stgnn", "dem_basin"),
    ("stgnn", "identity"),
    ("lstm", "nodewise"),
]
ALL_LEARNED_ARMS = [("stgnn", "obs_pre2026")] + COMPARISON_ARMS
SEEDS = [101, 202, 303]
FORCINGS = ["observed", "hrrr"]


def calculate_sha256(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as input_handle:
        while True:
            block = input_handle.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


if not os.path.exists(MANIFEST_PATH):
    raise FileNotFoundError("Comparison asset manifest is missing: " + MANIFEST_PATH)
with open(MANIFEST_PATH, "r", encoding="utf-8") as manifest_handle:
    manifest = json.load(manifest_handle)

if str(manifest.get("experiment")) != "SYSTEM_B_471_CHRONOLOGICAL_20260913":
    raise ValueError("Comparison manifest has the wrong experiment identifier.")
if int(manifest.get("node_count", -1)) != 68:
    raise ValueError("Comparison manifest must declare 68 graph nodes.")
if int(manifest.get("score_gauge_count", -1)) != 51:
    raise ValueError("Comparison manifest must declare 51 scoring gauges.")
if int(manifest.get("test_origin_count", -1)) != 4344:
    raise ValueError("Comparison manifest must declare 4,344 test origins.")
if int(manifest.get("rain6_steps", -1)) != 6:
    raise ValueError("rain6 must contain six time steps.")
if float(manifest.get("rain6_hours", -1.0)) != 1.5:
    raise ValueError("rain6 must represent 1.5 hours.")
if int(manifest.get("rain24_steps", -1)) != 24:
    raise ValueError("rain24 must contain 24 time steps.")
if float(manifest.get("rain24_hours", -1.0)) != 6.0:
    raise ValueError("rain24 must represent six hours.")
if int(manifest.get("decoder_steps", -1)) != 96:
    raise ValueError("The decoder must contain 96 time steps.")
if float(manifest.get("decoder_hours", -1.0)) != 24.0:
    raise ValueError("The decoder must represent 24 hours.")

manifest_arms = {
    (str(arm["model_kind"]), str(arm["graph_key"])): arm
    for arm in manifest["arms"]
}
for model_kind, graph_key in ALL_LEARNED_ARMS:
    if (model_kind, graph_key) not in manifest_arms:
        raise ValueError(
            "Comparison manifest is missing arm: "
            + model_kind
            + " "
            + graph_key
        )

reference_nodes = None
for arm in manifest["arms"]:
    development_path = str(arm["development_graph"])
    final_path = str(arm["final_graph"])
    for graph_path, hash_key in [
        (development_path, "development_graph_sha256"),
        (final_path, "final_graph_sha256"),
    ]:
        if not os.path.exists(graph_path):
            raise FileNotFoundError("Graph asset is missing: " + graph_path)
        actual_hash = calculate_sha256(graph_path)
        if actual_hash != str(arm[hash_key]):
            raise ValueError("Graph hash mismatch: " + graph_path)
        graph_data = np.load(graph_path, allow_pickle=True)
        nodes = [str(value) for value in graph_data["nodes"]]
        adjacency = graph_data["A_norm"]
        if len(nodes) != 68 or adjacency.shape != (68, 68):
            raise ValueError("Graph has the wrong node count or adjacency shape: " + graph_path)
        if reference_nodes is None:
            reference_nodes = nodes
        if nodes != reference_nodes:
            raise ValueError("Graph node order differs across comparison assets: " + graph_path)
    graph_key = str(arm["graph_key"])
    if graph_key in EXPECTED_GRAPH_EDGES:
        expected_development_edges, expected_final_edges = EXPECTED_GRAPH_EDGES[graph_key]
        if int(arm["development_edges"]) != expected_development_edges:
            raise ValueError("Unexpected development edge count for: " + graph_key)
        if int(arm["final_edges"]) != expected_final_edges:
            raise ValueError("Unexpected final edge count for: " + graph_key)

print("[VALIDATE] Comparison assets and hard invariants are complete.")

if arguments.phase == "setup":
    print("SYSTEM_B_COMPARISON_SETUP_VALID")
else:
    learned_run_count = 0
    score_directory_count = 0
    for model_kind, graph_key in ALL_LEARNED_ARMS:
        arm = manifest_arms[(model_kind, graph_key)]
        expected_final_hash = str(arm["final_graph_sha256"])
        for seed in SEEDS:
            if graph_key == "obs_pre2026":
                development_root = os.path.join(EXPERIMENT_ROOT, "models", "development")
                final_root = os.path.join(EXPERIMENT_ROOT, "models", "final")
            else:
                development_root = os.path.join(
                    REMOTE_EXPERIMENT_ROOT,
                    "models",
                    "development",
                )
                final_root = os.path.join(REMOTE_EXPERIMENT_ROOT, "models", "final")
            run_name = model_kind + "_" + graph_key + "_seed" + str(seed)
            development_directory = os.path.join(development_root, run_name)
            final_directory = os.path.join(final_root, run_name)
            development_completion_path = os.path.join(
                development_directory,
                "RUN_COMPLETE.json",
            )
            final_completion_path = os.path.join(final_directory, "RUN_COMPLETE.json")
            development_history_path = os.path.join(
                development_directory,
                "train_history.json",
            )
            final_metadata_path = os.path.join(final_directory, "gnn_meta.json")
            for required_path in [
                development_completion_path,
                final_completion_path,
                development_history_path,
                final_metadata_path,
            ]:
                if not os.path.exists(required_path):
                    raise FileNotFoundError("Incomplete learned arm: " + required_path)
            with open(
                development_history_path,
                "r",
                encoding="utf-8",
            ) as development_history_handle:
                development_history = json.load(development_history_handle)
            with open(final_metadata_path, "r", encoding="utf-8") as metadata_handle:
                final_metadata = json.load(metadata_handle)
            selected_epoch = int(development_history["best_epoch"])
            if int(final_metadata["epochs_requested"]) != selected_epoch:
                raise ValueError("Final epoch count differs from development selection: " + run_name)
            if str(final_metadata["system"]) != "System B":
                raise ValueError("Final model is not labeled System B: " + run_name)
            if str(final_metadata["stage"]) != "final":
                raise ValueError("Final model has the wrong stage label: " + run_name)
            if str(final_metadata["model_kind"]) != model_kind:
                raise ValueError("Final model kind is wrong: " + run_name)
            if str(final_metadata["graph_key"]) != graph_key:
                raise ValueError("Final graph key is wrong: " + run_name)
            if int(final_metadata["seed"]) != seed:
                raise ValueError("Final model seed is wrong: " + run_name)
            if str(final_metadata["graph_sha256"]) != expected_final_hash:
                raise ValueError("Final model graph hash is wrong: " + run_name)
            if int(final_metadata["origin_count"]) != 26208:
                raise ValueError("Final model has the wrong number of fitting origins: " + run_name)
            if int(final_metadata["rain6_steps"]) != 6:
                raise ValueError("Final model changed rain6 steps: " + run_name)
            if float(final_metadata["rain6_hours"]) != 1.5:
                raise ValueError("Final model changed rain6 hours: " + run_name)
            if int(final_metadata["rain24_steps"]) != 24:
                raise ValueError("Final model changed rain24 steps: " + run_name)
            if float(final_metadata["rain24_hours"]) != 6.0:
                raise ValueError("Final model changed rain24 hours: " + run_name)
            if int(final_metadata["decoder_steps"]) != 96:
                raise ValueError("Final model changed decoder steps: " + run_name)
            if float(final_metadata["decoder_hours"]) != 24.0:
                raise ValueError("Final model changed decoder hours: " + run_name)
            learned_run_count += 1

            for forcing in FORCINGS:
                if graph_key == "obs_pre2026":
                    score_root = os.path.join(EXPERIMENT_ROOT, "scored")
                else:
                    score_root = os.path.join(
                        REMOTE_EXPERIMENT_ROOT,
                        "scored",
                        "comparisons",
                    )
                score_name = run_name + "_" + forcing
                score_directory = os.path.join(score_root, score_name)
                lead_path = os.path.join(
                    score_directory,
                    "reforecast_2026H1_lead_metrics.csv",
                )
                gauge_path = os.path.join(
                    score_directory,
                    "reforecast_2026H1_metrics.csv",
                )
                origin_path = os.path.join(
                    score_directory,
                    "reforecast_2026H1_origin_window_metrics.csv",
                )
                score_metadata_path = os.path.join(score_directory, "score_meta.json")
                for required_path in [
                    lead_path,
                    gauge_path,
                    origin_path,
                    score_metadata_path,
                ]:
                    if not os.path.exists(required_path):
                        raise FileNotFoundError("Incomplete score directory: " + required_path)
                lead_frame = pd.read_csv(lead_path)
                if lead_frame["lead_step"].nunique() != 96:
                    raise ValueError("Score does not contain 96 forecast steps: " + score_name)
                if int(lead_frame["lead_step"].min()) != 1:
                    raise ValueError("Score does not begin at lead step one: " + score_name)
                if int(lead_frame["lead_step"].max()) != 96:
                    raise ValueError("Score does not end at lead step 96: " + score_name)
                with open(score_metadata_path, "r", encoding="utf-8") as score_handle:
                    score_metadata = json.load(score_handle)
                if int(score_metadata["origins"]) != 4344:
                    raise ValueError("Score has the wrong origin count: " + score_name)
                score_directory_count += 1

    result_files = [
        "fixed_lead_summary_by_seed.csv",
        "training_epoch_selection_summary.csv",
        "fixed_lead_summary_across_seeds.csv",
        "lead_curve_summary_by_seed.csv",
        "lead_curve_summary_across_seeds.csv",
        "gauge_metrics_all_seeds.csv",
        "origin_window_summary_by_seed.csv",
        "origin_gauge_mean_RMSE_by_seed.csv",
        "model_and_graph_ranking.csv",
        "paired_graph_effects_vs_observation.csv",
        "COMPARISON_RESULT_SUMMARY.json",
    ]
    for result_file in result_files:
        result_path = os.path.join(OUTPUT_DIRECTORY, result_file)
        if not os.path.exists(result_path):
            raise FileNotFoundError("Aggregated comparison result is missing: " + result_path)

    audit = {
        "status": "complete",
        "experiment": "SYSTEM_B_471_CHRONOLOGICAL_20260913",
        "learned_model_runs": learned_run_count,
        "score_directories": score_directory_count,
        "seeds": SEEDS,
        "forcings": FORCINGS,
        "test_origins": 4344,
        "forecast_steps": 96,
        "graph_nodes": 68,
        "scoring_gauges": 51,
    }
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    completion_audit_path = os.path.join(
        OUTPUT_DIRECTORY,
        "COMPARISON_COMPLETION_AUDIT.json",
    )
    with open(completion_audit_path, "w", encoding="utf-8") as audit_handle:
        json.dump(audit, audit_handle, indent=2)
    print("[SAVE]", completion_audit_path)
    print("[VALIDATE] Learned model runs:", learned_run_count)
    print("[VALIDATE] Score directories:", score_directory_count)
    print("SYSTEM_B_COMPARISON_COMPLETE_VALID")
