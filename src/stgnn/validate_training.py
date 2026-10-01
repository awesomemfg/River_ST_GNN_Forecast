"""Audit the locked System B setup or the completed end-to-end experiment."""
import argparse
import hashlib
import json
import os
import py_compile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
MIKE_EXPERIMENT_ROOT = (
    "project/hpc/Experiments/"
    "SYSTEM_B_471_CHRONOLOGICAL_20260913"
)
SEEDS = [101, 202, 303]

parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=["setup", "complete"], required=True)
arguments = parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_handle:
        while True:
            block = input_handle.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


checks = []


def record_check(name, passed, details):
    checks.append(
        {
            "name": str(name),
            "passed": bool(passed),
            "details": str(details),
        }
    )
    print("[CHECK]", name, "PASS" if passed else "FAIL", details)
    if not passed:
        raise ValueError("System B validation failed: " + str(name) + ": " + str(details))


required_documents = [
    "README.md",
    "training_protocol.md",
    "HANDOFF_NOTES.md",
    "CHAT_HISTORY.md",
    "training_protocol_lock.json",
]
for document_name in required_documents:
    document_path = os.path.join(EXPERIMENT_ROOT, document_name)
    record_check(
        "document_exists_" + document_name,
        os.path.exists(document_path),
        document_path,
    )

with open(
    os.path.join(EXPERIMENT_ROOT, "training_protocol_lock.json"),
    "r",
    encoding="utf-8",
) as protocol_handle:
    protocol = json.load(protocol_handle)
record_check(
    "final_graph_locked_to_471_edges",
    int(protocol["final_fit"]["graph_edges"]) == 471,
    protocol["final_fit"]["graph_edges"],
)
record_check(
    "test_locked_to_4344_origins",
    int(protocol["test"]["origin_count"]) == 4344,
    protocol["test"]["origin_count"],
)
hard_invariants = protocol["hard_invariants"]
record_check(
    "rain6_invariant",
    int(hard_invariants["rain6_steps"]) == 6
    and float(hard_invariants["rain6_hours"]) == 1.5,
    hard_invariants,
)
record_check(
    "rain24_invariant",
    int(hard_invariants["rain24_steps"]) == 24
    and float(hard_invariants["rain24_hours"]) == 6.0,
    hard_invariants,
)
record_check(
    "decoder_invariant",
    int(hard_invariants["decoder_steps"]) == 96
    and float(hard_invariants["decoder_hours"]) == 24.0,
    hard_invariants,
)

asset_manifest_path = os.path.join(
    EXPERIMENT_ROOT,
    "frozen_assets",
    "ASSET_MANIFEST.json",
)
record_check(
    "asset_manifest_exists",
    os.path.exists(asset_manifest_path),
    asset_manifest_path,
)
with open(asset_manifest_path, "r", encoding="utf-8") as asset_handle:
    asset_manifest = json.load(asset_handle)
record_check(
    "asset_manifest_node_count",
    int(asset_manifest["node_count"]) == 68,
    asset_manifest["node_count"],
)
record_check(
    "asset_manifest_graph_counts",
    int(asset_manifest["development_graph_edge_count"]) == 332
    and int(asset_manifest["final_graph_edge_count"]) == 471,
    {
        "development": asset_manifest["development_graph_edge_count"],
        "final": asset_manifest["final_graph_edge_count"],
    },
)
for asset_record in asset_manifest["assets"]:
    source_path = asset_record["source_path"]
    record_check(
        "asset_source_exists_" + os.path.basename(source_path),
        os.path.exists(source_path),
        source_path,
    )
    current_source_sha256 = sha256_file(source_path)
    record_check(
        "asset_source_hash_" + os.path.basename(source_path),
        current_source_sha256 == asset_record["source_sha256"],
        current_source_sha256,
    )
    frozen_path = asset_record.get("frozen_path")
    if frozen_path is not None:
        record_check(
            "asset_frozen_exists_" + os.path.basename(frozen_path),
            os.path.exists(frozen_path),
            frozen_path,
        )
        current_frozen_sha256 = sha256_file(frozen_path)
        record_check(
            "asset_frozen_hash_" + os.path.basename(frozen_path),
            current_frozen_sha256 == asset_record["frozen_sha256"],
            current_frozen_sha256,
        )

development_graph_path = os.path.join(
    EXPERIMENT_ROOT,
    "frozen_assets",
    "graph_obs_development_2023_2024.npz",
)
final_graph_path = os.path.join(
    EXPERIMENT_ROOT,
    "frozen_assets",
    "graph_obs_pre2026_471edges.npz",
)
development_graph = np.load(development_graph_path, allow_pickle=True)
final_graph = np.load(final_graph_path, allow_pickle=True)
development_nodes = [str(value) for value in development_graph["nodes"]]
final_nodes = [str(value) for value in final_graph["nodes"]]
record_check(
    "graph_node_order_match",
    development_nodes == final_nodes and len(final_nodes) == 68,
    "68 identical ordered nodes",
)
record_check(
    "graph_matrix_counts",
    int(np.count_nonzero(development_graph["A_norm"])) == 332
    and int(np.count_nonzero(final_graph["A_norm"])) == 471,
    "development=332 final=471",
)

python_scripts = sorted(
    os.path.join(SCRIPT_DIRECTORY, filename)
    for filename in os.listdir(SCRIPT_DIRECTORY)
    if filename.endswith(".py")
)
for python_script in python_scripts:
    py_compile.compile(python_script, doraise=True)
record_check(
    "python_scripts_compile",
    True,
    str(len(python_scripts)) + " scripts",
)

with open(
    os.path.join(SCRIPT_DIRECTORY, "train.py"),
    "r",
    encoding="utf-8",
) as training_script_handle:
    training_script_text = training_script_handle.read()
with open(
    os.path.join(SCRIPT_DIRECTORY, "hindcast_before_fix.py"),
    "r",
    encoding="utf-8",
) as evaluation_script_handle:
    evaluation_script_text = evaluation_script_handle.read()
record_check(
    "no_centered_flatline_window",
    "center=True" not in training_script_text
    and "center=True" not in evaluation_script_text,
    "training and inference use trailing windows",
)
record_check(
    "no_random_validation_split",
    "validation_split=" not in training_script_text
    and "take(n_val)" not in training_script_text,
    "validation positions are a separate chronological dataset",
)

development_training_origins = pd.date_range(
    protocol["development"]["training_origins_utc"][0],
    protocol["development"]["training_origins_utc"][1],
    freq="h",
)
development_validation_origins = pd.date_range(
    protocol["development"]["validation_origins_utc"][0],
    protocol["development"]["validation_origins_utc"][1],
    freq="h",
)
final_training_origins = pd.date_range(
    protocol["final_fit"]["training_origins_utc"][0],
    protocol["final_fit"]["training_origins_utc"][1],
    freq="h",
)
test_origins = pd.date_range(
    protocol["test"]["origin_start_utc"],
    protocol["test"]["origin_end_utc"],
    freq="h",
)
record_check(
    "origin_counts",
    len(development_training_origins) == 17448
    and len(development_validation_origins) == 8664
    and len(final_training_origins) == 26208
    and len(test_origins) == 4344,
    {
        "development_training": len(development_training_origins),
        "development_validation": len(development_validation_origins),
        "final_training": len(final_training_origins),
        "test": len(test_origins),
    },
)
last_development_target = development_training_origins[-1] + pd.Timedelta(hours=24)
first_validation_history = development_validation_origins[0] - pd.Timedelta(
    hours=71,
    minutes=45,
)
record_check(
    "development_validation_window_separation",
    first_validation_history > last_development_target,
    {
        "last_training_target": str(last_development_target),
        "first_validation_history": str(first_validation_history),
    },
)

if arguments.phase == "complete":
    for completion_document_name in [
        "EXECUTION_RECORD.md",
        "RESULTS_AND_FINDINGS.md",
    ]:
        completion_document_path = os.path.join(
            EXPERIMENT_ROOT,
            completion_document_name,
        )
        record_check(
            "completion_document_exists_" + completion_document_name,
            os.path.exists(completion_document_path),
            completion_document_path,
        )

    development_selected_epochs = {}
    final_fit_epochs = {}
    for stage_name in ["development", "final"]:
        for seed in SEEDS:
            model_directory = os.path.join(
                MIKE_EXPERIMENT_ROOT,
                "models",
                stage_name,
                "stgnn_obs_pre2026_seed" + str(seed),
            )
            completion_path = os.path.join(model_directory, "RUN_COMPLETE.json")
            metadata_path = os.path.join(model_directory, "gnn_meta.json")
            history_path = os.path.join(model_directory, "train_history.json")
            weights_path = os.path.join(model_directory, "gnn.weights.h5")
            for model_file in [completion_path, metadata_path, history_path, weights_path]:
                record_check(
                    stage_name + "_model_file_" + str(seed) + "_" + os.path.basename(model_file),
                    os.path.exists(model_file),
                    model_file,
                )
            with open(metadata_path, "r", encoding="utf-8") as metadata_handle:
                metadata = json.load(metadata_handle)
            with open(history_path, "r", encoding="utf-8") as history_handle:
                history = json.load(history_handle)
            record_check(
                stage_name + "_metadata_" + str(seed),
                str(metadata["system"]) == "System B"
                and str(metadata["stage"]) == stage_name
                and int(metadata["seed"]) == seed
                and int(metadata["rain6_steps"]) == 6
                and int(metadata["rain24_steps"]) == 24
                and int(metadata["decoder_steps"]) == 96,
                "metadata identity and rainfall invariants",
            )
            if stage_name == "final":
                final_fit_epochs[seed] = int(metadata["epochs_requested"])
                record_check(
                    "final_model_graph_" + str(seed),
                    int(metadata["graph_nonzero_count"]) == 471,
                    metadata["graph_nonzero_count"],
                )
                record_check(
                    "final_model_origin_count_" + str(seed),
                    int(metadata["origin_count"]) == 26208,
                    metadata["origin_count"],
                )
            else:
                development_selected_epochs[seed] = int(history["best_epoch"])

    for seed in SEEDS:
        record_check(
            "selected_epoch_transferred_to_final_fit_" + str(seed),
            int(development_selected_epochs[seed]) == int(final_fit_epochs[seed]),
            {
                "development_selected_epoch": development_selected_epochs[seed],
                "final_fit_epochs": final_fit_epochs[seed],
            },
        )

    support_inventory_path = os.path.join(
        EXPERIMENT_ROOT,
        "results",
        "stage_support_inventory.csv",
    )
    support_inventory = pd.read_csv(support_inventory_path)
    score_support_inventory = support_inventory[
        support_inventory["in_primary_51_gauge_score_mask"]
    ]
    record_check(
        "all_51_score_gauges_supported_in_final_fit",
        len(score_support_inventory) == 51
        and int(score_support_inventory["final_supported"].sum()) == 51,
        {
            "score_gauges": len(score_support_inventory),
            "final_supported": int(score_support_inventory["final_supported"].sum()),
        },
    )

    for forcing_name in ["observed", "hrrr"]:
        for seed in SEEDS:
            run_name = (
                "stgnn_obs_pre2026_seed"
                + str(seed)
                + "_"
                + forcing_name
            )
            run_path = os.path.join(EXPERIMENT_ROOT, "runs", run_name + ".npz")
            score_directory = os.path.join(EXPERIMENT_ROOT, "scored", run_name)
            record_check("run_exists_" + run_name, os.path.exists(run_path), run_path)
            run_archive = np.load(run_path, allow_pickle=True)
            record_check(
                "run_shape_" + run_name,
                tuple(run_archive["pred_ft"].shape) == (4344, 96, 68, 3)
                and len(run_archive["origins_utc"]) == 4344,
                run_archive["pred_ft"].shape,
            )
            run_metadata = json.loads(str(run_archive["run_meta"]))
            expected_forcing_label = (
                "perfect"
                if forcing_name == "observed"
                else os.path.join(
                    EXPERIMENT_ROOT,
                    "frozen_assets",
                    "hrrr_issue_time_stitch_L2.npz",
                )
            )
            record_check(
                "run_metadata_" + run_name,
                str(run_metadata["system"]) == "System B"
                and str(run_metadata["postproc"]) == "off"
                and str(run_metadata["forcing"]) == expected_forcing_label
                and str(run_metadata["flatline_detection"])
                == "causal trailing 96-step standard deviation",
                "System B, raw output, expected forcing, causal mask",
            )
            for score_filename in [
                "reforecast_2026H1_lead_metrics.csv",
                "reforecast_2026H1_metrics.csv",
                "reforecast_2026H1_origin_window_metrics.csv",
                "reforecast_2026H1_h96_timeseries.csv",
                "score_meta.json",
            ]:
                score_path = os.path.join(score_directory, score_filename)
                record_check(
                    "score_exists_" + run_name + "_" + score_filename,
                    os.path.exists(score_path),
                    score_path,
                )
            lead_score_path = os.path.join(
                score_directory,
                "reforecast_2026H1_lead_metrics.csv",
            )
            lead_score = pd.read_csv(lead_score_path)
            primary_lead_score = lead_score[
                lead_score["node"].isin(score_support_inventory["node"])
            ]
            record_check(
                "score_coverage_" + run_name,
                lead_score["lead_step"].nunique() == 96
                and int(lead_score["lead_step"].min()) == 1
                and int(lead_score["lead_step"].max()) == 96
                and primary_lead_score["node"].nunique() == 51
                and bool(primary_lead_score["RMSE_m"].notna().all()),
                "96 leads and 51 primary gauges with finite RMSE",
            )

    persistence_directory = os.path.join(EXPERIMENT_ROOT, "scored", "persistence")
    record_check(
        "persistence_score_exists",
        os.path.exists(
            os.path.join(persistence_directory, "reforecast_2026H1_metrics.csv")
        ),
        persistence_directory,
    )
    result_summary_path = os.path.join(
        EXPERIMENT_ROOT,
        "results",
        "SYSTEM_B_RESULT_SUMMARY.json",
    )
    record_check(
        "result_summary_exists",
        os.path.exists(result_summary_path),
        result_summary_path,
    )
    with open(result_summary_path, "r", encoding="utf-8") as result_handle:
        result_summary = json.load(result_handle)
    record_check(
        "result_summary_complete",
        str(result_summary["status"]) == "complete"
        and int(result_summary["test_origins"]) == 4344
        and int(result_summary["primary_scoring_gauges"]) == 51,
        result_summary,
    )
    fixed_lead_across_seed_path = os.path.join(
        EXPERIMENT_ROOT,
        "results",
        "fixed_lead_summary_across_seeds.csv",
    )
    fixed_lead_by_seed_path = os.path.join(
        EXPERIMENT_ROOT,
        "results",
        "fixed_lead_summary_by_seed.csv",
    )
    gauge_metric_path = os.path.join(
        EXPERIMENT_ROOT,
        "results",
        "gauge_metrics_all_seeds.csv",
    )
    fixed_lead_across_seed = pd.read_csv(fixed_lead_across_seed_path)
    fixed_lead_by_seed = pd.read_csv(fixed_lead_by_seed_path)
    gauge_metrics = pd.read_csv(gauge_metric_path)
    record_check(
        "aggregate_table_shapes_and_values",
        len(fixed_lead_across_seed) == 18
        and len(fixed_lead_by_seed) == 42
        and len(gauge_metrics) == 357
        and bool(fixed_lead_across_seed["median_RMSE_m_seed_mean"].notna().all())
        and bool(gauge_metrics["RMSE_all_m"].notna().all()),
        {
            "across_seed_rows": len(fixed_lead_across_seed),
            "by_seed_rows": len(fixed_lead_by_seed),
            "gauge_rows": len(gauge_metrics),
        },
    )

report = {
    "experiment": "SYSTEM_B_471_CHRONOLOGICAL_20260913",
    "phase": arguments.phase,
    "status": "passed",
    "check_count": len(checks),
    "checks": checks,
    "created_utc": datetime.now(timezone.utc).isoformat(),
}
report_path = os.path.join(
    EXPERIMENT_ROOT,
    "VALIDATION_REPORT_" + arguments.phase.upper() + ".json",
)
with open(report_path, "w", encoding="utf-8") as report_handle:
    json.dump(report, report_handle, indent=2)

print("[SAVE] Validation report:", report_path)
print("SYSTEM_B_VALIDATION_PASSED phase=" + arguments.phase)
