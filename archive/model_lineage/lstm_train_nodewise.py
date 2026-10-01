"""
Train one shared nodewise unidirectional LSTM baseline for the HESS revision.

This isolated research trainer preserves the frozen graph-ablation data,
splits, preprocessing, target, loss, and evaluation cohort. The LSTM processes
each gauge separately while sharing one set of weights across all 68 gauges.
It never reads from or writes to operational training or inference outputs.
"""

import argparse
import hashlib
import json
import os
import platform
import socket
from datetime import datetime, timezone


parser = argparse.ArgumentParser(
    description="Train one matched 68-node shared nodewise LSTM baseline."
)
parser.add_argument(
    "--experiment-root",
    required=True,
    help="Writable isolated experiment directory on MIKE.",
)
parser.add_argument(
    "--matrix-path",
    required=True,
    help="Frozen engineered feature matrix used by every arm.",
)
parser.add_argument(
    "--stage-to-rain-path",
    required=True,
    help="Frozen stage-to-rain mapping used by every arm.",
)
parser.add_argument(
    "--seed",
    required=True,
    type=int,
    help="Paired random seed.",
)
parser.add_argument(
    "--smoke",
    action="store_true",
    help="Run a small one-epoch execution test instead of a full fit.",
)
args = parser.parse_args()

os.environ["PYTHONHASHSEED"] = str(args.seed)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import pandas as pd
import tensorflow as tf


EXPERIMENT_NAME = "NODEWISE_LSTM_3SEED_20260910"
MODEL_FAMILY = "shared_nodewise_unidirectional_lstm"
EXPECTED_MATRIX_SHA256 = (
    "80ece534226d9be698ae2871f3bc881b028d67ec9d5eaf19e5dccf0639f1fd4d"
)
EXPECTED_STAGE_TO_RAIN_SHA256 = (
    "e79296fa5429a92c32fa00e6ab10e4eacaa159e803ce99a2f1044ac41f40d766"
)
EXPECTED_NODE_ORDER_ASSET_SHA256 = (
    "9e697ec0982f734139cc4477647c532439558c1daa0a08886cc67e7d0f6977eb"
)
EXPECTED_SCORING_MASK_SHA256 = (
    "6b2c73d2c5b3396dae27b5e645110696a3c715d3d7cec3e8325d7cc11be7550f"
)

CADENCE_MINUTES = 15
RAIN_SHORT_STEPS = 6
RAIN_LONG_STEPS = 24
RAIN_SHORT_HOURS = 1.5
RAIN_LONG_HOURS = 6.0
INPUT_STEPS = 288
INPUT_HOURS = 72
DECODER_STEPS = 96
DECODER_HOURS = 24
HIDDEN_UNITS = 64
INPUT_CHANNELS = 9
FORCING_CHANNELS = 9
QUANTILES = [0.5, 0.6, 0.7, 0.8, 0.9]
TENDENCY_LAG_STEPS = 24
API_HALFLIFE_STEPS = {
    "api24": 96.0,
    "api48": 192.0,
    "api72": 288.0,
}
BATCH_SIZE = 16
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 8
LEARNING_RATE = 1.0e-3
GRADIENT_CLIP_NORM = 1.0

TRAIN_ORIGIN_START = pd.Timestamp("2023-01-04 00:00:00", tz="UTC")
TRAIN_ORIGIN_END = pd.Timestamp("2024-12-31 23:00:00", tz="UTC")
VALIDATION_ORIGIN_START = pd.Timestamp("2025-01-05 00:00:00", tz="UTC")
VALIDATION_ORIGIN_END = pd.Timestamp("2025-10-31 23:00:00", tz="UTC")
TRAINING_STATISTICS_END = pd.Timestamp("2024-12-31 23:45:00", tz="UTC")
MATRIX_ANALYSIS_START = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
MATRIX_ANALYSIS_END = pd.Timestamp("2025-11-02 12:00:00", tz="UTC")
EXPECTED_TRAIN_ORIGINS = 17472
EXPECTED_VALIDATION_ORIGINS = 7200
EXPECTED_NODE_COUNT = 68
EXPECTED_PRIMARY_SCORING_COUNT = 30
MINIMUM_STAGE_STATISTICS_VALUES = 96

print("[START] Experiment:", EXPERIMENT_NAME)
print("[START] UTC:", datetime.now(timezone.utc).isoformat())
print("[START] Host:", socket.gethostname())
print("[START] Python:", platform.python_version())
print("[START] TensorFlow:", tf.__version__)
print("[START] NumPy:", np.__version__)
print("[START] pandas:", pd.__version__)
print("[CONFIG] model_family:", MODEL_FAMILY)
print("[CONFIG] seed:", args.seed)
print("[CONFIG] smoke:", args.smoke)
print("[CONFIG] experiment_root:", args.experiment_root)
print("[CONFIG] matrix_path:", args.matrix_path)
print("[CONFIG] stage_to_rain_path:", args.stage_to_rain_path)


def calculate_sha256(path):
    print("[HASH] Calculating SHA256:", path)
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        while True:
            block = input_file.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    result = digest.hexdigest()
    print("[HASH] Result:", result)
    return result


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print("[WRITE] JSON:", path)


def mapped_rain_array(frame, nodes, stage_to_rain, suffix):
    matching_columns = [column for column in frame.columns if column.endswith(suffix)]
    if len(matching_columns) == 0:
        raise RuntimeError(f"No matrix columns end with required suffix {suffix!r}.")

    network_mean = frame[matching_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).mean(axis=1)
    network_mean = network_mean.fillna(0.0).to_numpy(dtype=np.float32)
    output = np.zeros((len(frame), len(nodes)), dtype=np.float32)
    missing_mappings = []

    for node_index, node in enumerate(nodes):
        rain_node = str(stage_to_rain.get(node, ""))
        rain_column = f"{rain_node}{suffix}"
        if rain_column in frame.columns:
            values = pd.to_numeric(frame[rain_column], errors="coerce")
            output[:, node_index] = values.fillna(0.0).to_numpy(dtype=np.float32)
        else:
            output[:, node_index] = network_mean
            missing_mappings.append(
                {
                    "stage_node": node,
                    "requested_rain_column": rain_column,
                }
            )

    print(
        "[RAIN] suffix=",
        suffix,
        " mapped_nodes=",
        len(nodes) - len(missing_mappings),
        " fallback_nodes=",
        len(missing_mappings),
        sep="",
    )
    return output, missing_mappings


def calculate_api(rain_step, half_life_steps):
    decay = float(0.5 ** (1.0 / half_life_steps))
    output = np.zeros_like(rain_step, dtype=np.float32)
    accumulator = np.zeros(rain_step.shape[1], dtype=np.float32)
    for time_index in range(rain_step.shape[0]):
        accumulator = decay * accumulator + rain_step[time_index]
        output[time_index] = accumulator
    return output


def standardize_from_training(array, training_time_mask, label):
    training_values = array[training_time_mask]
    mean_value = float(np.mean(training_values, dtype=np.float64))
    standard_deviation = float(np.std(training_values, dtype=np.float64))
    standard_deviation = standard_deviation + 1.0e-6
    standardized = (array - mean_value) / standard_deviation
    standardized = standardized.astype(np.float32)
    print(
        "[SCALE]",
        label,
        "mean=",
        mean_value,
        "sd=",
        standard_deviation,
    )
    return standardized, mean_value, standard_deviation


matrix_sha256 = calculate_sha256(args.matrix_path)
if matrix_sha256 != EXPECTED_MATRIX_SHA256:
    raise RuntimeError(
        "Frozen matrix SHA256 does not match the inventory source. "
        f"Expected {EXPECTED_MATRIX_SHA256}, found {matrix_sha256}."
    )

stage_to_rain_sha256 = calculate_sha256(args.stage_to_rain_path)
if stage_to_rain_sha256 != EXPECTED_STAGE_TO_RAIN_SHA256:
    raise RuntimeError(
        "Frozen stage-to-rain SHA256 does not match the audited source. "
        f"Expected {EXPECTED_STAGE_TO_RAIN_SHA256}, found {stage_to_rain_sha256}."
    )

assets_directory = os.path.join(args.experiment_root, "assets")
node_order_asset_path = os.path.join(
    assets_directory,
    "node_order_identity_68nodes.npz",
)
scoring_mask_path = os.path.join(
    assets_directory,
    "primary_scoring_mask_pre2026_68nodes.npz",
)

node_order_asset_sha256 = calculate_sha256(node_order_asset_path)
if node_order_asset_sha256 != EXPECTED_NODE_ORDER_ASSET_SHA256:
    raise RuntimeError(
        "Node-order asset SHA256 mismatch. "
        f"Expected {EXPECTED_NODE_ORDER_ASSET_SHA256}, "
        f"found {node_order_asset_sha256}."
    )

scoring_mask_sha256 = calculate_sha256(scoring_mask_path)
if scoring_mask_sha256 != EXPECTED_SCORING_MASK_SHA256:
    raise RuntimeError(
        "Primary scoring mask SHA256 mismatch. "
        f"Expected {EXPECTED_SCORING_MASK_SHA256}, found {scoring_mask_sha256}."
    )

node_order_archive = np.load(node_order_asset_path, allow_pickle=True)
nodes = [str(value) for value in node_order_archive["nodes"]]
node_order_identity = node_order_archive["A_norm"].astype(np.float32)
node_count = len(nodes)
if node_count != EXPECTED_NODE_COUNT:
    raise RuntimeError(
        f"Expected {EXPECTED_NODE_COUNT} nodes, found {node_count}."
    )
if node_order_identity.shape != (EXPECTED_NODE_COUNT, EXPECTED_NODE_COUNT):
    raise RuntimeError(
        "Unexpected node-order identity shape "
        f"{node_order_identity.shape}; expected (68, 68)."
    )
if not np.array_equal(
    node_order_identity,
    np.eye(EXPECTED_NODE_COUNT, dtype=np.float32),
):
    raise RuntimeError("The node-order asset is not an exact identity matrix.")

scoring_archive = np.load(scoring_mask_path, allow_pickle=True)
scoring_nodes = [str(value) for value in scoring_archive["nodes"]]
primary_scoring_mask = scoring_archive["primary_scoring_mask"].astype(bool)
if scoring_nodes != nodes:
    raise RuntimeError("Graph and primary-scoring-mask node orders differ.")
if int(primary_scoring_mask.sum()) != EXPECTED_PRIMARY_SCORING_COUNT:
    raise RuntimeError(
        "Expected 30 primary scoring gauges, found "
        f"{int(primary_scoring_mask.sum())}."
    )
print("[MODEL] nodes:", node_count)
print("[MODEL] cross-gauge messages: NEVER")
print("[MODEL] weights shared across gauges: YES")
print("[SCORE MASK] primary gauges:", int(primary_scoring_mask.sum()))
print("[SCORE MASK] training-loss use: NEVER")

run_kind = "smoke" if args.smoke else "full"
run_name = f"nodewise_lstm_seed{args.seed}"
run_directory = os.path.join(args.experiment_root, "runs", run_kind, run_name)
os.makedirs(run_directory, exist_ok=False)
print("[OUTPUT] run_directory:", run_directory)

print("[DATA] Loading frozen matrix.")
full_frame = pd.read_pickle(args.matrix_path)
full_frame = full_frame.sort_index()
if full_frame.index.tz is None:
    full_frame.index = full_frame.index.tz_localize("UTC")
else:
    full_frame.index = full_frame.index.tz_convert("UTC")
if int(full_frame.index.duplicated().sum()) != 0:
    raise RuntimeError("Frozen matrix has duplicate timestamps.")

frame = full_frame.loc[MATRIX_ANALYSIS_START:MATRIX_ANALYSIS_END].copy()
expected_grid = pd.date_range(
    MATRIX_ANALYSIS_START,
    MATRIX_ANALYSIS_END,
    freq=f"{CADENCE_MINUTES}min",
)
if not frame.index.equals(expected_grid):
    raise RuntimeError(
        "Frozen matrix is not the expected complete 15-minute grid over the analysis range."
    )
print("[DATA] rows:", len(frame))
print("[DATA] columns:", frame.shape[1])
print("[DATA] first timestamp:", frame.index.min())
print("[DATA] last timestamp:", frame.index.max())

rain_base_columns = [
    column
    for column in frame.columns
    if column.endswith("_rain_in")
]
if len(rain_base_columns) == 0:
    raise RuntimeError("No base 15-minute rainfall columns were found.")

rain_invariant_rows = []
for base_column in rain_base_columns:
    short_column = f"{base_column}_accum_6h"
    long_column = f"{base_column}_accum_24h"
    if short_column not in frame.columns:
        raise RuntimeError(f"Missing required rainfall column {short_column}.")
    if long_column not in frame.columns:
        raise RuntimeError(f"Missing required rainfall column {long_column}.")

    base_values = pd.to_numeric(frame[base_column], errors="coerce")
    expected_short = base_values.rolling(window=RAIN_SHORT_STEPS).sum()
    expected_long = base_values.rolling(window=RAIN_LONG_STEPS).sum()
    actual_short = pd.to_numeric(frame[short_column], errors="coerce")
    actual_long = pd.to_numeric(frame[long_column], errors="coerce")

    short_finite = np.isfinite(expected_short.to_numpy()) & np.isfinite(
        actual_short.to_numpy()
    )
    long_finite = np.isfinite(expected_long.to_numpy()) & np.isfinite(
        actual_long.to_numpy()
    )
    if not np.array_equal(
        expected_short.isna().to_numpy(),
        actual_short.isna().to_numpy(),
    ):
        raise RuntimeError(f"NaN pattern mismatch for {short_column}.")
    if not np.array_equal(
        expected_long.isna().to_numpy(),
        actual_long.isna().to_numpy(),
    ):
        raise RuntimeError(f"NaN pattern mismatch for {long_column}.")

    short_difference = 0.0
    long_difference = 0.0
    if bool(short_finite.any()):
        short_difference = float(
            np.max(
                np.abs(
                    expected_short.to_numpy()[short_finite]
                    - actual_short.to_numpy()[short_finite]
                )
            )
        )
    if bool(long_finite.any()):
        long_difference = float(
            np.max(
                np.abs(
                    expected_long.to_numpy()[long_finite]
                    - actual_long.to_numpy()[long_finite]
                )
            )
        )
    if short_difference > 1.0e-7:
        raise RuntimeError(
            f"{short_column} is not the required 6-step rolling sum."
        )
    if long_difference > 1.0e-7:
        raise RuntimeError(
            f"{long_column} is not the required 24-step rolling sum."
        )

    rain_invariant_rows.append(
        {
            "base_column": base_column,
            "rain6_steps": RAIN_SHORT_STEPS,
            "rain6_hours": RAIN_SHORT_HOURS,
            "rain6_max_abs_difference": short_difference,
            "rain24_steps": RAIN_LONG_STEPS,
            "rain24_hours": RAIN_LONG_HOURS,
            "rain24_max_abs_difference": long_difference,
        }
    )

rain_invariant_frame = pd.DataFrame(rain_invariant_rows)
rain_invariant_path = os.path.join(run_directory, "rainfall_invariant_check.csv")
rain_invariant_frame.to_csv(rain_invariant_path, index=False)
print("[RAIN INVARIANT] base rainfall columns checked:", len(rain_base_columns))
print("[RAIN INVARIANT] rain6 steps:", RAIN_SHORT_STEPS)
print("[RAIN INVARIANT] rain6 physical hours:", RAIN_SHORT_HOURS)
print("[RAIN INVARIANT] rain24 steps:", RAIN_LONG_STEPS)
print("[RAIN INVARIANT] rain24 physical hours:", RAIN_LONG_HOURS)
print("[RAIN INVARIANT] decoder steps:", DECODER_STEPS)
print("[RAIN INVARIANT] decoder physical hours:", DECODER_HOURS)

stage_to_rain_frame = pd.read_csv(args.stage_to_rain_path)
required_mapping_columns = {"stage", "nearest_rain"}
if not required_mapping_columns.issubset(stage_to_rain_frame.columns):
    raise RuntimeError(
        "Stage-to-rain file does not contain stage and nearest_rain columns."
    )
stage_to_rain = stage_to_rain_frame.set_index("stage")["nearest_rain"].to_dict()

time_count = len(frame)
stage_values = np.full((time_count, node_count), np.nan, dtype=np.float32)
target_mask = np.zeros((time_count, node_count), dtype=bool)

for node_index, node in enumerate(nodes):
    stage_column = f"{node}_stage_ft"
    if stage_column not in frame.columns:
        print("[STAGE] Missing column:", stage_column)
        continue
    series = pd.to_numeric(frame[stage_column], errors="coerce")
    rolling_standard_deviation = series.rolling(
        window=96,
        center=True,
        min_periods=96,
    ).std()
    series = series.mask(rolling_standard_deviation < 1.0e-6)
    stage_values[:, node_index] = series.to_numpy(dtype=np.float32)
    target_mask[:, node_index] = np.isfinite(series.to_numpy())

training_time_mask = frame.index <= TRAINING_STATISTICS_END
stage_mean = np.zeros(node_count, dtype=np.float32)
stage_standard_deviation = np.ones(node_count, dtype=np.float32)
no_training_stage_mask = np.zeros(node_count, dtype=bool)
training_stage_value_counts = np.zeros(node_count, dtype=np.int64)

for node_index, node in enumerate(nodes):
    node_training_mask = training_time_mask & target_mask[:, node_index]
    values = stage_values[node_training_mask, node_index]
    values = values[np.isfinite(values)]
    training_stage_value_counts[node_index] = len(values)
    if len(values) < MINIMUM_STAGE_STATISTICS_VALUES:
        no_training_stage_mask[node_index] = True
        target_mask[:, node_index] = False
        print(
            "[STAGE MASK] node=",
            node,
            " training_values=",
            len(values),
            " action=mask_all_stage_inputs_and_targets",
            sep="",
        )
        continue
    stage_mean[node_index] = np.mean(values, dtype=np.float64)
    stage_standard_deviation[node_index] = (
        np.std(values, dtype=np.float64) + 1.0e-6
    )

stage_standardized = (
    stage_values - stage_mean[None, :]
) / stage_standard_deviation[None, :]
stage_standardized = np.nan_to_num(
    stage_standardized,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
).astype(np.float32)
stage_standardized[:, no_training_stage_mask] = 0.0
print("[STAGE] no-training-support nodes:", int(no_training_stage_mask.sum()))
print("[STAGE] target-mask fraction:", float(target_mask.mean()))
unsupported_primary_nodes = [
    node
    for node, is_primary, no_support in zip(
        nodes,
        primary_scoring_mask,
        no_training_stage_mask,
    )
    if bool(is_primary) and bool(no_support)
]
if len(unsupported_primary_nodes) != 0:
    raise RuntimeError(
        "Primary scoring gauges without training-stage support: "
        f"{unsupported_primary_nodes}"
    )
print("[STAGE] unsupported primary gauges: 0")

rain_short, missing_short_mappings = mapped_rain_array(
    frame,
    nodes,
    stage_to_rain,
    "_rain_in_accum_6h",
)
rain_long, missing_long_mappings = mapped_rain_array(
    frame,
    nodes,
    stage_to_rain,
    "_rain_in_accum_24h",
)
rain_step, missing_step_mappings = mapped_rain_array(
    frame,
    nodes,
    stage_to_rain,
    "_rain_in",
)

api24 = calculate_api(rain_step, API_HALFLIFE_STEPS["api24"])
api48 = calculate_api(rain_step, API_HALFLIFE_STEPS["api48"])
api72 = calculate_api(rain_step, API_HALFLIFE_STEPS["api72"])
rain_event_interaction = rain_short * api48
rain_acceleration = np.zeros_like(rain_step, dtype=np.float32)
rain_acceleration[4:] = rain_step[4:] - rain_step[:-4]
stage_tendency = np.zeros_like(stage_standardized, dtype=np.float32)
stage_tendency[TENDENCY_LAG_STEPS:] = (
    stage_standardized[TENDENCY_LAG_STEPS:]
    - stage_standardized[:-TENDENCY_LAG_STEPS]
)

rain_short_z, rain_short_mean, rain_short_sd = standardize_from_training(
    rain_short,
    training_time_mask,
    "rain6_step_1p5h",
)
rain_long_z, rain_long_mean, rain_long_sd = standardize_from_training(
    rain_long,
    training_time_mask,
    "rain24_step_6h",
)
api24_z, api24_mean, api24_sd = standardize_from_training(
    api24,
    training_time_mask,
    "api24h",
)
api48_z, api48_mean, api48_sd = standardize_from_training(
    api48,
    training_time_mask,
    "api48h",
)
api72_z, api72_mean, api72_sd = standardize_from_training(
    api72,
    training_time_mask,
    "api72h",
)
rain_event_z, rain_event_mean, rain_event_sd = standardize_from_training(
    rain_event_interaction,
    training_time_mask,
    "rain_event_interaction",
)
rain_acceleration_z, rain_acceleration_mean, rain_acceleration_sd = (
    standardize_from_training(
        rain_acceleration,
        training_time_mask,
        "rain_acceleration",
    )
)
stage_tendency_z, stage_tendency_mean, stage_tendency_sd = (
    standardize_from_training(
        stage_tendency,
        training_time_mask,
        "stage_tendency",
    )
)

hour_decimal = (
    frame.index.hour.to_numpy(dtype=np.float64)
    + frame.index.minute.to_numpy(dtype=np.float64) / 60.0
)
time_angle = 2.0 * np.pi * hour_decimal / 24.0
time_sine = np.sin(time_angle).astype(np.float32)
time_cosine = np.cos(time_angle).astype(np.float32)

train_origins = pd.date_range(
    TRAIN_ORIGIN_START,
    TRAIN_ORIGIN_END,
    freq="1h",
)
validation_origins = pd.date_range(
    VALIDATION_ORIGIN_START,
    VALIDATION_ORIGIN_END,
    freq="1h",
)
if len(train_origins) != EXPECTED_TRAIN_ORIGINS:
    raise RuntimeError(
        f"Expected {EXPECTED_TRAIN_ORIGINS} training origins, found {len(train_origins)}."
    )
if len(validation_origins) != EXPECTED_VALIDATION_ORIGINS:
    raise RuntimeError(
        "Expected "
        f"{EXPECTED_VALIDATION_ORIGINS} validation origins, "
        f"found {len(validation_origins)}."
    )

train_origin_positions = frame.index.get_indexer(train_origins)
validation_origin_positions = frame.index.get_indexer(validation_origins)
if bool(np.any(train_origin_positions < 0)):
    raise RuntimeError("At least one training origin is absent from the 15-minute grid.")
if bool(np.any(validation_origin_positions < 0)):
    raise RuntimeError("At least one validation origin is absent from the 15-minute grid.")

train_starts = train_origin_positions - INPUT_STEPS + 1
validation_starts = validation_origin_positions - INPUT_STEPS + 1
if int(train_starts.min()) < 0:
    raise RuntimeError("The first training origin does not have 288 history steps.")
if int(validation_starts.min()) < 0:
    raise RuntimeError("The first validation origin does not have 288 history steps.")
if int(train_origin_positions.max() + DECODER_STEPS) >= len(frame):
    raise RuntimeError("The last training origin does not have 96 target steps.")
if int(validation_origin_positions.max() + DECODER_STEPS) >= len(frame):
    raise RuntimeError("The last validation origin does not have 96 target steps.")

train_last_target = frame.index[int(train_origin_positions[-1] + DECODER_STEPS)]
validation_first_history = frame.index[int(validation_starts[0])]
if train_last_target >= validation_first_history:
    raise RuntimeError(
        "Training target support overlaps validation history support."
    )
support_gap_minutes = (
    validation_first_history - train_last_target
).total_seconds() / 60.0
print("[SPLIT] training origins:", len(train_origins))
print("[SPLIT] validation origins:", len(validation_origins))
print("[SPLIT] last training target:", train_last_target)
print("[SPLIT] first validation history:", validation_first_history)
print("[SPLIT] positive support gap minutes:", support_gap_minutes)

train_origin_frame = pd.DataFrame(
    {
        "origin_utc": train_origins.astype(str),
        "history_start_utc": frame.index[train_starts].astype(str),
        "target_end_utc": frame.index[
            train_origin_positions + DECODER_STEPS
        ].astype(str),
    }
)
validation_origin_frame = pd.DataFrame(
    {
        "origin_utc": validation_origins.astype(str),
        "history_start_utc": frame.index[validation_starts].astype(str),
        "target_end_utc": frame.index[
            validation_origin_positions + DECODER_STEPS
        ].astype(str),
    }
)
train_origin_path = os.path.join(run_directory, "training_origins.csv")
validation_origin_path = os.path.join(run_directory, "validation_origins.csv")
train_origin_frame.to_csv(train_origin_path, index=False)
validation_origin_frame.to_csv(validation_origin_path, index=False)
print("[WRITE] Training origins:", train_origin_path)
print("[WRITE] Validation origins:", validation_origin_path)

if args.smoke:
    train_starts_for_fit = train_starts[:8]
    validation_starts_for_fit = validation_starts[:4]
    batch_size = 2
    epochs = 1
else:
    train_starts_for_fit = train_starts
    validation_starts_for_fit = validation_starts
    batch_size = BATCH_SIZE
    epochs = MAX_EPOCHS

preprocessing_digest = hashlib.sha256()
for array in [
    np.asarray(nodes, dtype="S8"),
    train_starts,
    validation_starts,
    stage_mean,
    stage_standard_deviation,
    training_stage_value_counts,
    no_training_stage_mask,
    np.asarray(
        [
            rain_short_mean,
            rain_short_sd,
            rain_long_mean,
            rain_long_sd,
            api24_mean,
            api24_sd,
            api48_mean,
            api48_sd,
            api72_mean,
            api72_sd,
            rain_event_mean,
            rain_event_sd,
            rain_acceleration_mean,
            rain_acceleration_sd,
            stage_tendency_mean,
            stage_tendency_sd,
        ],
        dtype=np.float64,
    ),
]:
    preprocessing_digest.update(np.ascontiguousarray(array).tobytes())
preprocessing_sha256 = preprocessing_digest.hexdigest()
print("[PROTOCOL] preprocessing SHA256:", preprocessing_sha256)

metadata = {
    "experiment": EXPERIMENT_NAME,
    "research_only_not_production": True,
    "model_family": MODEL_FAMILY,
    "cell_type": "LSTM",
    "encoder_direction": "forward_only",
    "recurrent_direction": "forward_only",
    "parameter_sharing": "one_encoder_decoder_and_output_head_shared_across_all_nodes",
    "cross_gauge_messages": False,
    "self_message_path": True,
    "seed": args.seed,
    "smoke": args.smoke,
    "nodes": nodes,
    "node_count": node_count,
    "primary_scoring_nodes": [
        node
        for node, is_primary in zip(nodes, primary_scoring_mask)
        if bool(is_primary)
    ],
    "primary_scoring_count": int(primary_scoring_mask.sum()),
    "primary_scoring_mask_role": "evaluation_only_not_training_loss",
    "no_training_stage_nodes": [
        node
        for node, no_support in zip(nodes, no_training_stage_mask)
        if bool(no_support)
    ],
    "training_stage_value_counts": training_stage_value_counts.tolist(),
    "stage_mean_ft": stage_mean.tolist(),
    "stage_standard_deviation_ft": stage_standard_deviation.tolist(),
    "scalers": {
        "rain6_step_1p5h": [rain_short_mean, rain_short_sd],
        "rain24_step_6h": [rain_long_mean, rain_long_sd],
        "api24": [api24_mean, api24_sd],
        "api48": [api48_mean, api48_sd],
        "api72": [api72_mean, api72_sd],
        "rain_event_interaction": [rain_event_mean, rain_event_sd],
        "rain_acceleration": [rain_acceleration_mean, rain_acceleration_sd],
        "stage_tendency": [stage_tendency_mean, stage_tendency_sd],
    },
    "rainfall_semantics": {
        "cadence_minutes": CADENCE_MINUTES,
        "rain6_steps": RAIN_SHORT_STEPS,
        "rain6_hours": RAIN_SHORT_HOURS,
        "rain24_steps": RAIN_LONG_STEPS,
        "rain24_hours": RAIN_LONG_HOURS,
        "do_not_infer_physical_duration_from_legacy_column_suffix": True,
    },
    "input_steps": INPUT_STEPS,
    "input_hours": INPUT_HOURS,
    "decoder_steps": DECODER_STEPS,
    "decoder_hours": DECODER_HOURS,
    "input_channels": INPUT_CHANNELS,
    "forcing_channels": FORCING_CHANNELS,
    "hidden_units": HIDDEN_UNITS,
    "quantiles": QUANTILES,
    "tendency_lag_steps": TENDENCY_LAG_STEPS,
    "api_halflife_steps": API_HALFLIFE_STEPS,
    "training_origin_start_utc": str(TRAIN_ORIGIN_START),
    "training_origin_end_utc": str(TRAIN_ORIGIN_END),
    "training_origin_count": len(train_origins),
    "validation_origin_start_utc": str(VALIDATION_ORIGIN_START),
    "validation_origin_end_utc": str(VALIDATION_ORIGIN_END),
    "validation_origin_count": len(validation_origins),
    "training_statistics_end_utc": str(TRAINING_STATISTICS_END),
    "support_gap_minutes": support_gap_minutes,
    "batch_size": batch_size,
    "maximum_epochs": epochs,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "optimizer": "Adam",
    "learning_rate": LEARNING_RATE,
    "gradient_clip_norm": GRADIENT_CLIP_NORM,
    "loss": "mean_multi_quantile_pinball_with_DQ_target_weights",
    "matrix_path": args.matrix_path,
    "matrix_sha256": matrix_sha256,
    "stage_to_rain_path": args.stage_to_rain_path,
    "stage_to_rain_sha256": stage_to_rain_sha256,
    "node_order_asset_path": node_order_asset_path,
    "node_order_asset_sha256": node_order_asset_sha256,
    "scoring_mask_path": scoring_mask_path,
    "scoring_mask_sha256": scoring_mask_sha256,
    "preprocessing_sha256": preprocessing_sha256,
    "missing_rain_mappings": {
        "rain6": missing_short_mappings,
        "rain24": missing_long_mappings,
        "rain_step": missing_step_mappings,
    },
}
metadata_path = os.path.join(run_directory, "lstm_meta.json")

tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(args.seed)
try:
    tf.config.experimental.enable_op_determinism()
    print("[DETERMINISM] TensorFlow deterministic operations enabled.")
except Exception as error:
    raise RuntimeError(
        "TensorFlow deterministic operations could not be enabled."
    ) from error

tensor_map = {
    "stage": tf.constant(stage_standardized),
    "rain_short": tf.constant(rain_short_z),
    "rain_long": tf.constant(rain_long_z),
    "api24": tf.constant(api24_z),
    "api48": tf.constant(api48_z),
    "api72": tf.constant(api72_z),
    "rain_event": tf.constant(rain_event_z),
    "rain_acceleration": tf.constant(rain_acceleration_z),
    "stage_tendency": tf.constant(stage_tendency_z),
    "time_sine": tf.constant(time_sine),
    "time_cosine": tf.constant(time_cosine),
}
target_mask_tensor = tf.constant(target_mask.astype(np.float32))


def make_example(start_index):
    history = tf.stack(
        [
            tensor_map["stage"][start_index:start_index + INPUT_STEPS],
            tensor_map["rain_short"][start_index:start_index + INPUT_STEPS],
            tensor_map["rain_long"][start_index:start_index + INPUT_STEPS],
            tensor_map["api24"][start_index:start_index + INPUT_STEPS],
            tensor_map["api48"][start_index:start_index + INPUT_STEPS],
            tensor_map["api72"][start_index:start_index + INPUT_STEPS],
            tensor_map["stage_tendency"][start_index:start_index + INPUT_STEPS],
            tensor_map["rain_event"][start_index:start_index + INPUT_STEPS],
            tensor_map["rain_acceleration"][start_index:start_index + INPUT_STEPS],
        ],
        axis=-1,
    )
    forecast_start = start_index + INPUT_STEPS
    origin_index = forecast_start - 1
    forecast_end = forecast_start + DECODER_STEPS
    forecast_time_sine = tf.tile(
        tensor_map["time_sine"][forecast_start:forecast_end, None],
        [1, node_count],
    )
    forecast_time_cosine = tf.tile(
        tensor_map["time_cosine"][forecast_start:forecast_end, None],
        [1, node_count],
    )
    forcing = tf.stack(
        [
            tensor_map["rain_short"][forecast_start:forecast_end],
            tensor_map["rain_long"][forecast_start:forecast_end],
            tensor_map["api24"][forecast_start:forecast_end],
            tensor_map["api48"][forecast_start:forecast_end],
            tensor_map["api72"][forecast_start:forecast_end],
            tensor_map["rain_event"][forecast_start:forecast_end],
            tensor_map["rain_acceleration"][forecast_start:forecast_end],
            forecast_time_sine,
            forecast_time_cosine,
        ],
        axis=-1,
    )
    origin_stage = tensor_map["stage"][origin_index]
    target = (
        tensor_map["stage"][forecast_start:forecast_end]
        - origin_stage[None, :]
    )
    sample_weight = (
        target_mask_tensor[forecast_start:forecast_end]
        * target_mask_tensor[origin_index][None, :]
    )
    history.set_shape([INPUT_STEPS, node_count, INPUT_CHANNELS])
    forcing.set_shape([DECODER_STEPS, node_count, FORCING_CHANNELS])
    origin_stage.set_shape([node_count])
    target.set_shape([DECODER_STEPS, node_count])
    sample_weight.set_shape([DECODER_STEPS, node_count])
    return (history, forcing, origin_stage), target, sample_weight


dataset_options = tf.data.Options()
dataset_options.experimental_deterministic = True

training_dataset = tf.data.Dataset.from_tensor_slices(train_starts_for_fit)
training_dataset = training_dataset.shuffle(
    buffer_size=len(train_starts_for_fit),
    seed=args.seed,
    reshuffle_each_iteration=True,
)
training_dataset = training_dataset.map(
    make_example,
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=True,
)
training_dataset = training_dataset.batch(
    batch_size,
    drop_remainder=False,
)
training_dataset = training_dataset.prefetch(2)
training_dataset = training_dataset.with_options(dataset_options)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    validation_starts_for_fit
)
validation_dataset = validation_dataset.map(
    make_example,
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=True,
)
validation_dataset = validation_dataset.batch(
    batch_size,
    drop_remainder=False,
)
validation_dataset = validation_dataset.prefetch(2)
validation_dataset = validation_dataset.with_options(dataset_options)

quantile_tensor = tf.constant(QUANTILES, dtype=tf.float32)


def multi_quantile_pinball_loss(observed, predicted):
    error = observed[..., None] - predicted
    pinball = tf.maximum(
        quantile_tensor * error,
        (quantile_tensor - 1.0) * error,
    )
    return tf.reduce_mean(pinball, axis=-1)


class NodewiseLSTMQuantile(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = tf.keras.layers.LSTM(
            HIDDEN_UNITS,
            return_state=True,
        )
        self.decoder_cell = tf.keras.layers.LSTMCell(HIDDEN_UNITS)
        self.output_layer = tf.keras.layers.Dense(len(QUANTILES))

    def call(self, inputs, training=False):
        history, forcing, origin_stage = inputs
        del origin_stage
        batch_count = tf.shape(history)[0]
        transposed_history = tf.transpose(history, [0, 2, 1, 3])
        reshaped_history = tf.reshape(
            transposed_history,
            [batch_count * node_count, INPUT_STEPS, INPUT_CHANNELS],
        )
        encoded_output, hidden_state, cell_state = self.encoder(
            reshaped_history,
            training=training,
        )
        del encoded_output
        outputs = tf.TensorArray(tf.float32, size=DECODER_STEPS)
        for decoder_index in range(DECODER_STEPS):
            self_message = tf.reshape(
                hidden_state,
                [batch_count, node_count, HIDDEN_UNITS],
            )
            decoder_features = tf.concat(
                [forcing[:, decoder_index], self_message],
                axis=-1,
            )
            decoder_input = tf.reshape(
                decoder_features,
                [
                    batch_count * node_count,
                    FORCING_CHANNELS + HIDDEN_UNITS,
                ],
            )
            decoded_hidden, decoder_state = self.decoder_cell(
                decoder_input,
                [hidden_state, cell_state],
                training=training,
            )
            hidden_state = decoder_state[0]
            cell_state = decoder_state[1]
            nodewise_hidden = tf.reshape(
                decoded_hidden,
                [batch_count, node_count, HIDDEN_UNITS],
            )
            quantile_output = self.output_layer(
                nodewise_hidden,
                training=training,
            )
            outputs = outputs.write(decoder_index, quantile_output)
        stacked_outputs = outputs.stack()
        return tf.transpose(stacked_outputs, [1, 0, 2, 3])


model = NodewiseLSTMQuantile()
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=GRADIENT_CLIP_NORM,
    ),
    loss=multi_quantile_pinball_loss,
)

dummy_history = tf.zeros(
    [1, INPUT_STEPS, node_count, INPUT_CHANNELS],
    dtype=tf.float32,
)
dummy_forcing = tf.zeros(
    [1, DECODER_STEPS, node_count, FORCING_CHANNELS],
    dtype=tf.float32,
)
dummy_origin_stage = tf.zeros([1, node_count], dtype=tf.float32)
dummy_output = model(
    (dummy_history, dummy_forcing, dummy_origin_stage),
    training=False,
)
expected_output_shape = (1, DECODER_STEPS, node_count, len(QUANTILES))
if tuple(dummy_output.shape) != expected_output_shape:
    raise RuntimeError(
        f"Model output shape {tuple(dummy_output.shape)} does not equal "
        f"{expected_output_shape}."
    )
print("[MODEL] output shape:", tuple(dummy_output.shape))
model_parameter_count = int(model.count_params())
trainer_source_path = os.path.abspath(__file__)
trainer_source_sha256 = calculate_sha256(trainer_source_path)
metadata["model_parameter_count"] = model_parameter_count
metadata["trainer_source_path"] = trainer_source_path
metadata["trainer_source_sha256"] = trainer_source_sha256
metadata["unsupported_primary_nodes"] = unsupported_primary_nodes
print("[MODEL] parameter count:", model_parameter_count)

perturbed_history = tf.tensor_scatter_nd_update(
    dummy_history,
    indices=[[0, 0, 0, 0]],
    updates=[1.0],
)
perturbed_forcing = tf.tensor_scatter_nd_update(
    dummy_forcing,
    indices=[[0, 0, 0, 0]],
    updates=[1.0],
)
perturbed_output = model(
    (perturbed_history, perturbed_forcing, dummy_origin_stage),
    training=False,
)
other_node_difference = tf.reduce_max(
    tf.abs(perturbed_output[:, :, 1:, :] - dummy_output[:, :, 1:, :])
)
other_node_difference_value = float(other_node_difference.numpy())
if other_node_difference_value > 1.0e-7:
    raise RuntimeError(
        "Cross-node perturbation changed another gauge by "
        f"{other_node_difference_value}."
    )
metadata["cross_node_perturbation_max_abs_difference"] = (
    other_node_difference_value
)
print(
    "[MODEL AUDIT] cross-node perturbation maximum difference:",
    other_node_difference_value,
)

node_permutation = np.arange(node_count - 1, -1, -1, dtype=np.int32)
permuted_history = tf.gather(perturbed_history, node_permutation, axis=2)
permuted_forcing = tf.gather(perturbed_forcing, node_permutation, axis=2)
permuted_origin_stage = tf.gather(
    dummy_origin_stage,
    node_permutation,
    axis=1,
)
permuted_output = model(
    (permuted_history, permuted_forcing, permuted_origin_stage),
    training=False,
)
expected_permuted_output = tf.gather(
    perturbed_output,
    node_permutation,
    axis=2,
)
permutation_difference = tf.reduce_max(
    tf.abs(permuted_output - expected_permuted_output)
)
permutation_difference_value = float(permutation_difference.numpy())
if permutation_difference_value > 1.0e-7:
    raise RuntimeError(
        "Node permutation equivariance failed with maximum difference "
        f"{permutation_difference_value}."
    )
metadata["node_permutation_max_abs_difference"] = (
    permutation_difference_value
)
print(
    "[MODEL AUDIT] node permutation maximum difference:",
    permutation_difference_value,
)
write_json(metadata_path, metadata)


def print_model_summary(line, **kwargs):
    del kwargs
    print("[MODEL]", line)


model.summary(print_fn=print_model_summary)

history_path = os.path.join(run_directory, "training_history.csv")
callbacks = [
    tf.keras.callbacks.CSVLogger(history_path),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.TerminateOnNaN(),
]

print("[FIT] Starting training.")
fit_start_utc = datetime.now(timezone.utc)
fit_history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=callbacks,
    verbose=2,
)
fit_end_utc = datetime.now(timezone.utc)
print("[FIT] Completed training.")

weights_path = os.path.join(run_directory, "lstm.weights.h5")
model.save_weights(weights_path)
print("[SAVE] Weights:", weights_path)

validation_losses = fit_history.history.get("val_loss", [])
if len(validation_losses) == 0:
    raise RuntimeError("Training history does not contain validation loss.")
best_epoch_index = int(np.argmin(np.asarray(validation_losses, dtype=float)))
best_epoch_number = best_epoch_index + 1
best_validation_loss = float(validation_losses[best_epoch_index])

completion = {
    "status": "complete",
    "experiment": EXPERIMENT_NAME,
    "run_kind": run_kind,
    "run_name": run_name,
    "model_family": MODEL_FAMILY,
    "model_parameter_count": model_parameter_count,
    "seed": args.seed,
    "fit_start_utc": fit_start_utc.isoformat(),
    "fit_end_utc": fit_end_utc.isoformat(),
    "fit_elapsed_seconds": (fit_end_utc - fit_start_utc).total_seconds(),
    "epochs_completed": len(validation_losses),
    "best_epoch": best_epoch_number,
    "best_validation_loss": best_validation_loss,
    "weights_path": weights_path,
    "weights_sha256": calculate_sha256(weights_path),
    "metadata_path": metadata_path,
    "metadata_sha256": calculate_sha256(metadata_path),
    "training_history_path": history_path,
    "training_history_sha256": calculate_sha256(history_path),
    "rainfall_invariant_path": rain_invariant_path,
    "rainfall_invariant_sha256": calculate_sha256(rain_invariant_path),
    "preprocessing_sha256": preprocessing_sha256,
    "trainer_source_sha256": trainer_source_sha256,
}
completion_path = os.path.join(run_directory, "RUN_COMPLETE.json")
write_json(completion_path, completion)

print("[COMPLETE] run:", run_name)
print("[COMPLETE] best epoch:", best_epoch_number)
print("[COMPLETE] best validation loss:", best_validation_loss)
print("[COMPLETE] UTC:", datetime.now(timezone.utc).isoformat())
