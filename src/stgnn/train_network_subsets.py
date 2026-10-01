"""Train System B with chronological development and a full pre-2026 final fit.

System B preserves the paper's 68-node, nine-channel ST-GNN recipe. It repairs the
old reforecast trainer in two ways:

1. Development training and validation are chronological, not a random split of
   overlapping windows.
2. The final model is rebuilt from a random initialization on every eligible
   pre-2026 window, using a fixed epoch count selected during development.

The development observation graph is discovered from 2023-2024 stage data. The
final observation graph is the 471-edge graph discovered from all data available
through 31 December 2025. Test data are never read by this trainer.

This is a research experiment. It never uploads weights or modifies production.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import numpy.core as numpy_core
    import numpy.core.multiarray as numpy_multiarray
    import numpy.core.numeric as numpy_numeric

    numpy_aliases = [
        ("numpy._core", numpy_core),
        ("numpy._core.numeric", numpy_numeric),
        ("numpy._core.multiarray", numpy_multiarray),
    ]
    for alias_name, alias_module in numpy_aliases:
        sys.modules.setdefault(alias_name, alias_module)
except Exception as numpy_alias_exception:
    print("[WARN] NumPy compatibility aliases were not installed:", numpy_alias_exception)

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
sys.path.insert(0, SCRIPT_DIRECTORY)

from stgnn_models import build_model

parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["development", "final"], required=True)
parser.add_argument("--model-kind", choices=["stgnn", "lstm"], default="stgnn")
parser.add_argument("--graph-key", default="obs_pre2026")
parser.add_argument("--seed", type=int, required=True)
parser.add_argument(
    "--training-pickle",
    default=(
        "project/hpc/Training/GNN/Experiments/"
        "REFORECAST_GRAPH_RANKING_2026H1/graphs/training_matrix_through_20251231.pkl"
    ),
)
parser.add_argument(
    "--stage-to-rain",
    default=os.path.join(EXPERIMENT_ROOT, "frozen_assets", "stage_to_rain_gauge.csv"),
)
parser.add_argument(
    "--development-graph",
    default=os.path.join(EXPERIMENT_ROOT, "frozen_assets", "graph_obs_development_2023_2024.npz"),
)
parser.add_argument(
    "--final-graph",
    default=os.path.join(EXPERIMENT_ROOT, "frozen_assets", "graph_obs_pre2026_471edges.npz"),
)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--maximum-development-epochs", type=int, default=40)
parser.add_argument(
    "--models-root",
    default=os.path.join(EXPERIMENT_ROOT, "models"),
)
parser.add_argument(
    "--experiment-name",
    default="SYSTEM_B_471_CHRONOLOGICAL_20260913",
)
parser.add_argument("--expected-node-count", type=int, default=68)
parser.add_argument("--smoke", action="store_true")
arguments = parser.parse_args()

STAGE = str(arguments.stage)
MODEL_KIND = str(arguments.model_kind)
GRAPH_KEY = str(arguments.graph_key)
SEED = int(arguments.seed)
MODELS_ROOT = os.path.abspath(arguments.models_root)
EXPERIMENT_NAME = str(arguments.experiment_name)
EXPECTED_NODE_COUNT = int(arguments.expected_node_count)

if EXPECTED_NODE_COUNT <= 0:
    raise ValueError("The expected node count must be positive.")

QUANTILES = [0.5, 0.6, 0.7, 0.8, 0.9]
TENDENCY_LAG = 24
HALFLIVES = {
    "api24": 96.0,
    "api48": 192.0,
    "api72": 288.0,
}
INPUT_WINDOW = 288
FORECAST_WINDOW = 96
HIDDEN_UNITS = 64
INPUT_CHANNELS = 9
FORCING_CHANNELS = 9
NUMBER_OF_QUANTILES = len(QUANTILES)
BATCH_SIZE = 16
FLATLINE_WINDOW = 96

DEVELOPMENT_TRAIN_ORIGIN_START = pd.Timestamp("2023-01-04 00:00:00", tz="UTC")
DEVELOPMENT_TRAIN_ORIGIN_END = pd.Timestamp("2024-12-30 23:00:00", tz="UTC")
DEVELOPMENT_VALIDATION_ORIGIN_START = pd.Timestamp("2025-01-04 00:00:00", tz="UTC")
DEVELOPMENT_VALIDATION_ORIGIN_END = pd.Timestamp("2025-12-30 23:00:00", tz="UTC")
FINAL_TRAIN_ORIGIN_START = pd.Timestamp("2023-01-04 00:00:00", tz="UTC")
FINAL_TRAIN_ORIGIN_END = pd.Timestamp("2025-12-30 23:00:00", tz="UTC")
DEVELOPMENT_STATISTICS_END = pd.Timestamp("2024-12-31 23:45:00", tz="UTC")
FINAL_STATISTICS_END = pd.Timestamp("2025-12-31 23:45:00", tz="UTC")

EXPECTED_DEVELOPMENT_TRAIN_ORIGINS = 17448
EXPECTED_DEVELOPMENT_VALIDATION_ORIGINS = 8664
EXPECTED_FINAL_TRAIN_ORIGINS = 26208

if STAGE == "development":
    graph_path = os.path.abspath(arguments.development_graph)
    output_directory = os.path.join(
        MODELS_ROOT,
        "development",
        MODEL_KIND + "_" + GRAPH_KEY + "_seed" + str(SEED),
    )
    statistics_end = DEVELOPMENT_STATISTICS_END
else:
    graph_path = os.path.abspath(arguments.final_graph)
    output_directory = os.path.join(
        MODELS_ROOT,
        "final",
        MODEL_KIND + "_" + GRAPH_KEY + "_seed" + str(SEED),
    )
    statistics_end = FINAL_STATISTICS_END

os.makedirs(output_directory, exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_handle:
        while True:
            block = input_handle.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def normalize_time_index(frame):
    normalized = frame.sort_index().copy()
    normalized.index = pd.DatetimeIndex(normalized.index)
    if normalized.index.tz is None:
        normalized.index = normalized.index.tz_localize("UTC")
    else:
        normalized.index = normalized.index.tz_convert("UTC")
    if normalized.index.has_duplicates:
        raise ValueError("The training matrix contains duplicate timestamps.")
    return normalized


def build_origin_positions(index, origin_start, origin_end):
    row_by_time = {timestamp: row for row, timestamp in enumerate(index)}
    requested_origins = pd.date_range(origin_start, origin_end, freq="h")
    positions = []
    missing_origins = []
    for origin in requested_origins:
        if origin not in row_by_time:
            missing_origins.append(str(origin))
            continue
        position = int(row_by_time[origin])
        history_start = position - INPUT_WINDOW + 1
        target_end = position + FORECAST_WINDOW
        if history_start < 0:
            missing_origins.append(str(origin) + " insufficient history")
            continue
        if target_end >= len(index):
            missing_origins.append(str(origin) + " insufficient target")
            continue
        positions.append(position)
    if missing_origins:
        raise ValueError(
            "Origin construction failed for "
            + str(len(missing_origins))
            + " requested origins. First failures: "
            + str(missing_origins[:5])
        )
    return np.asarray(positions, dtype=np.int64)


def build_rain_array(frame, nodes, stage_to_rain, suffix):
    matching_columns = [column for column in frame.columns if column.endswith(suffix)]
    if matching_columns:
        global_mean = (
            frame[matching_columns]
            .apply(pd.to_numeric, errors="coerce")
            .mean(axis=1)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
    else:
        global_mean = np.zeros(len(frame), dtype=np.float32)
    values = np.zeros((len(frame), len(nodes)), dtype=np.float32)
    for node_index, node in enumerate(nodes):
        rain_code = str(stage_to_rain.get(node, ""))
        column = rain_code + suffix
        if column in frame.columns:
            values[:, node_index] = (
                pd.to_numeric(frame[column], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32)
            )
        else:
            values[:, node_index] = global_mean
    return values


def build_api(rain_step, halflife):
    decay = float(0.5 ** (1.0 / halflife))
    values = np.zeros_like(rain_step, dtype=np.float32)
    accumulator = np.zeros(rain_step.shape[1], dtype=np.float32)
    for row in range(rain_step.shape[0]):
        accumulator = decay * accumulator + rain_step[row]
        values[row] = accumulator
    return values


def standardize_with_mask(values, row_mask):
    selected = values[row_mask]
    mean_value = float(np.nanmean(selected))
    standard_deviation = float(np.nanstd(selected))
    if not np.isfinite(mean_value):
        raise ValueError("A feature mean is not finite.")
    if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ValueError("A feature standard deviation is not positive and finite.")
    adjusted_standard_deviation = standard_deviation + 1.0e-6
    standardized = ((values - mean_value) / adjusted_standard_deviation).astype(np.float32)
    return standardized, mean_value, adjusted_standard_deviation


required_paths = [arguments.training_pickle, arguments.stage_to_rain, graph_path]
for required_path in required_paths:
    if not os.path.exists(required_path):
        raise FileNotFoundError("Required System B input does not exist: " + required_path)

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as determinism_exception:
    raise RuntimeError(
        "TensorFlow deterministic operations could not be enabled: "
        + str(determinism_exception)
    )

available_gpus = tf.config.list_physical_devices("GPU")
for gpu_device in available_gpus:
    tf.config.experimental.set_memory_growth(gpu_device, True)

print("[CONFIG] Current directory:", os.getcwd())
print("[CONFIG] Experiment root:", EXPERIMENT_ROOT)
print("[CONFIG] Stage:", STAGE)
print("[CONFIG] Model kind:", MODEL_KIND)
print("[CONFIG] Graph key:", GRAPH_KEY)
print("[CONFIG] Seed:", SEED)
print("[CONFIG] Models root:", MODELS_ROOT)
print("[CONFIG] Experiment name:", EXPERIMENT_NAME)
print("[CONFIG] Expected node count:", EXPECTED_NODE_COUNT)
print("[CONFIG] Training pickle:", arguments.training_pickle)
print("[CONFIG] Graph path:", graph_path)
print("[CONFIG] Stage-to-rain mapping:", arguments.stage_to_rain)
print("[CONFIG] Output directory:", output_directory)
print("[CONFIG] Available GPUs:", available_gpus)

training_pickle_sha256 = sha256_file(arguments.training_pickle)
graph_sha256 = sha256_file(graph_path)
stage_to_rain_sha256 = sha256_file(arguments.stage_to_rain)
print("[HASH] Training pickle:", training_pickle_sha256)
print("[HASH] Graph:", graph_sha256)
print("[HASH] Stage-to-rain:", stage_to_rain_sha256)

graph = np.load(graph_path, allow_pickle=True)
nodes = [str(value) for value in graph["nodes"]]
adjacency = graph["A_norm"].astype(np.float32)
node_count = len(nodes)
if node_count != EXPECTED_NODE_COUNT:
    raise ValueError(
        "The graph node count does not match --expected-node-count. Expected: "
        + str(EXPECTED_NODE_COUNT)
        + "; found: "
        + str(node_count)
    )
if adjacency.shape != (node_count, node_count):
    raise ValueError("Adjacency shape is inconsistent with the node count.")
graph_nonzero_count = int(np.count_nonzero(adjacency))
if STAGE == "final" and MODEL_KIND == "stgnn" and GRAPH_KEY == "obs_pre2026":
    if graph_nonzero_count != 471:
        raise ValueError(
            "The final System B observation graph must contain 471 nonzero directed weights. Found: "
            + str(graph_nonzero_count)
        )
if MODEL_KIND == "lstm":
    model_adjacency = np.eye(node_count, dtype=np.float32)
else:
    model_adjacency = adjacency
print("[GRAPH] Nodes:", node_count)
print("[GRAPH] Stored nonzero directed weights:", graph_nonzero_count)
print("[GRAPH] Model kind uses graph messages:", MODEL_KIND == "stgnn")

print("[DATA] Loading frozen pre-2026 matrix.")
dataframe = pd.read_pickle(arguments.training_pickle)
dataframe = normalize_time_index(dataframe)
if dataframe.index.max() != FINAL_STATISTICS_END:
    raise ValueError(
        "The frozen training matrix must end at 2025-12-31 23:45 UTC. Found: "
        + str(dataframe.index.max())
    )
if dataframe.index.min() > pd.Timestamp("2023-01-01 00:00:00", tz="UTC"):
    raise ValueError("The frozen training matrix begins later than 1 January 2023.")
time_differences = dataframe.index.to_series().diff().dropna()
if not bool((time_differences == pd.Timedelta(minutes=15)).all()):
    raise ValueError("The frozen training matrix is not a regular 15-minute time series.")
print("[DATA] Matrix shape:", dataframe.shape)
print("[DATA] Matrix range:", dataframe.index.min(), "through", dataframe.index.max())

time_count = len(dataframe)
stage_values = np.full((time_count, node_count), np.nan, dtype=np.float32)
validity_mask = np.zeros((time_count, node_count), dtype=bool)
for node_index, node in enumerate(nodes):
    stage_column = node + "_stage_ft"
    if stage_column not in dataframe.columns:
        print("[WARN] Stage column is absent:", stage_column)
        continue
    stage_series = pd.to_numeric(dataframe[stage_column], errors="coerce")
    trailing_standard_deviation = stage_series.rolling(
        window=FLATLINE_WINDOW,
        center=False,
        min_periods=FLATLINE_WINDOW,
    ).std()
    stage_series = stage_series.mask(trailing_standard_deviation < 1.0e-6)
    stage_values[:, node_index] = stage_series.to_numpy(dtype=np.float32)
    validity_mask[:, node_index] = np.isfinite(stage_values[:, node_index])

stage_to_rain = (
    pd.read_csv(arguments.stage_to_rain)
    .set_index("stage")["nearest_rain"]
    .astype(str)
    .to_dict()
)
rain6 = build_rain_array(dataframe, nodes, stage_to_rain, "_rain_in_accum_6h")
rain24 = build_rain_array(dataframe, nodes, stage_to_rain, "_rain_in_accum_24h")
rain_step = build_rain_array(dataframe, nodes, stage_to_rain, "_rain_in")
api24 = build_api(rain_step, HALFLIVES["api24"])
api48 = build_api(rain_step, HALFLIVES["api48"])
api72 = build_api(rain_step, HALFLIVES["api72"])
rain_effect = rain6 * api48
rain_acceleration = np.zeros_like(rain_step, dtype=np.float32)
rain_acceleration[4:] = rain_step[4:] - rain_step[:-4]

statistics_row_mask = dataframe.index <= statistics_end
stage_mean = np.zeros(node_count, dtype=np.float32)
stage_standard_deviation = np.ones(node_count, dtype=np.float32)
no_training_support = np.zeros(node_count, dtype=bool)
for node_index in range(node_count):
    values = stage_values[statistics_row_mask, node_index]
    values = values[np.isfinite(values)]
    if len(values) < 96:
        no_training_support[node_index] = True
        validity_mask[:, node_index] = False
        print(
            "[STAGE MASK] Gauge has fewer than 96 finite values in the fitting period; "
            "all inputs and targets are masked:",
            nodes[node_index],
            "count=",
            len(values),
        )
        continue
    stage_mean[node_index] = float(np.mean(values))
    stage_standard_deviation[node_index] = float(np.std(values) + 1.0e-6)

stage_standardized = np.nan_to_num(
    (stage_values - stage_mean[None, :]) / stage_standard_deviation[None, :]
).astype(np.float32)
stage_standardized[:, no_training_support] = 0.0
stage_tendency = np.zeros_like(stage_standardized, dtype=np.float32)
stage_tendency[TENDENCY_LAG:] = (
    stage_standardized[TENDENCY_LAG:] - stage_standardized[:-TENDENCY_LAG]
)

rain6_standardized, rain6_mean, rain6_standard_deviation = standardize_with_mask(
    rain6,
    statistics_row_mask,
)
rain24_standardized, rain24_mean, rain24_standard_deviation = standardize_with_mask(
    rain24,
    statistics_row_mask,
)
api24_standardized, api24_mean, api24_standard_deviation = standardize_with_mask(
    api24,
    statistics_row_mask,
)
api48_standardized, api48_mean, api48_standard_deviation = standardize_with_mask(
    api48,
    statistics_row_mask,
)
api72_standardized, api72_mean, api72_standard_deviation = standardize_with_mask(
    api72,
    statistics_row_mask,
)
rain_effect_standardized, rain_effect_mean, rain_effect_standard_deviation = standardize_with_mask(
    rain_effect,
    statistics_row_mask,
)
rain_acceleration_standardized, rain_acceleration_mean, rain_acceleration_standard_deviation = standardize_with_mask(
    rain_acceleration,
    statistics_row_mask,
)
stage_tendency_standardized, stage_tendency_mean, stage_tendency_standard_deviation = standardize_with_mask(
    stage_tendency,
    statistics_row_mask,
)

hour_angle = 2.0 * np.pi * (
    dataframe.index.hour.to_numpy(dtype=np.float32)
    + dataframe.index.minute.to_numpy(dtype=np.float32) / 60.0
) / 24.0
time_sine = np.sin(hour_angle).astype(np.float32)
time_cosine = np.cos(hour_angle).astype(np.float32)

tensor_values = {
    "stage": tf.constant(stage_standardized),
    "rain6": tf.constant(rain6_standardized),
    "rain24": tf.constant(rain24_standardized),
    "api24": tf.constant(api24_standardized),
    "api48": tf.constant(api48_standardized),
    "api72": tf.constant(api72_standardized),
    "tendency": tf.constant(stage_tendency_standardized),
    "rain_effect": tf.constant(rain_effect_standardized),
    "rain_acceleration": tf.constant(rain_acceleration_standardized),
    "time_sine": tf.constant(time_sine),
    "time_cosine": tf.constant(time_cosine),
}
validity_tensor = tf.constant(validity_mask.astype(np.float32))


def make_training_example(origin_position):
    history_start = origin_position - INPUT_WINDOW + 1
    forecast_start = origin_position + 1
    forecast_end = forecast_start + FORECAST_WINDOW
    history = tf.stack(
        [
            tensor_values["stage"][history_start:origin_position + 1],
            tensor_values["rain6"][history_start:origin_position + 1],
            tensor_values["rain24"][history_start:origin_position + 1],
            tensor_values["api24"][history_start:origin_position + 1],
            tensor_values["api48"][history_start:origin_position + 1],
            tensor_values["api72"][history_start:origin_position + 1],
            tensor_values["tendency"][history_start:origin_position + 1],
            tensor_values["rain_effect"][history_start:origin_position + 1],
            tensor_values["rain_acceleration"][history_start:origin_position + 1],
        ],
        axis=-1,
    )
    future_time_sine = tf.tile(
        tensor_values["time_sine"][forecast_start:forecast_end, None],
        [1, node_count],
    )
    future_time_cosine = tf.tile(
        tensor_values["time_cosine"][forecast_start:forecast_end, None],
        [1, node_count],
    )
    forcing = tf.stack(
        [
            tensor_values["rain6"][forecast_start:forecast_end],
            tensor_values["rain24"][forecast_start:forecast_end],
            tensor_values["api24"][forecast_start:forecast_end],
            tensor_values["api48"][forecast_start:forecast_end],
            tensor_values["api72"][forecast_start:forecast_end],
            tensor_values["rain_effect"][forecast_start:forecast_end],
            tensor_values["rain_acceleration"][forecast_start:forecast_end],
            future_time_sine,
            future_time_cosine,
        ],
        axis=-1,
    )
    stage_at_origin = tensor_values["stage"][origin_position]
    target = tensor_values["stage"][forecast_start:forecast_end] - stage_at_origin[None, :]
    sample_weight = (
        validity_tensor[forecast_start:forecast_end]
        * validity_tensor[origin_position][None, :]
    )
    return (history, forcing, stage_at_origin), target, sample_weight


if STAGE == "development":
    training_positions = build_origin_positions(
        dataframe.index,
        DEVELOPMENT_TRAIN_ORIGIN_START,
        DEVELOPMENT_TRAIN_ORIGIN_END,
    )
    validation_positions = build_origin_positions(
        dataframe.index,
        DEVELOPMENT_VALIDATION_ORIGIN_START,
        DEVELOPMENT_VALIDATION_ORIGIN_END,
    )
    if len(training_positions) != EXPECTED_DEVELOPMENT_TRAIN_ORIGINS:
        raise ValueError(
            "Unexpected development training origin count: " + str(len(training_positions))
        )
    if len(validation_positions) != EXPECTED_DEVELOPMENT_VALIDATION_ORIGINS:
        raise ValueError(
            "Unexpected development validation origin count: " + str(len(validation_positions))
        )
    if arguments.smoke:
        training_positions = training_positions[:64]
        validation_positions = validation_positions[:32]
    training_dataset = tf.data.Dataset.from_tensor_slices(training_positions)
    training_dataset = training_dataset.map(
        make_training_example,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    training_dataset = training_dataset.shuffle(
        buffer_size=min(2048, len(training_positions)),
        seed=SEED,
        reshuffle_each_iteration=True,
    )
    training_dataset = training_dataset.batch(BATCH_SIZE).prefetch(2)
    validation_dataset = tf.data.Dataset.from_tensor_slices(validation_positions)
    validation_dataset = validation_dataset.map(
        make_training_example,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(2)
    training_epochs = 1 if arguments.smoke else int(arguments.maximum_development_epochs)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        )
    ]
    print("[SPLIT] Development training origins:", len(training_positions))
    print(
        "[SPLIT] Development training period:",
        DEVELOPMENT_TRAIN_ORIGIN_START,
        "through",
        DEVELOPMENT_TRAIN_ORIGIN_END,
    )
    print("[SPLIT] Development validation origins:", len(validation_positions))
    print(
        "[SPLIT] Development validation period:",
        DEVELOPMENT_VALIDATION_ORIGIN_START,
        "through",
        DEVELOPMENT_VALIDATION_ORIGIN_END,
    )
else:
    training_positions = build_origin_positions(
        dataframe.index,
        FINAL_TRAIN_ORIGIN_START,
        FINAL_TRAIN_ORIGIN_END,
    )
    if len(training_positions) != EXPECTED_FINAL_TRAIN_ORIGINS:
        raise ValueError("Unexpected final training origin count: " + str(len(training_positions)))
    if arguments.smoke:
        training_positions = training_positions[:64]
    development_directory = os.path.join(
        MODELS_ROOT,
        "development",
        MODEL_KIND + "_" + GRAPH_KEY + "_seed" + str(SEED),
    )
    development_history_path = os.path.join(development_directory, "train_history.json")
    if arguments.epochs is None:
        if not os.path.exists(development_history_path):
            raise FileNotFoundError(
                "Final fitting requires the development history or an explicit --epochs value: "
                + development_history_path
            )
        with open(development_history_path, "r", encoding="utf-8") as development_handle:
            development_history = json.load(development_handle)
        training_epochs = int(development_history["best_epoch"])
    else:
        training_epochs = int(arguments.epochs)
    if training_epochs <= 0:
        raise ValueError("Final training epoch count must be positive.")
    if arguments.smoke:
        training_epochs = 1
    training_dataset = tf.data.Dataset.from_tensor_slices(training_positions)
    training_dataset = training_dataset.map(
        make_training_example,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    training_dataset = training_dataset.shuffle(
        buffer_size=min(2048, len(training_positions)),
        seed=SEED,
        reshuffle_each_iteration=True,
    )
    training_dataset = training_dataset.batch(BATCH_SIZE).prefetch(2)
    validation_dataset = None
    callbacks = []
    print("[SPLIT] Final fitting origins:", len(training_positions))
    print(
        "[SPLIT] Final fitting period:",
        FINAL_TRAIN_ORIGIN_START,
        "through",
        FINAL_TRAIN_ORIGIN_END,
    )
    print("[SPLIT] Final fixed epoch count selected during development:", training_epochs)

quantile_tensor = tf.constant(QUANTILES, dtype=tf.float32)


def multi_quantile_pinball_loss(observed, predicted):
    residual = observed[..., None] - predicted
    pinball = tf.maximum(
        quantile_tensor * residual,
        (quantile_tensor - 1.0) * residual,
    )
    return tf.reduce_mean(pinball, axis=-1)


model = build_model(
    MODEL_KIND,
    model_adjacency,
    node_count,
    INPUT_WINDOW,
    FORECAST_WINDOW,
    INPUT_CHANNELS,
    FORCING_CHANNELS,
    HIDDEN_UNITS,
    NUMBER_OF_QUANTILES,
)
dummy_history = np.zeros(
    (1, INPUT_WINDOW, node_count, INPUT_CHANNELS),
    dtype=np.float32,
)
dummy_forcing = np.zeros(
    (1, FORECAST_WINDOW, node_count, FORCING_CHANNELS),
    dtype=np.float32,
)
dummy_stage_at_origin = np.zeros((1, node_count), dtype=np.float32)
model(
    [dummy_history, dummy_forcing, dummy_stage_at_origin],
    training=False,
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1.0e-3, clipnorm=1.0),
    loss=multi_quantile_pinball_loss,
)
print("[MODEL] Parameter count:", model.count_params())
print("[TRAIN] Epochs requested:", training_epochs)
history_object = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=training_epochs,
    callbacks=callbacks,
    verbose=2,
)

history = {
    key: [float(value) for value in values]
    for key, values in history_object.history.items()
}
if STAGE == "development":
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    best_validation_loss = float(np.min(history["val_loss"]))
else:
    best_epoch = int(training_epochs)
    best_validation_loss = None

weights_path = os.path.join(output_directory, "gnn.weights.h5")
metadata_path = os.path.join(output_directory, "gnn_meta.json")
history_path = os.path.join(output_directory, "train_history.json")
completion_path = os.path.join(output_directory, "RUN_COMPLETE.json")
model.save_weights(weights_path)

metadata = {
    "system": "System B",
    "experiment": EXPERIMENT_NAME,
    "research_only": True,
    "stage": STAGE,
    "model_kind": MODEL_KIND,
    "graph_key": GRAPH_KEY,
    "graph_mode": GRAPH_KEY if MODEL_KIND == "stgnn" else "none_nodewise",
    "seed": SEED,
    "nodes": nodes,
    "mu": stage_mean.tolist(),
    "sd": stage_standard_deviation.tolist(),
    "r6": [rain6_mean, rain6_standard_deviation],
    "r24": [rain24_mean, rain24_standard_deviation],
    "api24": [api24_mean, api24_standard_deviation],
    "api48": [api48_mean, api48_standard_deviation],
    "api72": [api72_mean, api72_standard_deviation],
    "re": [rain_effect_mean, rain_effect_standard_deviation],
    "raccel": [rain_acceleration_mean, rain_acceleration_standard_deviation],
    "tend": [stage_tendency_mean, stage_tendency_standard_deviation],
    "halflives": HALFLIVES,
    "tend_lag": TENDENCY_LAG,
    "feat_version": "DQ",
    "quantiles": QUANTILES,
    "IN_CH": INPUT_CHANNELS,
    "F_CH": FORCING_CHANNELS,
    "IN_W": INPUT_WINDOW,
    "LB_W": FORECAST_WINDOW,
    "H": HIDDEN_UNITS,
    "rain6_steps": 6,
    "rain6_hours": 1.5,
    "rain24_steps": 24,
    "rain24_hours": 6.0,
    "decoder_steps": 96,
    "decoder_hours": 24.0,
    "flatline_detection": "causal trailing 96-step standard deviation",
    "training_pickle": os.path.abspath(arguments.training_pickle),
    "training_pickle_sha256": training_pickle_sha256,
    "graph": graph_path,
    "graph_sha256": graph_sha256,
    "graph_nonzero_count": graph_nonzero_count,
    "no_fitting_support_nodes": [
        nodes[node_index]
        for node_index in range(node_count)
        if bool(no_training_support[node_index])
    ],
    "fitting_support_node_count": int(node_count - np.count_nonzero(no_training_support)),
    "stage_to_rain": os.path.abspath(arguments.stage_to_rain),
    "stage_to_rain_sha256": stage_to_rain_sha256,
    "statistics_end_utc": str(statistics_end),
    "origin_count": int(len(training_positions)),
    "epochs_requested": int(training_epochs),
    "best_epoch": int(best_epoch),
    "best_validation_loss": best_validation_loss,
    "created_utc": datetime.now(timezone.utc).isoformat(),
}
if STAGE == "development":
    metadata["training_origin_start_utc"] = str(DEVELOPMENT_TRAIN_ORIGIN_START)
    metadata["training_origin_end_utc"] = str(DEVELOPMENT_TRAIN_ORIGIN_END)
    metadata["validation_origin_start_utc"] = str(DEVELOPMENT_VALIDATION_ORIGIN_START)
    metadata["validation_origin_end_utc"] = str(DEVELOPMENT_VALIDATION_ORIGIN_END)
    metadata["validation_assignment"] = "chronological block; never shuffled with training origins"
else:
    metadata["training_origin_start_utc"] = str(FINAL_TRAIN_ORIGIN_START)
    metadata["training_origin_end_utc"] = str(FINAL_TRAIN_ORIGIN_END)
    metadata["validation_assignment"] = "none during final fit; fixed epoch count imported from development"

with open(metadata_path, "w", encoding="utf-8") as metadata_handle:
    json.dump(metadata, metadata_handle, indent=2)
with open(history_path, "w", encoding="utf-8") as history_handle:
    json.dump(
        {
            "history": history,
            "epochs_run": len(history["loss"]),
            "best_epoch": int(best_epoch),
            "best_validation_loss": best_validation_loss,
            "n_parameters": int(model.count_params()),
        },
        history_handle,
        indent=2,
    )

completion = {
    "status": "complete",
    "stage": STAGE,
    "model_kind": MODEL_KIND,
    "graph_key": GRAPH_KEY,
    "seed": SEED,
    "weights_path": weights_path,
    "weights_sha256": sha256_file(weights_path),
    "metadata_path": metadata_path,
    "metadata_sha256": sha256_file(metadata_path),
    "history_path": history_path,
    "history_sha256": sha256_file(history_path),
    "completed_utc": datetime.now(timezone.utc).isoformat(),
}
with open(completion_path, "w", encoding="utf-8") as completion_handle:
    json.dump(completion, completion_handle, indent=2)

print("[SAVE] Weights:", weights_path)
print("[SAVE] Metadata:", metadata_path)
print("[SAVE] History:", history_path)
print("[SAVE] Completion manifest:", completion_path)
print("[RESULT] Best epoch:", best_epoch)
print("[RESULT] Best validation loss:", best_validation_loss)
print("SYSTEM_B_TRAINING_COMPLETE")
