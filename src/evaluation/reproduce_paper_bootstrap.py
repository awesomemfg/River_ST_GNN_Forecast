"""Reproduce every paper h+24 RMSE interval using the deposited derived inputs.

Usage from the repository root:
python src/evaluation/reproduce_paper_resampling.py --data /path/to/evaluation-data

No TensorFlow, credentials, raw telemetry, or original project paths are needed.
The published origin order and 10,000 draws with seed 20260930 are retained.
Identical GRU/identity comparisons reuse the first comparison's gauge draws.
All other rows retain their original random-stream positions. Network rows use
the separately specified physical-gauge cluster bootstrap and 30 draw positions.
Block sums are evaluated by cumulative sums, not by changing the statistic.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, required=True,
                    help="Evaluation-data directory containing bootstrap_inputs/")
parser.add_argument("--output", type=Path)
arguments = parser.parse_args()
inputs = arguments.data.resolve() / "bootstrap_inputs"
if arguments.output is None:
    output = Path("results/paired_resampling.csv").resolve()
else:
    output = arguments.output.resolve()
if not inputs.is_dir():
    raise FileNotFoundError("Required resampling input directory not found: " + str(inputs))
print("Input directory:", inputs, flush=True)
print("Output file:", output, flush=True)
output.parent.mkdir(parents=True, exist_ok=True)
with (inputs / "comparisons.csv").open(newline="") as handle:
    comparisons = list(csv.DictReader(handle))
with np.load(inputs / "origins_and_events.npz", allow_pickle=False) as archive:
    origins = archive["origins_utc"]
    event = archive["event_origin"]
origin_count = len(origins)
replicates = 10000
block = 72
blocks_per_draw = int(np.ceil(origin_count / block))
if origin_count != 5832 or int(event.sum()) != 927:
    raise ValueError("The deposited origin/event counts do not match the paper")
generator = np.random.default_rng(20260930)
block_starts = generator.integers(0, origin_count - block + 1,
                                 size=(replicates, blocks_per_draw))
network_generator = np.random.default_rng(20260930)
network_starts = network_generator.integers(0, origin_count - block + 1,
                                          size=(replicates, blocks_per_draw))
if not np.array_equal(block_starts, network_starts):
    raise ValueError("The two bootstrap designs unexpectedly use different time blocks")
with (inputs / "network_units.csv").open(newline="") as handle:
    network_units = list(csv.DictReader(handle))
physical_gauges = sorted(set(row["gauge"] for row in network_units))
clusters = []
for gauge in physical_gauges:
    clusters.append(np.asarray([int(row["unit_index"]) for row in network_units
                               if row["gauge"] == gauge], dtype=int))
cluster_draws = network_generator.integers(0, len(clusters),
                                         size=(replicates, len(clusters)))
max_positions = len(clusters) * max(len(cluster) for cluster in clusters)
cluster_positions = np.full((replicates, max_positions), -1, dtype=int)
for replicate in range(replicates):
    selected = np.concatenate([clusters[index] for index in cluster_draws[replicate]])
    cluster_positions[replicate, :len(selected)] = selected
print("Origins:", origin_count, "events:", int(event.sum()), flush=True)
print("Network positions:", len(network_units), "physical gauges:", len(clusters), flush=True)
cache = {}
alias_draws = {}
rows = []


def calculate_rmse(filename, origin_set):
    key = (filename, origin_set)
    if key in cache:
        return cache[key]
    with np.load(inputs / filename, allow_pickle=False) as archive:
        error = archive["squared_error_ft2"]
        valid = archive["valid"].astype(np.float64)
        units = archive["units"]
    if origin_set == "event-only":
        mask = event.astype(np.float64)
    else:
        mask = np.ones(origin_count, dtype=np.float64)
    per_seed = []
    bootstrap_sum = np.zeros((replicates, len(units)), dtype=np.float64)
    for seed_index in range(error.shape[0]):
        masked_error = error[seed_index] * mask[:, None]
        masked_valid = valid[seed_index] * mask[:, None]
        totals = masked_error.sum(axis=0)
        counts = masked_valid.sum(axis=0)
        per_seed.append(np.sqrt(totals / np.maximum(counts, 1e-12)) * 0.3048)
        error_prefix = np.concatenate([np.zeros((1, len(units))),
                                       np.cumsum(masked_error, axis=0)])
        valid_prefix = np.concatenate([np.zeros((1, len(units))),
                                       np.cumsum(masked_valid, axis=0)])
        for start in range(0, replicates, 128):
            stop = min(start + 128, replicates)
            selected_starts = block_starts[start:stop]
            selected_stops = selected_starts + block
            sampled_error = (error_prefix[selected_stops] - error_prefix[selected_starts]).sum(axis=1)
            sampled_valid = (valid_prefix[selected_stops] - valid_prefix[selected_starts]).sum(axis=1)
            bootstrap_sum[start:stop] += np.sqrt(sampled_error / np.maximum(sampled_valid, 1e-12)) * 0.3048
        # Verify the prefix-sum implementation against direct weighted scoring.
        for replicate in range(3):
            weights = np.zeros(origin_count, dtype=np.float64)
            for selected_start in block_starts[replicate]:
                weights[selected_start:selected_start + block] += 1.0
            weights *= mask
            direct_error = (error[seed_index] * weights[:, None]).sum(axis=0)
            direct_valid = (valid[seed_index] * weights[:, None]).sum(axis=0)
            direct = np.sqrt(direct_error / np.maximum(direct_valid, 1e-12)) * 0.3048
            stops = block_starts[replicate] + block
            prefix_error = (error_prefix[stops] - error_prefix[block_starts[replicate]]).sum(axis=0)
            prefix_valid = (valid_prefix[stops] - valid_prefix[block_starts[replicate]]).sum(axis=0)
            accelerated = np.sqrt(prefix_error / np.maximum(prefix_valid, 1e-12)) * 0.3048
            if not np.allclose(direct, accelerated, rtol=0, atol=1e-10):
                raise ValueError("Prefix/direct scoring disagreement for " + filename)
    point = np.mean(per_seed, axis=0)
    draws = bootstrap_sum / error.shape[0]
    cache[key] = (point, draws, len(units))
    print("Calculated", filename, origin_set, "seeds:", error.shape[0], flush=True)
    return cache[key]


for comparison in comparisons:
    origin_set = comparison["origin_set"]
    baseline, base_draws, units = calculate_rmse(comparison["baseline_file"], origin_set)
    candidate, candidate_draws, candidate_units = calculate_rmse(comparison["candidate_file"], origin_set)
    if candidate_units != units:
        raise ValueError("Compared configurations have different scoring units")
    if comparison["sampling"] == "physical_gauge_clusters":
        selected_base = np.take_along_axis(base_draws, np.maximum(cluster_positions, 0), axis=1)
        selected_candidate = np.take_along_axis(candidate_draws, np.maximum(cluster_positions, 0), axis=1)
        selected_base[cluster_positions < 0] = np.nan
        selected_candidate[cluster_positions < 0] = np.nan
        differences = np.nanmedian(selected_candidate, axis=1) - np.nanmedian(selected_base, axis=1)
        resampled_units = len(clusters)
    else:
        gauge_draws = generator.integers(0, units, size=(replicates, units))
        alias_key = (comparison["baseline_file"], comparison["candidate_file"], origin_set)
        if alias_key in alias_draws:
            gauge_draws = alias_draws[alias_key]
            print("Reusing identical prediction comparison:", comparison["comparison"], flush=True)
        else:
            alias_draws[alias_key] = gauge_draws
        selected_base = np.take_along_axis(base_draws, gauge_draws, axis=1)
        selected_candidate = np.take_along_axis(candidate_draws, gauge_draws, axis=1)
        differences = np.median(selected_candidate, axis=1) - np.median(selected_base, axis=1)
        resampled_units = units
    low, middle, high = np.percentile(differences, [2.5, 50, 97.5])
    base_point = float(np.median(baseline))
    candidate_point = float(np.median(candidate))
    row = {"family": comparison["family"], "comparison": comparison["comparison"],
           "origin_set": origin_set, "units": units, "resampled_units": resampled_units,
           "sampling": comparison["sampling"], "baseline_rmse_m": base_point,
           "candidate_rmse_m": candidate_point, "difference_m": candidate_point - base_point,
           "bootstrap_median_m": float(middle), "ci_low_m": float(low), "ci_high_m": float(high),
           "share_positive": float(np.mean(differences > 0)),
           "interval_excludes_zero": bool(low > 0 or high < 0)}
    rows.append(row)
    print(comparison["family"], origin_set, comparison["comparison"],
          "difference:", row["difference_m"], "interval:", low, high, flush=True)
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("Saved:", output, "comparisons:", len(rows), flush=True)
