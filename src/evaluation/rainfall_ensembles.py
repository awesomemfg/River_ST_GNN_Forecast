"""Form the six-model rainfall ensembles for January to August, from the member stage forecasts.

An ensemble here is not a rainfall product. It is the element-wise mean or median, across members, of
the members' full 96-step stage trajectories, formed per seed so a weight set is never mixed with
another. The trajectory is built first and scored afterwards, because the peak of the mean is not the
mean of the peaks.

This mirrors the published builder exactly:
  project/Experiments/SYSTEM_B_ENSEMBLE_FORCING_20260914/
  scripts/04_build_ensembles.py
same member list, same two statistics, same output layout, and the same strict checks that every
member shares origins, nodes and issue-time stage.

  ens6_mean    the six weather centers, mean
  ens6_median  the same six, median

  ens8_mean    the six plus the GEFS ensemble mean and the GEFS probability-matched mean

The GEFS rainfall comes from a separate fetcher, so July and August had to be pulled for all 31
individual members before the probability-matched mean could be rebuilt. Those runs exist now, so the
eight-member ensemble builds for the same period as the other two.

Inputs:  ../runs_members_jan_aug/<member>_seed<seed>_jan_aug.npz
Outputs: ../runs_ens_jan_aug/<ensemble>_seed<seed>_jan_aug.npz

Usage:
  conda run -n operational python -u rainfall_ensembles.py
"""
import json
import os

import numpy as np

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
MEMBER_RUNS = os.path.join(EXPERIMENT_ROOT, "runs_members_jan_aug")
OUTPUT_RUNS = os.path.join(EXPERIMENT_ROOT, "runs_ens_jan_aug")

SEEDS = [101, 202, 303]
MEMBERS6 = ["gfs_hrrr", "gfs_global", "ncep_nbm_conus", "ecmwf_ifs025", "gem_global", "jma_seamless"]
MEMBERS8 = MEMBERS6 + ["gefs_geavg", "gefs_pmm"]
ENSEMBLES = {
    "ens6_mean": (MEMBERS6, "mean"),
    "ens6_median": (MEMBERS6, "median"),
    "ens8_mean": (MEMBERS8, "mean"),
}
REQUIRED_MEMBERS = sorted({member for members, _ in ENSEMBLES.values() for member in members})
EXPECTED_ORIGINS = 5832

os.makedirs(OUTPUT_RUNS, exist_ok=True)


def log(*parts):
    print("[ens]", *parts, flush=True)


for seed in SEEDS:
    loaded = {}
    reference = None
    for member in REQUIRED_MEMBERS:
        path = os.path.join(MEMBER_RUNS, member + "_seed" + str(seed) + "_jan_aug.npz")
        if not os.path.exists(path):
            raise SystemExit("[FATAL] missing member archive: " + path)
        archive = np.load(path, allow_pickle=True)
        if reference is None:
            reference = {
                "origins_utc": archive["origins_utc"],
                "nodes": archive["nodes"],
                "quantiles_saved": archive["quantiles_saved"],
                "issue_stage_ft": archive["issue_stage_ft"],
                "run_meta": json.loads(str(archive["run_meta"])),
            }
            if len(reference["origins_utc"]) != EXPECTED_ORIGINS:
                raise SystemExit(
                    "[FATAL] expected "
                    + str(EXPECTED_ORIGINS)
                    + " origins, found "
                    + str(len(reference["origins_utc"]))
                )
        else:
            if not np.array_equal(archive["origins_utc"], reference["origins_utc"]):
                raise SystemExit("[FATAL] origins differ: " + path)
            if not np.array_equal(archive["nodes"], reference["nodes"]):
                raise SystemExit("[FATAL] nodes differ: " + path)
            if not np.array_equal(archive["issue_stage_ft"], reference["issue_stage_ft"], equal_nan=True):
                raise SystemExit("[FATAL] issue-time stage differs: " + path)
        loaded[member] = archive["pred_ft"].astype(np.float32)
        archive.close()
        log("seed", seed, "loaded", member, loaded[member].shape)

    for name, (members, statistic) in ENSEMBLES.items():
        stack = np.stack([loaded[member] for member in members], axis=0)
        if statistic == "mean":
            predictions = stack.mean(axis=0)
        else:
            predictions = np.median(stack, axis=0)
        meta = dict(reference["run_meta"])
        meta["forcing"] = "ensemble of member runs, January to August"
        meta["forcing_sha256"] = ""
        meta["ensemble_statistic"] = statistic
        meta["ensemble_members"] = [
            os.path.join(MEMBER_RUNS, member + "_seed" + str(seed) + "_jan_aug.npz")
            for member in members
        ]
        meta["ensemble_builder"] = os.path.abspath(__file__)
        out_path = os.path.join(OUTPUT_RUNS, name + "_seed" + str(seed) + "_jan_aug.npz")
        np.savez_compressed(
            out_path,
            origins_utc=reference["origins_utc"],
            nodes=reference["nodes"],
            quantiles_saved=reference["quantiles_saved"],
            pred_ft=predictions.astype(np.float32),
            issue_stage_ft=reference["issue_stage_ft"],
            run_meta=json.dumps(meta),
        )
        log("wrote", out_path)
        del stack, predictions
    del loaded

print("ENSEMBLES_JAN_AUG_DONE", flush=True)
