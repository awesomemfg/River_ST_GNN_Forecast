"""Merge each weather model's January to June rainfall archive with its July to August fetch, then
convert the result into the per-origin layout the evaluator reads.

Why a merge is needed
---------------------
The archived member rainfall stops at 2026-07-01 23:45, so extending Section 4.1 to 31 August is not
a recomputation: the July and August record had to be fetched. The fetcher writes a separate file per
member and skips any file that already exists, so the January to June archive could not be harmed and
was left untouched. This script joins the two records without modifying either source.

The 1 July overlap (96 fifteen-minute steps) is resolved in favor of the original archive, so every
value the published results already rest on is preserved exactly.

The conversion itself is the committed converter, called unchanged apart from its new origin-window
arguments:
  project/Experiments/HRRR_FORCING_AND_LSTM_20260911/scripts/
  weather_model_issue_layout.py

Outputs, written to ../forcing_jan_aug:
  <member>_merged_jan_aug.pkl          the joined rainfall record
  <member>_issue_layout_jan_aug.npz    the per-origin layout for 5,832 origins

GEFS is deliberately absent. Its members come from a separate fetcher, so the eight-member ensemble
cannot be extended to August without that fetch, while the six-model ensemble can.

Usage:
  conda run -n operational python -u weather_model_forcing_jan_aug.py
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIRECTORY)
OUTPUT_DIRECTORY = os.path.join(EXPERIMENT_ROOT, "forcing_jan_aug")

ORIGINAL_QPF = "project/Experiments/QPF_ENSEMBLE_20260831/qpf"
FETCHED_QPF = os.path.join(EXPERIMENT_ROOT, "qpf_jul_aug")
CONVERTER = (
    "project/Experiments/HRRR_FORCING_AND_LSTM_20260911/"
    "scripts/weather_model_issue_layout.py"
)

MEMBERS = [
    "ecmwf_ifs025",
    "gfs_global",
    "gem_global",
    "jma_seamless",
    "ncep_nbm_conus",
    "gfs_hrrr",
]
ORIGIN_START = "2026-01-01 00:00"
ORIGIN_END = "2026-08-31 23:00"
EXPECTED_STEPS = 23328
EXPECTED_ORIGINS = 5832

# The member list and the two source directories are arguments so the GEFS pair, whose July and
# August fetch landed in its own directory, runs through this same merge rather than a copy of it.
# The defaults reproduce the deterministic six-member build exactly.
parser = argparse.ArgumentParser()
parser.add_argument("--members", default=",".join(MEMBERS))
parser.add_argument("--original-dir", default=ORIGINAL_QPF)
parser.add_argument("--fetched-dir", default=FETCHED_QPF)
arguments = parser.parse_args()
MEMBERS = [name.strip() for name in arguments.members.split(",") if name.strip()]
ORIGINAL_QPF = arguments.original_dir
FETCHED_QPF = arguments.fetched_dir

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)


def log(*parts):
    print("[merge]", *parts, flush=True)


for member in MEMBERS:
    original_path = os.path.join(ORIGINAL_QPF, f"qpf_{member}.pkl")
    fetched_path = os.path.join(FETCHED_QPF, f"qpf_{member}.pkl")
    merged_path = os.path.join(OUTPUT_DIRECTORY, f"{member}_merged_jan_aug.pkl")
    layout_path = os.path.join(OUTPUT_DIRECTORY, f"{member}_issue_layout_jan_aug.npz")

    for required in (original_path, fetched_path):
        if not os.path.exists(required):
            raise SystemExit("[FATAL] missing rainfall record: " + required)

    original = pd.read_pickle(original_path).sort_index()
    fetched = pd.read_pickle(fetched_path).sort_index()
    if list(original.columns) != list(fetched.columns):
        raise SystemExit("[FATAL] column mismatch between the two records for " + member)

    # The original archive wins wherever the two overlap.
    merged = pd.concat([original, fetched[~fetched.index.isin(original.index)]]).sort_index()
    if len(merged) != EXPECTED_STEPS:
        raise SystemExit(
            "[FATAL] "
            + member
            + " merged to "
            + str(len(merged))
            + " steps, expected "
            + str(EXPECTED_STEPS)
        )
    step_differences = merged.index.to_series().diff().dropna()
    if bool((step_differences != pd.Timedelta(minutes=15)).any()):
        raise SystemExit("[FATAL] the merged record is not a regular 15-minute series: " + member)
    merged.to_pickle(merged_path)
    log(member, "merged", merged.shape, merged.index.min(), "->", merged.index.max())

    result = subprocess.run(
        [
            sys.executable,
            "-u",
            CONVERTER,
            "--qpf-pickle",
            merged_path,
            "--out",
            layout_path,
            "--origin-start",
            ORIGIN_START,
            "--origin-end",
            ORIGIN_END,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.strip().splitlines():
        log(member, line)

    import numpy as np

    archive = np.load(layout_path, allow_pickle=True)
    if len(archive["origins_utc"]) != EXPECTED_ORIGINS:
        raise SystemExit(
            "[FATAL] "
            + member
            + " layout holds "
            + str(len(archive["origins_utc"]))
            + " origins, expected "
            + str(EXPECTED_ORIGINS)
        )
    archive.close()

log("all members converted to", OUTPUT_DIRECTORY)
print("MEMBER_FORCING_JAN_AUG_DONE", flush=True)
