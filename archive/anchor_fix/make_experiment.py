"""Copy the EXTEND_2026_AUG31_EVENTS_20260918 pipeline into this experiment, pointed at the anchor-fix evaluators.

Run make_evaluators.py first. This script
  1. copies every script of the original experiment into this experiment's scripts folder (this folder's own
     make_*.py files are kept);
  2. replaces the original experiment path by this experiment's path, and each evaluator name by its
     *_anchorfix.py copy, counting every replacement;
  3. links the inputs that do not depend on the anchor (frozen assets and rainfall forcing) to the originals.
Outputs of the original experiment (runs, scores, summaries, figures) are not copied: the drivers rebuild them here.
The graph "links" folder is rebuilt here, pointing at this experiment's graph runs.
"""
import glob
import os
import shutil

EXPERIMENTS = "project/Experiments/"
OLD = EXPERIMENTS + "EXTEND_2026_AUG31_EVENTS_20260918"
NEW = EXPERIMENTS + "EXTEND_2026_AUG31_ANCHORFIX_20260924"
EVALUATOR_NAMES = [
    ("evaluate_system_b_dropout.py", "evaluate_system_b_dropout_anchorfix.py"),
    ("hindcast_before_fix.py", "hindcast.py"),
    ("lstm_inference_before_fix.py", "inference.py"),
    ("evaluate_replay_postproc.py", "simulated_operational_forecast.py"),
]
INPUT_FOLDERS = ["frozen_assets", "forcing_jan_aug", "qpf_jul_aug", "gefs_jul_aug"]

os.makedirs(os.path.join(NEW, "scripts"), exist_ok=True)
counts = {}
for path in sorted(glob.glob(os.path.join(OLD, "scripts", "*"))):
    name = os.path.basename(path)
    if not os.path.isfile(path) or name.startswith("__"):
        continue
    if name.endswith("_anchorfix.py"):
        shutil.copy2(path, os.path.join(NEW, "scripts", name))
        continue
    text = open(path, encoding="utf-8", errors="strict").read()
    text = text.replace(OLD, NEW)
    for old_name, new_name in EVALUATOR_NAMES:
        counts[old_name] = counts.get(old_name, 0) + text.count(old_name)
        text = text.replace(old_name, new_name)
    target = os.path.join(NEW, "scripts", name)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(text)
    shutil.copymode(path, target)

for folder in INPUT_FOLDERS:
    link = os.path.join(NEW, folder)
    if not os.path.exists(link):
        os.symlink(os.path.join(OLD, folder), link)

old_links = os.path.join(OLD, "runs_graphs_all_links")
new_links = os.path.join(NEW, "runs_graphs_all_links")
os.makedirs(new_links, exist_ok=True)
for link in sorted(glob.glob(os.path.join(old_links, "*.npz"))):
    target = os.readlink(link).replace(OLD, NEW)
    destination = os.path.join(new_links, os.path.basename(link))
    if not os.path.lexists(destination):
        os.symlink(target, destination)

remaining = []
for path in glob.glob(os.path.join(NEW, "scripts", "*")):
    if os.path.isfile(path) and not os.path.basename(path).startswith("make_anchorfix"):
        text = open(path, encoding="utf-8").read()
        if OLD in text:
            remaining.append(os.path.basename(path))
print("[experiment] evaluator name replacements:", counts)
print("[experiment] scripts still naming the old experiment:", remaining or "none")
print("[experiment] inputs linked:", INPUT_FOLDERS)
