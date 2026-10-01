#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
# Wait until all nine replay archives are finished zip files, then summarize them.
set -u

EXPERIMENT="project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924"
RUN_DIR="${EXPERIMENT}/runs"
MODELS="stgnn lstm gru"
SEEDS="101 202 303"

echo "[summary] waiting for nine valid postprocessing archives"
ready=0
while [ "${ready}" -eq 0 ]
do
  ready=1
  if pgrep -f "simulated_operational_forecast.py" >/dev/null
  then
    ready=0
  fi
  for MODEL in ${MODELS}
  do
    for SEED in ${SEEDS}
    do
      FILE="${RUN_DIR}/${MODEL}_seed${SEED}_hrrr_jan_aug_postproc.npz"
      if [ ! -f "${FILE}" ]
      then
        ready=0
      fi
    done
  done
  if [ "${ready}" -eq 1 ]
  then
    python3 - << 'PY'
import glob
import sys
import zipfile

folder = "project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924/runs"
paths = sorted(glob.glob(folder + "/*_hrrr_jan_aug_postproc.npz"))
if len(paths) != 9:
    sys.exit(1)
for path in paths:
    if not zipfile.is_zipfile(path):
        sys.exit(1)
print("[summary] nine valid zip archives")
PY
    if [ "$?" -ne 0 ]
    then
      ready=0
    fi
  fi
  if [ "${ready}" -eq 0 ]
  then
    sleep 30
  fi
done

echo "[summary] all nine archives are valid $(date -Is)"
conda run -n operational python -u "${REPO_ROOT}/src/evaluation/summarize_simulated_forecast.py"
echo "REPLAY_SUMMARY_DONE"
conda run -n operational python -u "archive/figure_script_generation/write_simulated_forecast_paragraphs.py"
echo "REPLAY_PARAGRAPHS_DONE"
conda run -n operational python -u "archive/figure_script_generation/source_simulated_forecast_figures.py"
echo "REPLAY_FIGURES_DONE"
