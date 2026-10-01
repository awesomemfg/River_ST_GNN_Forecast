#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
# Wait until all nine January-August replay archives exist, then write the Section 3.3 summaries.
set -u

EXPERIMENT="project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924"
RUN_DIR="${EXPERIMENT}/runs"
MODELS="stgnn lstm gru"
SEEDS="101 202 303"

echo "[summary] waiting for nine postprocessing archives"
ready=0
while [ "${ready}" -eq 0 ]
do
  ready=1
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
  if [ "${ready}" -eq 0 ]
  then
    sleep 30
  fi
done

echo "[summary] all nine archives are present $(date -Is)"
conda run -n operational python -u "${REPO_ROOT}/src/evaluation/summarize_simulated_forecast.py"
echo "REPLAY_SUMMARY_DONE"
conda run -n operational python -u "archive/figure_script_generation/write_simulated_forecast_paragraphs.py"
echo "REPLAY_PARAGRAPHS_DONE"
conda run -n operational python -u "archive/figure_script_generation/source_simulated_forecast_figures.py"
echo "REPLAY_FIGURES_DONE"
