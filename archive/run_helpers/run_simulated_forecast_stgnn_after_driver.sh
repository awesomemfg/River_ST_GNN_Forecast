#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
# The first replay driver already passed the Spatio-Temporal Graph Neural Network because the
# evaluator could not import its model code. That import is fixed. This script waits until that
# driver exits, then runs the three graph-model seeds with postprocessing on.
set -u

EXPERIMENT="project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924"
SYSTEM_B="project/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913"
XUE="project/Experiments/HRRR_FORCING_AND_LSTM_20260911"
FORCING="${XUE}/data/forcing/hrrr_issue_time_stitch_L2_jan_aug.npz"
EVALUATOR="${REPO_ROOT}/src/stgnn/simulated_operational_forecast.py"
RUN_DIR="${EXPERIMENT}/runs"
LOG_DIR="${EXPERIMENT}/logs"
ORIGIN_START="2026-01-01 00:00"
ORIGIN_END="2026-08-31 23:00"
THREADS=8

echo "[stgnn] waiting for the replay driver to finish"
while pgrep -f "run_simulated_forecast.sh" >/dev/null
do
  sleep 30
done
echo "[stgnn] driver finished $(date -Is)"

for SEED in 101 202 303
do
  OUT="${RUN_DIR}/stgnn_seed${SEED}_hrrr_jan_aug_postproc.npz"
  if [ -f "${OUT}" ]
  then
    echo "[stgnn] keep existing ${OUT}"
  else
    echo "[stgnn] seed ${SEED} start $(date -Is)"
    conda run -n operational python -u "${EVALUATOR}" \
      --kind stgnn \
      --model-dir "${SYSTEM_B}/models/final/stgnn_obs_pre2026_seed${SEED}" \
      --forcing "${FORCING}" \
      --origin-start "${ORIGIN_START}" \
      --origin-end "${ORIGIN_END}" \
      --postproc on \
      --threads "${THREADS}" \
      --out "${OUT}" > "${LOG_DIR}/stgnn_seed${SEED}_hrrr_postproc.log" 2>&1
    echo "[stgnn] seed ${SEED} exit $? $(date -Is)"
  fi
done
echo "REPLAY_STGNN_POSTPROC_DONE"
