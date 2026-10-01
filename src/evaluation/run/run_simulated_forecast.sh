#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
# Replay of the actual forecast, 1 January to 31 August 2026.
#
# Same frozen weights and the same issue-time High-Resolution Rapid Refresh rainfall as the raw
# forecast-rain hindcast. The difference is Section 2.5.2 postprocessing, switched on for every
# origin: stale-data repair of the input history, the continuity blend on the first 3 h, and the
# rain-conditioned rise cap. Stale-data repair changes the model input, so these archives cannot
# be made by editing the raw predictions.
#
# The raw archives stay in place. These files are named *_hrrr_jan_aug_postproc.npz.
#
# Usage:
#   bash run_simulated_forecast.sh
set -u

EXPERIMENT="project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924"
SYSTEM_B="project/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913"
XUE="project/Experiments/HRRR_FORCING_AND_LSTM_20260911"
LSTM_WEIGHTS="project/hpc/Experiments/HRRR_FORCING_AND_LSTM_20260911/weights"
MIKE_MODELS="project/hpc/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913/models/final"
IDENTITY_GRAPH="${SYSTEM_B}/frozen_assets/comparisons/graph_identity_final.npz"
FORCING="${XUE}/data/forcing/hrrr_issue_time_stitch_L2_jan_aug.npz"
EVALUATOR="${REPO_ROOT}/src/stgnn/simulated_operational_forecast.py"

ORIGIN_START="2026-01-01 00:00"
ORIGIN_END="2026-08-31 23:00"
THREADS=8

RUN_DIR="${EXPERIMENT}/runs"
LOG_DIR="${EXPERIMENT}/logs"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

if [ ! -f "${FORCING}" ]
then
  echo "[driver] FATAL: forcing file missing: ${FORCING}"
  exit 1
fi
if [ ! -f "${EVALUATOR}" ]
then
  echo "[driver] FATAL: evaluator missing: ${EVALUATOR}"
  exit 1
fi
echo "[driver] forcing ${FORCING}"
echo "[driver] postprocessing on for every origin"

for SEED in 101 202 303
do
  OUT="${RUN_DIR}/stgnn_seed${SEED}_hrrr_jan_aug_postproc.npz"
  if [ -f "${OUT}" ]
  then
    echo "[driver] keep existing ${OUT}"
  else
    echo "[driver] ST-GNN seed ${SEED} start $(date -Is)"
    conda run -n operational python -u "${EVALUATOR}" \
      --kind stgnn \
      --model-dir "${SYSTEM_B}/models/final/stgnn_obs_pre2026_seed${SEED}" \
      --forcing "${FORCING}" \
      --origin-start "${ORIGIN_START}" \
      --origin-end "${ORIGIN_END}" \
      --postproc on \
      --threads "${THREADS}" \
      --out "${OUT}" > "${LOG_DIR}/stgnn_seed${SEED}_hrrr_postproc.log" 2>&1
    echo "[driver] ST-GNN seed ${SEED} exit $? $(date -Is)"
  fi
done

for SEED in 101 202 303
do
  OUT="${RUN_DIR}/lstm_seed${SEED}_hrrr_jan_aug_postproc.npz"
  if [ -f "${OUT}" ]
  then
    echo "[driver] keep existing ${OUT}"
  else
    echo "[driver] LSTM seed ${SEED} start $(date -Is)"
    conda run -n operational python -u "${REPO_ROOT}/src/lstm/inference.py" \
      --kind lstm \
      --model-dir "${LSTM_WEIGHTS}/lstm_seed${SEED}" \
      --forcing "${FORCING}" \
      --origin-start "${ORIGIN_START}" \
      --origin-end "${ORIGIN_END}" \
      --postproc on \
      --threads "${THREADS}" \
      --out "${OUT}" > "${LOG_DIR}/lstm_seed${SEED}_hrrr_postproc.log" 2>&1
    echo "[driver] LSTM seed ${SEED} exit $? $(date -Is)"
  fi
done

for SEED in 101 202 303
do
  OUT="${RUN_DIR}/gru_seed${SEED}_hrrr_jan_aug_postproc.npz"
  if [ -f "${OUT}" ]
  then
    echo "[driver] keep existing ${OUT}"
  else
    echo "[driver] GRU seed ${SEED} start $(date -Is)"
    conda run -n operational python -u "${EVALUATOR}" \
      --kind stgnn \
      --model-dir "${MIKE_MODELS}/stgnn_identity_seed${SEED}" \
      --graph "${IDENTITY_GRAPH}" \
      --forcing "${FORCING}" \
      --origin-start "${ORIGIN_START}" \
      --origin-end "${ORIGIN_END}" \
      --postproc on \
      --threads "${THREADS}" \
      --out "${OUT}" > "${LOG_DIR}/gru_seed${SEED}_hrrr_postproc.log" 2>&1
    echo "[driver] GRU seed ${SEED} exit $? $(date -Is)"
  fi
done

echo "[driver] replay archives:"
ls -la "${RUN_DIR}"/*_hrrr_jan_aug_postproc.npz
echo "REPLAY_POSTPROC_DONE"
