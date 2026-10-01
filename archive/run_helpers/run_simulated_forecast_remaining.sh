#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
# Seed 101 of the graph model is already running and has no parent driver.
# Wait until its archive exists, then run the other two graph-model seeds and the three
# gated-recurrent-unit seeds. Do not start a second copy of seed 101.
set -u

EXPERIMENT="project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924"
SYSTEM_B="project/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913"
XUE="project/Experiments/HRRR_FORCING_AND_LSTM_20260911"
MIKE_MODELS="project/hpc/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913/models/final"
IDENTITY_GRAPH="${SYSTEM_B}/frozen_assets/comparisons/graph_identity_final.npz"
FORCING="${XUE}/data/forcing/hrrr_issue_time_stitch_L2_jan_aug.npz"
EVALUATOR="${REPO_ROOT}/src/stgnn/simulated_operational_forecast.py"
RUN_DIR="${EXPERIMENT}/runs"
LOG_DIR="${EXPERIMENT}/logs"
ORIGIN_START="2026-01-01 00:00"
ORIGIN_END="2026-08-31 23:00"
THREADS=8

FIRST="${RUN_DIR}/stgnn_seed101_hrrr_jan_aug_postproc.npz"
echo "[remaining] waiting for ${FIRST}"
while [ ! -f "${FIRST}" ]
do
  sleep 20
done
echo "[remaining] found seed 101 $(date -Is)"

run_one() {
  local LABEL="$1"
  local MODEL_DIR="$2"
  local OUT="$3"
  local LOG="$4"
  local GRAPH_ARG="$5"
  if [ -f "${OUT}" ]
  then
    echo "[remaining] keep existing ${OUT}"
    return 0
  fi
  echo "[remaining] ${LABEL} start $(date -Is)"
  if [ -n "${GRAPH_ARG}" ]
  then
    conda run -n operational python -u "${EVALUATOR}" \
      --kind stgnn \
      --model-dir "${MODEL_DIR}" \
      --graph "${GRAPH_ARG}" \
      --forcing "${FORCING}" \
      --origin-start "${ORIGIN_START}" \
      --origin-end "${ORIGIN_END}" \
      --postproc on \
      --threads "${THREADS}" \
      --out "${OUT}" > "${LOG}" 2>&1
  else
    conda run -n operational python -u "${EVALUATOR}" \
      --kind stgnn \
      --model-dir "${MODEL_DIR}" \
      --forcing "${FORCING}" \
      --origin-start "${ORIGIN_START}" \
      --origin-end "${ORIGIN_END}" \
      --postproc on \
      --threads "${THREADS}" \
      --out "${OUT}" > "${LOG}" 2>&1
  fi
  echo "[remaining] ${LABEL} exit $? $(date -Is)"
}

run_one "ST-GNN seed 202" \
  "${SYSTEM_B}/models/final/stgnn_obs_pre2026_seed202" \
  "${RUN_DIR}/stgnn_seed202_hrrr_jan_aug_postproc.npz" \
  "${LOG_DIR}/stgnn_seed202_hrrr_postproc.log" \
  ""

run_one "ST-GNN seed 303" \
  "${SYSTEM_B}/models/final/stgnn_obs_pre2026_seed303" \
  "${RUN_DIR}/stgnn_seed303_hrrr_jan_aug_postproc.npz" \
  "${LOG_DIR}/stgnn_seed303_hrrr_postproc.log" \
  ""

run_one "GRU seed 101" \
  "${MIKE_MODELS}/stgnn_identity_seed101" \
  "${RUN_DIR}/gru_seed101_hrrr_jan_aug_postproc.npz" \
  "${LOG_DIR}/gru_seed101_hrrr_postproc.log" \
  "${IDENTITY_GRAPH}"

run_one "GRU seed 202" \
  "${MIKE_MODELS}/stgnn_identity_seed202" \
  "${RUN_DIR}/gru_seed202_hrrr_jan_aug_postproc.npz" \
  "${LOG_DIR}/gru_seed202_hrrr_postproc.log" \
  "${IDENTITY_GRAPH}"

run_one "GRU seed 303" \
  "${MIKE_MODELS}/stgnn_identity_seed303" \
  "${RUN_DIR}/gru_seed303_hrrr_jan_aug_postproc.npz" \
  "${LOG_DIR}/gru_seed303_hrrr_postproc.log" \
  "${IDENTITY_GRAPH}"

echo "REPLAY_REMAINING_DONE"
