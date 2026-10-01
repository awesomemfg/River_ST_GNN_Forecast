#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
# The GRU-only arm: January to August 2026, observed rain and forecast rain.
#
# Nothing is trained here. The GRU-only baseline already exists. It is the System B ST-GNN fitted on
# an identity adjacency, trained on 2026-09-13 with the same recipe, width, channels, loss, epoch
# count and seeds as the selected model:
#   weights  models/final/stgnn_identity_seed{101,202,303}   (graph_key identity, 68 nonzero weights)
#   graph    frozen_assets/comparisons/graph_identity_final.npz
#            sha256 71aa9498f5c805fd6ead3b94a18604de1ca7b7b6afb2e4991a06281b3bf8f258, which is the
#            hash recorded in the trained models' metadata, so the evaluator's graph check passes.
#
# With an identity adjacency the graph message sum_u A[d,u] h_u reduces to the gauge's own hidden
# state, so the model is a gauge-by-gauge GRU with no exchange between gauges. Against the selected
# ST-GNN it isolates message passing; against the nodewise LSTM it isolates the recurrent cell.
#
# The evaluator is called with --kind stgnn because that is the model class in the metadata. The
# output files are named gru_* so the scorer reports them as GRU.
#
# Usage:
#   bash run_hindcast_gru.sh
set -u

EXPERIMENT="project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924"
SYSTEM_B="project/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913"
XUE="project/Experiments/HRRR_FORCING_AND_LSTM_20260911"
MIKE_MODELS="project/hpc/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913/models/final"
IDENTITY_GRAPH="${SYSTEM_B}/frozen_assets/comparisons/graph_identity_final.npz"
FORCING="${XUE}/data/forcing/hrrr_issue_time_stitch_L2_jan_aug.npz"

ORIGIN_START="2026-01-01 00:00"
ORIGIN_END="2026-08-31 23:00"
THREADS=8

RUN_DIR="${EXPERIMENT}/runs"
LOG_DIR="${EXPERIMENT}/logs"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

if [ ! -f "${IDENTITY_GRAPH}" ]; then
  echo "[driver] FATAL: identity graph missing: ${IDENTITY_GRAPH}"
  exit 1
fi

run_one() {
  local SEED="$1"
  local ARM="$2"
  local FORCING_ARGUMENT="$3"
  local OUT="${RUN_DIR}/gru_seed${SEED}_${ARM}_jan_aug.npz"
  if [ -f "${OUT}" ]; then
    echo "[driver] keep existing ${OUT}"
    return 0
  fi
  echo "[driver] GRU seed ${SEED} ${ARM} start $(date -Is)"
  conda run -n operational python -u "${REPO_ROOT}/src/stgnn/hindcast.py" \
    --kind stgnn \
    --model-dir "${MIKE_MODELS}/stgnn_identity_seed${SEED}" \
    --graph "${IDENTITY_GRAPH}" \
    --forcing "${FORCING_ARGUMENT}" \
    --origin-start "${ORIGIN_START}" \
    --origin-end "${ORIGIN_END}" \
    --threads "${THREADS}" \
    --out "${OUT}" > "${LOG_DIR}/gru_seed${SEED}_${ARM}.log" 2>&1
  echo "[driver] GRU seed ${SEED} ${ARM} exit $? $(date -Is)"
}

for SEED in 101 202 303; do
  run_one "${SEED}" "observed" "perfect"
done

if [ -f "${FORCING}" ]; then
  for SEED in 101 202 303; do
    run_one "${SEED}" "hrrr" "${FORCING}"
  done
else
  echo "[driver] forecast-rain forcing missing, skipping that arm: ${FORCING}"
fi

echo "[driver] archives written:"
ls -la "${RUN_DIR}"/gru_*_jan_aug.npz
echo "EXTENDED_GRU_HINDCAST_DONE"
