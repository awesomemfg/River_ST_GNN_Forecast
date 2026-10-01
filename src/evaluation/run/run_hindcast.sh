#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
# Run the frozen models at every hourly origin from 1 January to 31 August 2026, with observed rain.
#
# Nothing is retrained. These are the same frozen weights and the same evaluators that produced the
# January to June archives the manuscript reports, with only the origin window extended:
#   ST-GNN  hindcast.py            (System B guards, models/final/stgnn_obs_pre2026_seed*)
#   LSTM    inference.py      (the evaluator that produced the paper's LSTM archives;
#                                            its weights carry no System B stamp, so the System B
#                                            evaluator refuses them by design)
# Persistence needs no run: the scorer builds it from the stage at issue time.
#
# The origin window 2026-01-01 00:00 to 2026-08-31 23:00 is 5,832 hourly origins, against 4,344 for
# January to June. The evaluation matrix ends 2026-09-11 05:00 UTC, so every 24 h window closes.
#
# Usage:
#   nohup nice -n 10 bash run_hindcast.sh > ../logs/run_extended.log 2>&1 &
set -u

EXPERIMENT="project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924"
SYSTEM_B="project/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913"
XUE="project/Experiments/HRRR_FORCING_AND_LSTM_20260911"
LSTM_WEIGHTS="project/hpc/Experiments/HRRR_FORCING_AND_LSTM_20260911/weights"

ORIGIN_START="2026-01-01 00:00"
ORIGIN_END="2026-08-31 23:00"
THREADS=8

RUN_DIR="${EXPERIMENT}/runs"
LOG_DIR="${EXPERIMENT}/logs"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

echo "[driver] free space on the experiment filesystem:"
df -h "${EXPERIMENT}"

for SEED in 101 202 303; do
  OUT="${RUN_DIR}/stgnn_seed${SEED}_observed_jan_aug.npz"
  if [ -f "${OUT}" ]; then
    echo "[driver] keep existing ${OUT}"
  else
    echo "[driver] ST-GNN seed ${SEED} start $(date -Is)"
    conda run -n operational python -u "${REPO_ROOT}/src/stgnn/hindcast.py" \
      --kind stgnn \
      --model-dir "${SYSTEM_B}/models/final/stgnn_obs_pre2026_seed${SEED}" \
      --forcing perfect \
      --origin-start "${ORIGIN_START}" \
      --origin-end "${ORIGIN_END}" \
      --threads "${THREADS}" \
      --out "${OUT}" > "${LOG_DIR}/stgnn_seed${SEED}.log" 2>&1
    echo "[driver] ST-GNN seed ${SEED} exit $? $(date -Is)"
  fi
done

for SEED in 101 202 303; do
  OUT="${RUN_DIR}/lstm_seed${SEED}_observed_jan_aug.npz"
  if [ -f "${OUT}" ]; then
    echo "[driver] keep existing ${OUT}"
  else
    echo "[driver] LSTM seed ${SEED} start $(date -Is)"
    conda run -n operational python -u "${REPO_ROOT}/src/lstm/inference.py" \
      --kind lstm \
      --model-dir "${LSTM_WEIGHTS}/lstm_seed${SEED}" \
      --forcing perfect \
      --origin-start "${ORIGIN_START}" \
      --origin-end "${ORIGIN_END}" \
      --threads "${THREADS}" \
      --out "${OUT}" > "${LOG_DIR}/lstm_seed${SEED}.log" 2>&1
    echo "[driver] LSTM seed ${SEED} exit $? $(date -Is)"
  fi
done

echo "[driver] archives written:"
ls -la "${RUN_DIR}"
echo "EXTENDED_HINDCAST_DONE"
