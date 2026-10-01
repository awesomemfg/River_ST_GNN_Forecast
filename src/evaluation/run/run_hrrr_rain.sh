#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
# The forecast-rain arm: the same frozen models at every hourly origin from 1 January to 31 August
# 2026, driven by the High-Resolution Rapid Refresh (HRRR) rainfall forecast that existed at each
# issue time, instead of observed rain.
#
# The forcing file is hrrr_issue_time_stitch_L2_jan_aug.npz, built from 244 daily cycle files by
# hrrr_issue_time_forcing.py with an explicit origin range. The frozen January to June forcing
# file is untouched: its hash is recorded in the System B asset manifest and in every earlier run's
# metadata. The stitch rule uses forecast hours 1 to 18 from hourly cycles and 1 to 48 from the
# 00, 06, 12 and 18 UTC cycles, with a 2 hour availability lag, so no f000 analysis rain is used.
#
# Usage:
#   bash run_hrrr_rain.sh
set -u

EXPERIMENT="project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924"
SYSTEM_B="project/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913"
XUE="project/Experiments/HRRR_FORCING_AND_LSTM_20260911"
LSTM_WEIGHTS="project/hpc/Experiments/HRRR_FORCING_AND_LSTM_20260911/weights"
FORCING="${XUE}/data/forcing/hrrr_issue_time_stitch_L2_jan_aug.npz"

ORIGIN_START="2026-01-01 00:00"
ORIGIN_END="2026-08-31 23:00"
THREADS=8

RUN_DIR="${EXPERIMENT}/runs"
LOG_DIR="${EXPERIMENT}/logs"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

if [ ! -f "${FORCING}" ]; then
  echo "[driver] FATAL: forcing file missing: ${FORCING}"
  exit 1
fi
echo "[driver] forcing ${FORCING}"

for SEED in 101 202 303; do
  OUT="${RUN_DIR}/stgnn_seed${SEED}_hrrr_jan_aug.npz"
  if [ -f "${OUT}" ]; then
    echo "[driver] keep existing ${OUT}"
  else
    echo "[driver] ST-GNN seed ${SEED} start $(date -Is)"
    conda run -n operational python -u "${REPO_ROOT}/src/stgnn/hindcast.py" \
      --kind stgnn \
      --model-dir "${SYSTEM_B}/models/final/stgnn_obs_pre2026_seed${SEED}" \
      --forcing "${FORCING}" \
      --origin-start "${ORIGIN_START}" \
      --origin-end "${ORIGIN_END}" \
      --threads "${THREADS}" \
      --out "${OUT}" > "${LOG_DIR}/stgnn_seed${SEED}_hrrr.log" 2>&1
    echo "[driver] ST-GNN seed ${SEED} exit $? $(date -Is)"
  fi
done

for SEED in 101 202 303; do
  OUT="${RUN_DIR}/lstm_seed${SEED}_hrrr_jan_aug.npz"
  if [ -f "${OUT}" ]; then
    echo "[driver] keep existing ${OUT}"
  else
    echo "[driver] LSTM seed ${SEED} start $(date -Is)"
    conda run -n operational python -u "${REPO_ROOT}/src/lstm/inference.py" \
      --kind lstm \
      --model-dir "${LSTM_WEIGHTS}/lstm_seed${SEED}" \
      --forcing "${FORCING}" \
      --origin-start "${ORIGIN_START}" \
      --origin-end "${ORIGIN_END}" \
      --threads "${THREADS}" \
      --out "${OUT}" > "${LOG_DIR}/lstm_seed${SEED}_hrrr.log" 2>&1
    echo "[driver] LSTM seed ${SEED} exit $? $(date -Is)"
  fi
done

echo "[driver] archives written:"
ls -la "${RUN_DIR}"/*_hrrr_jan_aug.npz
echo "EXTENDED_HRRR_HINDCAST_DONE"
