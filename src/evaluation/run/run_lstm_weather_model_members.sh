#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
# The LSTM driven by each weather model's rainfall forecast, January to June 2026.
#
# Why this exists
# ---------------
# The ST-GNN and the LSTM were compared under only two rainfall inputs: observed rain, where the
# ST-GNN leads clearly, and issue-time HRRR, which is the worst rainfall product available. Under
# that worst product the two tie at h+24 (0.1907 against 0.1893, a fifteenth of the seed floor).
#
# The six weather models span a measured quality ladder between those two extremes. Running the LSTM
# across the same six forcings the ST-GNN already used lets the head-to-head be read as a function of
# rainfall quality rather than at one unfavorable point.
#
# The result is reported whichever way it lands. If the ST-GNN pulls ahead as rain improves, that
# supports the mechanism. If the two stay tied, the honest conclusion is to lead with h+6 and h+12,
# where the ST-GNN is ahead by 0.008 to 0.009 m under the same forecast rain.
#
# Nothing is trained. These are the frozen LSTM weights and the evaluator that produced the paper's
# existing LSTM archives. The origin window matches the existing ST-GNN member runs exactly, so the
# two arms pair gauge by gauge and origin by origin.
#
# Usage:
#   bash run_lstm_weather_model_members.sh
set -u

EXPERIMENT="project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924"
XUE="project/Experiments/HRRR_FORCING_AND_LSTM_20260911"
ENSEMBLE="project/Experiments/SYSTEM_B_ENSEMBLE_FORCING_20260914"
LSTM_WEIGHTS="project/hpc/Experiments/HRRR_FORCING_AND_LSTM_20260911/weights"

ORIGIN_START="2026-01-01 00:00"
ORIGIN_END="2026-06-30 23:00"
THREADS=8

RUN_DIR="${EXPERIMENT}/runs_members_lstm"
LOG_DIR="${EXPERIMENT}/logs"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

MEMBERS=(ecmwf_ifs025 gfs_global gem_global jma_seamless ncep_nbm_conus gfs_hrrr)

for MEMBER in "${MEMBERS[@]}"; do
  FORCING="${ENSEMBLE}/forcing/${MEMBER}_issue_layout.npz"
  if [ ! -f "${FORCING}" ]; then
    echo "[driver] SKIP, forcing missing: ${FORCING}"
    continue
  fi
  for SEED in 101 202 303; do
    OUT="${RUN_DIR}/lstm_${MEMBER}_seed${SEED}.npz"
    if [ -f "${OUT}" ]; then
      echo "[driver] keep existing ${OUT}"
      continue
    fi
    echo "[driver] LSTM ${MEMBER} seed ${SEED} start $(date -Is)"
    conda run -n operational python -u "${REPO_ROOT}/src/lstm/inference.py" \
      --kind lstm \
      --model-dir "${LSTM_WEIGHTS}/lstm_seed${SEED}" \
      --forcing "${FORCING}" \
      --origin-start "${ORIGIN_START}" \
      --origin-end "${ORIGIN_END}" \
      --threads "${THREADS}" \
      --out "${OUT}" > "${LOG_DIR}/lstm_${MEMBER}_seed${SEED}.log" 2>&1
    echo "[driver] LSTM ${MEMBER} seed ${SEED} exit $? $(date -Is)"
  done
done

echo "[driver] archives written:"
ls -la "${RUN_DIR}" | tail -6
echo "LSTM_MEMBERS_DONE"
