#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
# Per-origin window statistics for January to August, for the three forcings Table 6 quotes.
#
# Why this is needed
# ------------------
# Table 6 ends with a crest column: the share of rises of at least 0.61 m whose forecast crest falls
# more than 0.15 m short. That statistic needs the observed rise, the peak error and the peak bias of
# every origin, which only the window scorer produces. The event scorer used elsewhere in this
# experiment computes fixed-lead metrics and cannot supply it.
#
# The crest numbers currently in hand come from scored tables that cover January to June only, so
# they report 10,031 strong rises for a period that now holds 5,832 origins. These runs replace them
# on the correct period.
#
#   observed rain        ../runs/stgnn_seed<S>_observed_jan_aug.npz
#   issue-time HRRR      ../runs/stgnn_seed<S>_hrrr_jan_aug.npz
#   six-model ensemble   ../runs_ens_jan_aug/ens6_mean_seed<S>_jan_aug.npz
#
# Nothing is run through a model again. This scores archives that already exist.
#
# Usage:
#   bash run_window_scoring.sh
set -u

EXPERIMENT="project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924"
SYSTEM_B="project/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913"
SCORER="${REPO_ROOT}/src/evaluation/score_forecast_windows.py"

ORIGIN_START="2026-01-01 00:00"
ORIGIN_END="2026-08-31 23:00"

SCORED_DIR="${EXPERIMENT}/scored_jan_aug"
LOG_DIR="${EXPERIMENT}/logs"
mkdir -p "${SCORED_DIR}" "${LOG_DIR}"

score_one() {
  local LABEL="$1"
  local ARCHIVE="$2"
  local SEED="$3"
  local OUT="${SCORED_DIR}/${LABEL}_seed${SEED}"

  if [ -f "${OUT}/reforecast_2026H1_origin_window_metrics.csv" ]; then
    echo "[scorer] keep existing ${LABEL} seed ${SEED}"
    return 0
  fi
  if [ ! -f "${ARCHIVE}" ]; then
    echo "[scorer] SKIP, archive missing: ${ARCHIVE}"
    return 0
  fi
  echo "[scorer] ${LABEL} seed ${SEED} start $(date -Is)"
  conda run -n operational python -u "${SCORER}" \
    --run "${ARCHIVE}" \
    --out-dir "${OUT}" \
    --origin-start "${ORIGIN_START}" \
    --origin-end "${ORIGIN_END}" > "${LOG_DIR}/window_${LABEL}_seed${SEED}.log" 2>&1
  echo "[scorer] ${LABEL} seed ${SEED} exit $? $(date -Is)"
}

for SEED in 101 202 303; do
  score_one observed  "${EXPERIMENT}/runs/stgnn_seed${SEED}_observed_jan_aug.npz"        "${SEED}"
  score_one hrrr      "${EXPERIMENT}/runs/stgnn_seed${SEED}_hrrr_jan_aug.npz"            "${SEED}"
  score_one ens6_mean "${EXPERIMENT}/runs_ens_jan_aug/ens6_mean_seed${SEED}_jan_aug.npz" "${SEED}"
  # The LSTM is needed for the conclusion that compares both models against persistence on rises of
  # at least 0.61 m. That comparison uses the observed rise and the peak error of every window, so
  # the fixed-lead scores cannot supply it.
  score_one lstm_observed "${EXPERIMENT}/runs/lstm_seed${SEED}_observed_jan_aug.npz" "${SEED}"
  score_one lstm_hrrr     "${EXPERIMENT}/runs/lstm_seed${SEED}_hrrr_jan_aug.npz"     "${SEED}"
done

echo "[scorer] scored directories:"
ls -d "${SCORED_DIR}"/*/ 2>/dev/null | wc -l
echo "WINDOW_SCORING_JAN_AUG_DONE"
