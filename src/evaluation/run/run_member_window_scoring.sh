#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
# Window statistics for every weather-model member over January to August 2026.
#
# Why this is needed
# ------------------
# Section 4.1 states that the individual weather-model members miss 66 % to 71 % of the crests of
# rises of at least 0.61 m. That statistic needs the observed rise and the peak error of every
# origin, which only the window scorer produces.
#
# run_window_scoring.sh already did this for three forcings: observed rain, the native
# issue-time HRRR, and the six-model mean. The members were never scored that way, so the per-member
# crest share exists only for January to June. This adds the members on the correct period.
#
# Nothing is run through a model again. This scores archives that already exist in
# runs_members_jan_aug/ and runs_ens_jan_aug/.
#
# Members are Open-Meteo products. Note that gfs_hrrr is Open-Meteo's blended GFS and HRRR product,
# which is NOT the native issue-time HRRR stitch scored by run_window_scoring.sh. The two
# score differently, 0.1851 m against 0.1907 m at h+24, and the manuscript quotes the native one.
#
# Usage:
#   bash run_member_window_scoring.sh
set -u

EXPERIMENT="project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924"
SYSTEM_B="project/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913"
SCORER="${REPO_ROOT}/src/evaluation/score_forecast_windows.py"

ORIGIN_START="2026-01-01 00:00"
ORIGIN_END="2026-08-31 23:00"

SCORED_DIR="${EXPERIMENT}/scored_members_jan_aug"
LOG_DIR="${EXPERIMENT}/logs"
mkdir -p "${SCORED_DIR}" "${LOG_DIR}"

MEMBERS="ecmwf_ifs025 gefs_geavg gefs_pmm gem_global gfs_global gfs_hrrr jma_seamless ncep_nbm_conus"
ENSEMBLES="ens6_median ens8_mean"

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
  for MEMBER in ${MEMBERS}; do
    score_one "${MEMBER}" "${EXPERIMENT}/runs_members_jan_aug/${MEMBER}_seed${SEED}_jan_aug.npz" "${SEED}"
  done
  for ENSEMBLE in ${ENSEMBLES}; do
    score_one "${ENSEMBLE}" "${EXPERIMENT}/runs_ens_jan_aug/${ENSEMBLE}_seed${SEED}_jan_aug.npz" "${SEED}"
  done
done

echo "[scorer] scored directories:"
ls -d "${SCORED_DIR}"/*/ 2>/dev/null | wc -l
echo "MEMBER_WINDOW_SCORING_JAN_AUG_DONE"
