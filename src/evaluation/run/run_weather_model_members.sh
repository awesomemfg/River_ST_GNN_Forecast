#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
# Section 4.1 extended: the ST-GNN driven by each weather model's rainfall forecast, at every hourly
# origin from 1 January to 31 August 2026.
#
# Nothing is trained. These are the frozen 471-edge System B weights and the same evaluator that
# produced the January to June member runs, with two changes: the origin window is longer, and the
# forcing is the merged January to August layout built by weather_model_forcing_jan_aug.py.
#
# The published January to June member runs and their forcing files are untouched. The new archives
# carry a _jan_aug suffix so the two periods can never be confused.
#
# Six weather centers times three seeds is 18 runs, roughly 100 minutes. GEFS is absent because its
# rainfall comes from a separate fetcher that was not run for July and August, so the eight-member
# ensemble stays a January to June result while the six-model ensemble extends.
#
# Usage:
#   bash run_weather_model_members.sh
set -u

EXPERIMENT="project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924"
SYSTEM_B="project/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913"
FORCING_DIR="${EXPERIMENT}/forcing_jan_aug"

ORIGIN_START="2026-01-01 00:00"
ORIGIN_END="2026-08-31 23:00"
THREADS=8

RUN_DIR="${EXPERIMENT}/runs_members_jan_aug"
LOG_DIR="${EXPERIMENT}/logs"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

MEMBERS=(ecmwf_ifs025 gfs_global gem_global jma_seamless ncep_nbm_conus gfs_hrrr)

for MEMBER in "${MEMBERS[@]}"; do
  FORCING="${FORCING_DIR}/${MEMBER}_issue_layout_jan_aug.npz"
  if [ ! -f "${FORCING}" ]; then
    echo "[driver] SKIP, forcing missing: ${FORCING}"
    continue
  fi
  for SEED in 101 202 303; do
    OUT="${RUN_DIR}/${MEMBER}_seed${SEED}_jan_aug.npz"
    if [ -f "${OUT}" ]; then
      echo "[driver] keep existing ${OUT}"
      continue
    fi
    echo "[driver] ${MEMBER} seed ${SEED} start $(date -Is)"
    conda run -n operational python -u "${REPO_ROOT}/src/stgnn/hindcast.py" \
      --kind stgnn \
      --model-dir "${SYSTEM_B}/models/final/stgnn_obs_pre2026_seed${SEED}" \
      --forcing "${FORCING}" \
      --origin-start "${ORIGIN_START}" \
      --origin-end "${ORIGIN_END}" \
      --threads "${THREADS}" \
      --out "${OUT}" > "${LOG_DIR}/member_${MEMBER}_seed${SEED}_jan_aug.log" 2>&1
    echo "[driver] ${MEMBER} seed ${SEED} exit $? $(date -Is)"
  done
done

echo "[driver] archives written:"
ls -la "${RUN_DIR}" | tail -6
echo "MEMBERS_JAN_AUG_DONE"
