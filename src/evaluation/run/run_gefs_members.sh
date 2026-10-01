#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
# The ST-GNN driven by the two GEFS rainfall products, January to August 2026.
#
# ens8_mean needs eight members: the six weather centers, already run, plus the GEFS ensemble mean
# and the GEFS probability-matched mean. This fills in those last two so the eight-member ensemble
# can be rebuilt for the longer period.
#
# Nothing is trained. Same frozen weights, same evaluator, same origin window as the six
# deterministic members, so all eight pair origin for origin.
#
# Usage:
#   bash run_gefs_members.sh
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

for MEMBER in gefs_geavg gefs_pmm; do
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

echo "[driver] GEFS archives:"
ls -la "${RUN_DIR}"/gefs_*_jan_aug.npz 2>/dev/null
echo "GEFS_MEMBERS_JAN_AUG_DONE"
