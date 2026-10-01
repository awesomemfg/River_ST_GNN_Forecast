#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
# Section 4.2 extended: every reduced gauge network run at each hourly origin from 1 January to
# 31 August 2026, with observed rain.
#
# Nothing is trained. These are the frozen arm weights already on the mount, each paired with the
# graph it was fitted on, so the evaluator's hash check passes:
#
#   boundary experiment  13 arms  inparish51 and ipsub{10,20,30,40} draws 1 to 3
#   ladder experiment     5 arms  sub{10,20,30,40,50}, the fixed random ladder from all 68
#
# With three seeds that is 54 evaluations at 5,832 origins, roughly five hours. The 68-gauge control
# is not re-run here: its January to August archives already exist in ../runs.
#
# The published Figure 16 uses fixed-lead h+24 RMSE per gauge, so the event version needs per-origin
# predictions rather than the pooled score files, which is why the archives are rebuilt rather than
# the existing scores being filtered.
#
# Usage:
#   bash run_network_subsets.sh
set -u

EXPERIMENT="project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924"
SYSTEM_B="project/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913"
BOUNDARY="project/hpc/Experiments/SYSTEM_B_BOUNDARY_INPARISH_20260914"
LADDER="project/hpc/Experiments/INPARISH_LADDER_SYSTEM_B_20260915"

ORIGIN_START="2026-01-01 00:00"
ORIGIN_END="2026-08-31 23:00"
THREADS=8

RUN_DIR="${EXPERIMENT}/runs_ladder"
LOG_DIR="${EXPERIMENT}/logs"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

run_arm() {
  local ROOT="$1"
  local ARM="$2"
  local SEED="$3"
  local MODEL_DIR="${ROOT}/models/final/stgnn_${ARM}_seed${SEED}"
  local GRAPH="${ROOT}/frozen_assets/graph_${ARM}_final.npz"
  local OUT="${RUN_DIR}/stgnn_${ARM}_seed${SEED}_observed_jan_aug.npz"

  if [ -f "${OUT}" ]; then
    echo "[driver] keep existing ${OUT}"
    return 0
  fi
  if [ ! -d "${MODEL_DIR}" ]; then
    echo "[driver] SKIP, weights missing: ${MODEL_DIR}"
    return 0
  fi
  if [ ! -f "${GRAPH}" ]; then
    echo "[driver] SKIP, graph missing: ${GRAPH}"
    return 0
  fi
  echo "[driver] ${ARM} seed ${SEED} start $(date -Is)"
  conda run -n operational python -u "${REPO_ROOT}/src/stgnn/hindcast.py" \
    --kind stgnn \
    --model-dir "${MODEL_DIR}" \
    --graph "${GRAPH}" \
    --forcing perfect \
    --origin-start "${ORIGIN_START}" \
    --origin-end "${ORIGIN_END}" \
    --threads "${THREADS}" \
    --out "${OUT}" > "${LOG_DIR}/ladder_${ARM}_seed${SEED}.log" 2>&1
  echo "[driver] ${ARM} seed ${SEED} exit $? $(date -Is)"
}

BOUNDARY_ARMS="inparish51"
for SIZE in 10 20 30 40; do
  for DRAW in 1 2 3; do
    BOUNDARY_ARMS="${BOUNDARY_ARMS} ipsub${SIZE}_d${DRAW}"
  done
done

for ARM in ${BOUNDARY_ARMS}; do
  for SEED in 101 202 303; do
    run_arm "${BOUNDARY}" "${ARM}" "${SEED}"
  done
done

for SIZE in 10 20 30 40 50; do
  for SEED in 101 202 303; do
    run_arm "${LADDER}" "sub${SIZE}" "${SEED}"
  done
done

echo "[driver] archives written:"
ls -la "${RUN_DIR}" | tail -5
echo "NETWORK_LADDER_JAN_AUG_DONE"
