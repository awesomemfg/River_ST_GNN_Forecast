#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
# The four remaining candidate graphs, at every hourly origin from 1 January to 31 August 2026.
#
# Table 3 of the manuscript compares six graphs. Two of them already cover the longer period: the
# selected observation lead-lag graph and the identity graph. This runs the other four so the whole
# table rests on one period.
#
#   dem                 DEM downslope
#   dem_basin           DEM, within basin
#   hecras              HEC-RAS hydraulic
#   obs_pre2026_basin   observation lead-lag, within basin
#
# Nothing is trained. Each arm uses its own frozen weights and the graph it was fitted on, so the
# evaluator's hash check passes. Observed rainfall, postprocessing off, three seeds, matching the
# other January to August runs origin for origin.
#
# Usage:
#   bash run_alternative_graphs.sh
set -u

EXPERIMENT="project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924"
SYSTEM_B="project/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913"
MIKE_MODELS="project/hpc/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913/models/final"
GRAPHS="${SYSTEM_B}/frozen_assets/comparisons"

ORIGIN_START="2026-01-01 00:00"
ORIGIN_END="2026-08-31 23:00"
THREADS=8

RUN_DIR="${EXPERIMENT}/runs_graphs_jan_aug"
LOG_DIR="${EXPERIMENT}/logs"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

for KEY in dem dem_basin hecras obs_pre2026_basin; do
  GRAPH="${GRAPHS}/graph_${KEY}_final.npz"
  if [ ! -f "${GRAPH}" ]; then
    echo "[driver] SKIP, graph missing: ${GRAPH}"
    continue
  fi
  for SEED in 101 202 303; do
    MODEL_DIR="${MIKE_MODELS}/stgnn_${KEY}_seed${SEED}"
    OUT="${RUN_DIR}/stgnn_${KEY}_seed${SEED}_observed_jan_aug.npz"
    if [ -f "${OUT}" ]; then
      echo "[driver] keep existing ${OUT}"
      continue
    fi
    if [ ! -d "${MODEL_DIR}" ]; then
      echo "[driver] SKIP, weights missing: ${MODEL_DIR}"
      continue
    fi
    echo "[driver] ${KEY} seed ${SEED} start $(date -Is)"
    conda run -n operational python -u "${REPO_ROOT}/src/stgnn/hindcast.py" \
      --kind stgnn \
      --model-dir "${MODEL_DIR}" \
      --graph "${GRAPH}" \
      --forcing perfect \
      --origin-start "${ORIGIN_START}" \
      --origin-end "${ORIGIN_END}" \
      --threads "${THREADS}" \
      --out "${OUT}" > "${LOG_DIR}/graph_${KEY}_seed${SEED}.log" 2>&1
    echo "[driver] ${KEY} seed ${SEED} exit $? $(date -Is)"
  done
done

echo "[driver] archives written:"
ls -la "${RUN_DIR}" | tail -6
echo "GRAPHS_JAN_AUG_DONE"
