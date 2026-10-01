#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
# Run the GRU-only model against each weather-model rainfall forecast, January to June 2026.
#
# Purpose
# -------
# The graph effect (GRU minus ST-GNN) is large with observed rain and reverses at h+24 with
# issue-time HRRR rain. The question is whether that is a property of forcing quality rather than of
# the two rain products. The six weather-model forecasts span a quality ladder measured on the same
# origins by the ST-GNN itself, h+24 median RMSE:
#
#   observed rain 0.1499 | JMA 0.1654 | NBM 0.1656 | GEM 0.1691 | ECMWF 0.1799 | GFS 0.1875
#   HRRR 0.1967 | issue-time HRRR 0.1984
#
# The ST-GNN side already exists in SYSTEM_B_ENSEMBLE_FORCING_20260914/runs. Only the GRU side is
# missing, so this fills it in and lets the graph effect be paired at every rung of the ladder.
#
# Nothing is trained. The weights are the existing identity-adjacency models, and the graph asset is
# the one whose hash matches their metadata.
#
# The origin window is January to June because that is what the member forcings cover, and it matches
# the existing ST-GNN member runs exactly, which is what makes the pairing valid.
#
# Usage:
#   bash run_gru_weather_model_members.sh
set -u

EXPERIMENT="project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924"
SYSTEM_B="project/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913"
ENSEMBLE="project/Experiments/SYSTEM_B_ENSEMBLE_FORCING_20260914"
MIKE_MODELS="project/hpc/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913/models/final"
IDENTITY_GRAPH="${SYSTEM_B}/frozen_assets/comparisons/graph_identity_final.npz"

ORIGIN_START="2026-01-01 00:00"
ORIGIN_END="2026-06-30 23:00"
THREADS=8

RUN_DIR="${EXPERIMENT}/runs_members"
LOG_DIR="${EXPERIMENT}/logs"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

MEMBERS=(ecmwf_ifs025 gfs_global gem_global jma_seamless ncep_nbm_conus gfs_hrrr)

if [ ! -f "${IDENTITY_GRAPH}" ]; then
  echo "[driver] FATAL: identity graph missing: ${IDENTITY_GRAPH}"
  exit 1
fi

for MEMBER in "${MEMBERS[@]}"; do
  FORCING="${ENSEMBLE}/forcing/${MEMBER}_issue_layout.npz"
  if [ ! -f "${FORCING}" ]; then
    echo "[driver] SKIP, forcing missing: ${FORCING}"
    continue
  fi
  for SEED in 101 202 303; do
    OUT="${RUN_DIR}/gru_${MEMBER}_seed${SEED}.npz"
    if [ -f "${OUT}" ]; then
      echo "[driver] keep existing ${OUT}"
      continue
    fi
    echo "[driver] GRU ${MEMBER} seed ${SEED} start $(date -Is)"
    conda run -n operational python -u "${REPO_ROOT}/src/stgnn/hindcast.py" \
      --kind stgnn \
      --model-dir "${MIKE_MODELS}/stgnn_identity_seed${SEED}" \
      --graph "${IDENTITY_GRAPH}" \
      --forcing "${FORCING}" \
      --origin-start "${ORIGIN_START}" \
      --origin-end "${ORIGIN_END}" \
      --threads "${THREADS}" \
      --out "${OUT}" > "${LOG_DIR}/gru_${MEMBER}_seed${SEED}.log" 2>&1
    echo "[driver] GRU ${MEMBER} seed ${SEED} exit $? $(date -Is)"
  done
done

echo "[driver] archives written:"
ls -la "${RUN_DIR}"
echo "GRU_ENSEMBLE_MEMBERS_DONE"
