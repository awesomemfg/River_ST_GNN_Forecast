#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
# The reduced gauge networks again, this time with the issue-time HRRR rainfall forecast.
#
# Section 4.2 reports the network experiments under both observed rainfall and the rainfall forecast
# the parish actually has. The observed-rain arms already cover 1 January to 31 August. This adds the
# forecast-rain arms over the same origins so both forcings rest on one period.
#
#   boundary experiment  13 arms  inparish51 and ipsub{10,20,30,40} draws 1 to 3
#   network experiment    5 arms  sub{10,20,30,40,50}
#
# With three seeds that is 54 evaluations at 5,832 origins, about five hours. Nothing is trained.
# The 68-gauge control is not re-run: its forecast-rain archives already exist in ../runs.
#
# Usage:
#   bash run_network_subsets_hrrr_rain.sh
set -u

EXPERIMENT="project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924"
SYSTEM_B="project/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913"
XUE="project/Experiments/HRRR_FORCING_AND_LSTM_20260911"
BOUNDARY="project/hpc/Experiments/SYSTEM_B_BOUNDARY_INPARISH_20260914"
LADDER="project/hpc/Experiments/INPARISH_LADDER_SYSTEM_B_20260915"
FORCING="${XUE}/data/forcing/hrrr_issue_time_stitch_L2_jan_aug.npz"

ORIGIN_START="2026-01-01 00:00"
ORIGIN_END="2026-08-31 23:00"
THREADS=8

RUN_DIR="${EXPERIMENT}/runs_ladder_hrrr"
LOG_DIR="${EXPERIMENT}/logs"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

if [ ! -f "${FORCING}" ]; then
  echo "[driver] FATAL: forcing missing: ${FORCING}"
  exit 1
fi

run_arm() {
  local ROOT="$1"
  local ARM="$2"
  local SEED="$3"
  local MODEL_DIR="${ROOT}/models/final/stgnn_${ARM}_seed${SEED}"
  local GRAPH="${ROOT}/frozen_assets/graph_${ARM}_final.npz"
  local OUT="${RUN_DIR}/stgnn_${ARM}_seed${SEED}_hrrr_jan_aug.npz"

  if [ -f "${OUT}" ]; then
    echo "[driver] keep existing ${OUT}"
    return 0
  fi
  if [ ! -d "${MODEL_DIR}" ] || [ ! -f "${GRAPH}" ]; then
    echo "[driver] SKIP, missing weights or graph for ${ARM} seed ${SEED}"
    return 0
  fi
  echo "[driver] ${ARM} seed ${SEED} start $(date -Is)"
  conda run -n operational python -u "${REPO_ROOT}/src/stgnn/hindcast.py" \
    --kind stgnn \
    --model-dir "${MODEL_DIR}" \
    --graph "${GRAPH}" \
    --forcing "${FORCING}" \
    --origin-start "${ORIGIN_START}" \
    --origin-end "${ORIGIN_END}" \
    --threads "${THREADS}" \
    --out "${OUT}" > "${LOG_DIR}/ladder_hrrr_${ARM}_seed${SEED}.log" 2>&1
  echo "[driver] ${ARM} seed ${SEED} exit $? $(date -Is)"
}

ARMS="inparish51"
for SIZE in 10 20 30 40; do
  for DRAW in 1 2 3; do
    ARMS="${ARMS} ipsub${SIZE}_d${DRAW}"
  done
done

for ARM in ${ARMS}; do
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
ls "${RUN_DIR}"/*.npz 2>/dev/null | wc -l
echo "NETWORK_LADDER_HRRR_JAN_AUG_DONE"
