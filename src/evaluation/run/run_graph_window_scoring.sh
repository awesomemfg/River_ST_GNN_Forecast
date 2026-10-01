#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
# Window statistics for the six graphs over January to August 2026.
#
# Why this is needed
# ------------------
# Section 3.1 ends by saying that the selected graph does not lead every peak measure: the HEC-RAS
# graph had the lower mean peak error for rises of at least 0.15 m, and missed the crest by more than
# 0.15 m less often. That is the paragraph arguing against the paper's own preferred graph, so it is
# worth keeping.
#
# The statistic needs the observed rise and the peak error of every origin, which only the window
# scorer produces. The six graphs were scored at fixed leads for January to August, but never window
# scored, so those two numbers exist only for January to June.
#
# Nothing is run through a model again. This scores archives that already exist. The archive
# directory is discovered rather than assumed, because the graph runs have been kept in more than one
# place during this experiment.
#
# Usage:
#   bash run_graph_window_scoring.sh
set -u

EXPERIMENT="project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924"
SYSTEM_B="project/Experiments/SYSTEM_B_471_CHRONOLOGICAL_20260913"
SCORER="${REPO_ROOT}/src/evaluation/score_forecast_windows.py"

ORIGIN_START="2026-01-01 00:00"
ORIGIN_END="2026-08-31 23:00"

SCORED_DIR="${EXPERIMENT}/scored_graphs_jan_aug"
LOG_DIR="${EXPERIMENT}/logs"
mkdir -p "${SCORED_DIR}" "${LOG_DIR}"

GRAPHS="gObsLagLead gObsBasin gDem gDemBasin gHecras gIdentity"

find_archive() {
  # Print the first archive that matches this graph and seed, wherever it is kept.
  local GRAPH="$1"
  local SEED="$2"
  local HIT
  HIT=$(ls -1 "${EXPERIMENT}"/runs*/"${GRAPH}_seed${SEED}"*.npz 2>/dev/null | head -1)
  echo "${HIT}"
}

score_one() {
  local LABEL="$1"
  local ARCHIVE="$2"
  local SEED="$3"
  local OUT="${SCORED_DIR}/${LABEL}_seed${SEED}"

  if [ -f "${OUT}/reforecast_2026H1_origin_window_metrics.csv" ]; then
    echo "[scorer] keep existing ${LABEL} seed ${SEED}"
    return 0
  fi
  if [ -z "${ARCHIVE}" ] || [ ! -f "${ARCHIVE}" ]; then
    echo "[scorer] SKIP, no archive found for ${LABEL} seed ${SEED}"
    return 0
  fi
  echo "[scorer] ${LABEL} seed ${SEED} from $(basename "${ARCHIVE}") start $(date -Is)"
  conda run -n operational python -u "${SCORER}" \
    --run "${ARCHIVE}" \
    --out-dir "${OUT}" \
    --origin-start "${ORIGIN_START}" \
    --origin-end "${ORIGIN_END}" > "${LOG_DIR}/window_graph_${LABEL}_seed${SEED}.log" 2>&1
  echo "[scorer] ${LABEL} seed ${SEED} exit $? $(date -Is)"
}

echo "[scorer] archives discovered:"
for SEED in 101 202 303; do
  for GRAPH in ${GRAPHS}; do
    echo "   ${GRAPH} seed ${SEED}: $(find_archive "${GRAPH}" "${SEED}")"
  done
done

for SEED in 101 202 303; do
  for GRAPH in ${GRAPHS}; do
    score_one "${GRAPH}" "$(find_archive "${GRAPH}" "${SEED}")" "${SEED}"
  done
done

echo "[scorer] scored directories:"
ls -d "${SCORED_DIR}"/*/ 2>/dev/null | wc -l
echo "GRAPH_WINDOW_SCORING_JAN_AUG_DONE"
