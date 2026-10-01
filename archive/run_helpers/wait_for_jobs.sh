#!/usr/bin/env bash
# Block until both detached jobs finish, then print one summary.
#
# Both jobs were started with nohup, so nothing notifies the agent when they end. This watcher waits
# on their completion tokens instead of polling them by hand:
#   extended hindcast  EXTENDED_HINDCAST_DONE in the driver log
#   HRRR fetch         HRRR_FETCH_DONE in the fetch log, or the fetch process going away
#
# Progress is judged from files, not from the logs, because conda run buffers a child's output until
# the child exits. A run that is halfway through still shows an empty log.
#
# Usage:
#   bash wait_for_jobs.sh            (run it in the background)
set -u

EXPERIMENT="project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924"
XUE="project/Experiments/HRRR_FORCING_AND_LSTM_20260911"

DRIVER_LOG="${EXPERIMENT}/logs/run_extended.log"
FETCH_LOG="${XUE}/logs/01_fetch_july_august.log"
CYCLE_DIR="${XUE}/data/cycles"
PROGRESS_LOG="${EXPERIMENT}/logs/wait_for_jobs_progress.log"

MAX_SECONDS=5400
INTERVAL=60
WAITED=0

runs_done() {
  grep -q "EXTENDED_HINDCAST_DONE" "${DRIVER_LOG}" 2>/dev/null
}

fetch_done() {
  if grep -q "HRRR_FETCH_DONE" "${FETCH_LOG}" 2>/dev/null; then
    return 0
  fi
  if ! pgrep -f "hrrr_fetch_points.py" > /dev/null 2>&1; then
    return 0
  fi
  return 1
}

while [ "${WAITED}" -lt "${MAX_SECONDS}" ]; do
  ARCHIVES=$(ls "${EXPERIMENT}/runs"/*_observed_jan_aug.npz 2>/dev/null | wc -l)
  CYCLES=$(ls "${CYCLE_DIR}"/hrrr_points_*.npz 2>/dev/null | wc -l)
  echo "$(date -Is) archives ${ARCHIVES}/6  cycle_day_files ${CYCLES}/244" >> "${PROGRESS_LOG}"
  if runs_done && fetch_done; then
    break
  fi
  sleep "${INTERVAL}"
  WAITED=$((WAITED + INTERVAL))
done

echo "=== watcher summary $(date -Is) ==="
echo "waited_seconds ${WAITED}"
echo "--- extended hindcast ---"
tail -20 "${DRIVER_LOG}"
ls -la "${EXPERIMENT}/runs"/*_observed_jan_aug.npz 2>/dev/null
echo "--- HRRR fetch ---"
echo "cycle day files: $(ls "${CYCLE_DIR}"/hrrr_points_*.npz 2>/dev/null | wc -l) of 244 expected"
ls "${CYCLE_DIR}"/hrrr_points_*.npz 2>/dev/null | tail -3
tail -5 "${FETCH_LOG}" 2>/dev/null
echo "WATCHER_DONE"
