#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
# Rebuild the Revision 14 result figures from the anchor-fix rerun, in dependency order.
# Run after EXTEND_2026_AUG31_ANCHORFIX_20260924/logs/anchorfix_all.log prints ANCHORFIX_ALL_DONE.
# The scripts are the counted copies written by make_figure_scripts.py. The operational env gives the same
# matplotlib (3.10.3) that drew the Revision 13 figures.
#
# Usage:
#   bash ${REPO_ROOT}/src/figures/run_all_figures.sh > ../work/run_rev14_figures.log 2>&1
set -u
HERE="project/manuscript/20260924_Revision_14/scripts"
PY="python"
FAILED=""
cd "${HERE}" || exit 1
mkdir -p ../figures ../work

for SCRIPT in ${REPO_ROOT}/src/figures/fig08_hindcast_skill.py \
              ${REPO_ROOT}/src/figures/fig09_hindcast_examples.py \
              ${REPO_ROOT}/src/figures/figB1_B2_C1_appendix.py \
              ${REPO_ROOT}/src/figures/fig10_simulated_forecast_skill.py \
              ${REPO_ROOT}/src/figures/simulated_forecast_model_comparison_labels.py \
              ${REPO_ROOT}/src/figures/fig11_event_only_model_comparison.py \
              ${REPO_ROOT}/src/figures/fig11_event_only_labels.py \
              ${REPO_ROOT}/src/figures/fig12_simulated_forecast_examples.py \
              ${REPO_ROOT}/src/figures/fig13_fig14_fig16_rainfall_crest_gauge_count.py; do
  echo "[figures] ${SCRIPT} start $(date -Is)"
  nice -n 10 "${PY}" -u "${SCRIPT}"
  STATUS=$?
  echo "[figures] ${SCRIPT} exit ${STATUS} $(date -Is)"
  if [ "${STATUS}" -ne 0 ]; then
    FAILED="${FAILED} ${SCRIPT}"
  fi
done
echo "[figures] failed:${FAILED:- none}"
echo "REV14_FIGURES_DONE"
