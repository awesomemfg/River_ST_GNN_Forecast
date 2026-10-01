#!/usr/bin/env bash
# Repository root, from this script's location in the repository.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"
# Rerun every January to August 2026 evaluation of the HESS paper with the anchor fix (Farid, 2026-09-24:
# "do change lines then rerun evrything").
#
# The fix itself is in the four *_anchorfix.py evaluators (${REPO_ROOT}/archive/anchor_fix/make_evaluators.py). This experiment is a
# copy of EXTEND_2026_AUG31_EVENTS_20260918 pointed at them (${REPO_ROOT}/archive/anchor_fix/make_experiment.py), with the one-off
# recovery wait removed from the replay driver (${REPO_ROOT}/archive/anchor_fix/make_driver_fixes.py).
#
# Phase A  model evaluations, 288 archives. The twelve drivers run side by side, each one sequential inside,
#          at 8 threads each (96 of the 128 cores). Every driver keeps an archive that already exists.
# Phase B  the rainfall-ensemble archives, built from the member archives of phase A.
# Phase C  per-origin window scoring of the archives that Table 6, Section 4.1 and the graph comparison use.
# Phase D  the scoring and summary scripts, in the order the original experiment ran them.
#
# The January to June links (runs_janjun_links) are not rebuilt: no paper script reads their scores.
# The paper figures and tables are regenerated afterwards from this experiment's outputs.
#
# Usage (detached, so an interrupted agent session does not stop it):
#   setsid nohup bash ${REPO_ROOT}/src/evaluation/run/run_all.sh > ../logs/anchorfix_all.log 2>&1 < /dev/null &
set -u
EXPERIMENT="project/Experiments/EXTEND_2026_AUG31_ANCHORFIX_20260924"
SCRIPTS="${EXPERIMENT}/scripts"
LOGS="${EXPERIMENT}/logs"
PY="python"
FAILED=""
mkdir -p "${LOGS}"
cd "${SCRIPTS}" || exit 1

step() {
  local NAME="$1"
  shift
  echo "[master] ${NAME} start $(date -Is)"
  nice -n 10 "$@" > "${LOGS}/${NAME}.log" 2>&1
  local STATUS=$?
  echo "[master] ${NAME} exit ${STATUS} $(date -Is)"
  if [ "${STATUS}" -ne 0 ]; then
    FAILED="${FAILED} ${NAME}"
  fi
}

echo "[master] phase A, model evaluations, start $(date -Is)"
DRIVERS="run_extended_hindcast_jan_aug run_extended_hindcast_hrrr_jan_aug run_extended_hindcast_gru_jan_aug \
run_replay_postproc_jan_aug run_graphs_jan_aug run_network_ladder_jan_aug run_network_ladder_hrrr_jan_aug \
run_members_jan_aug run_gefs_members_jan_aug run_lstm_members run_gru_ensemble_members run_dropout_jan_aug"
for DRIVER in ${DRIVERS}; do
  nice -n 10 bash "${SCRIPTS}/${DRIVER}.sh" > "${LOGS}/driver_${DRIVER}.log" 2>&1 &
  echo "[master] launched ${DRIVER} pid $!"
done
wait
echo "[master] phase A done $(date -Is)"
for DRIVER in ${DRIVERS}; do
  echo "[master] ${DRIVER} last line: $(tail -1 "${LOGS}/driver_${DRIVER}.log")"
done

echo "[master] phase B, rainfall ensembles"
step ensembles_jan_aug "${PY}" -u ${REPO_ROOT}/src/evaluation/rainfall_ensembles.py

echo "[master] phase C, window scoring"
nice -n 10 bash ${REPO_ROOT}/src/evaluation/run/run_window_scoring.sh > "${LOGS}/driver_window_scoring.log" 2>&1 &
nice -n 10 bash ${REPO_ROOT}/src/evaluation/run/run_member_window_scoring.sh > "${LOGS}/driver_member_window_scoring.log" 2>&1 &
nice -n 10 bash ${REPO_ROOT}/src/evaluation/run/run_graph_window_scoring.sh > "${LOGS}/driver_graph_window_scoring.log" 2>&1 &
wait
echo "[master] phase C done $(date -Is)"

echo "[master] phase D, scoring and summaries"
step inventory "${PY}" -u ${REPO_ROOT}/src/evaluation/structure_and_stage_inventory.py
step score_event_conditioned "${PY}" -u ${REPO_ROOT}/src/evaluation/score_fixed_leads.py
step score_event_conditioned_hrrr "${PY}" -u ${REPO_ROOT}/src/evaluation/score_fixed_leads.py --pattern "*_hrrr_jan_aug.npz" --tag _hrrr
step bootstrap_graph_and_cell "${PY}" -u ${REPO_ROOT}/src/evaluation/bootstrap_graph_and_cell_effects.py
step graph_effect_quality "${PY}" -u ${REPO_ROOT}/src/evaluation/graph_effect_vs_rainfall_quality.py
step event_hydrographs "${PY}" -u ${REPO_ROOT}/src/figures/figC2_C3_event_hydrographs.py
step event_rainfall_skill "${PY}" -u ${REPO_ROOT}/src/evaluation/rainfall_forcing_skill.py
step event_fig13 "${PY}" -u ${REPO_ROOT}/src/evaluation/event_only_rainfall_skill_figure.py
step event_fig14 "${PY}" -u ${REPO_ROOT}/src/evaluation/event_only_crest_error_figure.py
step event_definition_comparison "${PY}" -u ${REPO_ROOT}/src/evaluation/event_definition_overlap.py
step event_network_levels "${PY}" -u ${REPO_ROOT}/src/evaluation/gauge_network_skill.py
step event_fig16 "${PY}" -u ${REPO_ROOT}/src/evaluation/event_only_gauge_count_figure.py
step lstm_vs_stgnn "${PY}" -u ${REPO_ROOT}/src/evaluation/lstm_vs_stgnn_by_forcing.py
step score_ens_jan_aug "${PY}" -u ${REPO_ROOT}/src/evaluation/score_fixed_leads.py --runs-dir ../runs_ens_jan_aug --pattern "*_jan_aug.npz" --tag _ens_jan_aug
step score_members_jan_aug "${PY}" -u ${REPO_ROOT}/src/evaluation/score_fixed_leads.py --runs-dir ../runs_members_jan_aug --pattern "*_jan_aug.npz" --tag _members_jan_aug
step score_graphs_jan_aug "${PY}" -u ${REPO_ROOT}/src/evaluation/score_fixed_leads.py --runs-dir ../runs_graphs_all_links --pattern "*_jan_aug.npz" --tag _graphs_jan_aug
step network_levels_hrrr "${PY}" -u ${REPO_ROOT}/src/evaluation/gauge_network_skill.py --ladder-dir ../runs_ladder_hrrr --forcing hrrr --tag _hrrr
step size_effect "${PY}" -u ${REPO_ROOT}/src/evaluation/network_size_effect.py
step size_effect_hrrr "${PY}" -u ${REPO_ROOT}/src/evaluation/network_size_effect.py --levels ../outputs/event_network_levels_h24_hrrr.csv --tag _hrrr
step score_dropout "${PY}" -u ${REPO_ROOT}/src/evaluation/score_fixed_leads.py --runs-dir ../runs_dropout_jan_aug --pattern "*_jan_aug.npz" --tag _dropout
step dropout_effects "${PY}" -u dropout_scenario_effects.py
step crest_jan_aug "${PY}" -u ${REPO_ROOT}/src/evaluation/rainfall_forcing_skill.py --scored-root ../scored_jan_aug --tag _jan_aug
step crest_jan_aug_members "${PY}" -u ${REPO_ROOT}/src/evaluation/rainfall_forcing_skill.py --scored-root ../scored_jan_aug --members-root ../scored_members_jan_aug --tag _jan_aug_members
step crest_jan_aug_graphs "${PY}" -u ${REPO_ROOT}/src/evaluation/rainfall_forcing_skill.py --scored-root ../scored_jan_aug --graphs-root ../scored_graphs_jan_aug --tag _jan_aug_graphs
step false_peaks "${PY}" -u ${REPO_ROOT}/src/evaluation/false_peaks.py
step rise_strata "${PY}" -u ${REPO_ROOT}/src/evaluation/skill_by_rise_size.py
step section42_numbers "${PY}" -u ${REPO_ROOT}/src/evaluation/section42_network_numbers.py
step figure_inputs "${PY}" -u ${REPO_ROOT}/src/evaluation/figure_inputs.py
step fig13_fig14_inputs "${PY}" -u ${REPO_ROOT}/src/evaluation/rainfall_and_crest_figure_inputs.py
step fig16_decision "${PY}" -u ${REPO_ROOT}/src/evaluation/gauge_count_figure_inputs.py
step appendix_inputs "${PY}" -u ${REPO_ROOT}/src/evaluation/appendix_figure_inputs.py
step summarize_hrrr "${PY}" -u ${REPO_ROOT}/src/evaluation/summarize_hrrr_rain_runs.py
step summarize_replay_postproc "${PY}" -u ${REPO_ROOT}/src/evaluation/summarize_simulated_forecast.py
step argument_figures "${PY}" -u ${REPO_ROOT}/src/evaluation/summary_figures.py

echo "[master] failed steps:${FAILED:- none}"
echo "ANCHORFIX_ALL_DONE $(date -Is)"
