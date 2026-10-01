# Figure and table guide

The figure numbering below follows the current manuscript. Original recipe filenames retain their earlier numbers. The recipes and scientific figures have not been redrawn or simplified. Figure scripts are under `src/figures/`.

| Figure | Recipe |
| --- | --- |
| 1 | fig01_study_area.py |
| 2 | fig02_operational_lifecycle.py |
| 3 | fig03_candidate_graphs.py |
| 4 | fig07_selected_graph.py |
| 5 | fig04_workflow.py |
| 6 | fig05_lstm_step.py |
| 7 | Public forecast-interface screenshot; no script |
| 8 | fig08_hindcast_skill.py |
| 9 | fig09_hindcast_examples.py |
| 10 | fig15_gauge_networks.py |
| 11, 12, 13 | fig13_fig14_fig16_rainfall_crest_gauge_count.py |
| 14 | fig10_simulated_forecast_skill.py |
| 15 | fig11_event_only_model_comparison.py, then fig11_event_only_labels.py |
| 16 | fig12_simulated_forecast_examples.py |
| A1 | figA1_terrain_and_mesh.py |
| B1, B2, C1 | figB1_B2_C1_appendix.py |
| C2, C3 | figC2_C3_event_hydrographs.py |

| Table | Calculation |
| --- | --- |
| 3: hindcast metrics | `src/tables/table4_hindcast.py` |
| 4: rainfall-forcing comparison | `src/tables/section41_rainfall_numbers.py`, using `src/evaluation/rainfall_forcing_skill.py` |
| 5: event-only simulated operational forecasts | `src/evaluation/score_fixed_leads.py` |
| A1, B1, B2 | `src/tables/tables_A1_B1_B2.py` |
| A2: graph comparisons | `src/evaluation/score_fixed_leads.py`, with `src/evaluation/run/run_alternative_graphs.sh` |

Tables describing inputs and metric definitions do not require a numerical generator. Plotting scripts require the original inputs and project layout. Run them from the repository root.
