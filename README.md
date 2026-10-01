# Record B: code and trained weights for "Operational ST-GNN stage forecasting, Ascension Parish" (HESS submission)

Licence: Apache-2.0 (`LICENSE`, `NOTICE`), including the trained weights. DOI: 10.5281/zenodo.23063618. Data: Record A, DOI 10.5281/zenodo.23046776.

This record contains the workflow behind the paper:
- data preparation;
- graph construction;
- the nine-channel ST-GNN, the GRU (the ST-GNN on the identity graph) and the LSTM;
- chronological training;
- inference with the Sect. 2.5.2 postprocessing;
- evaluation;
- the scripts that produce each table and figure.


## Layout

| Path | Contents |
|---|---|
| `src/data/` | Download and cleaning of the gauge, rainfall, wind and structure records (APG SCADA, USGS, NOAA CO-OPS, ASOS), datum correction and the feature matrix (`download_and_prepare_matrix.py`); gauge cohort; in-parish gauge list; station-to-rain-gauge table |
| `src/graphs/` | The six graphs of Table 3 and the Sect. 4.2 networks. Scripts: `observation_graph_2023_2024.py` (epoch-selection graph), `observation_graphs_pre2026.py` (the 471-edge graph, correlation of at least 0.40 and lag of at least 30 min, and its within-basin variant), `hecras_graph.py`, `dem_graph.py`, `dem_within_basin_graph.py`, `identity_graph.py`, `network_subset_graphs.py`. `freeze_*.py` records every graph with its SHA-256. |
| `src/stgnn/` | `stgnn_models.py` (ST-GNN), `train.py` (epoch selection on 2025, then a fresh fit on all pre-2026 origins), `train_network_subsets.py`, `hindcast.py`, `simulated_operational_forecast.py` (continuity blend, stale-data repair, rainfall-dependent rise cap), validation checks, and `slurm/` job scripts |
| `src/lstm/` | `lstm_models.py`, `train.py`, `inference.py`, `slurm/` |
| `src/forcing/` | HRRR issue-time rainfall (`hrrr_fetch_points.py`, `hrrr_issue_time_forcing.py`); weather-model and GEFS rainfall (`weather_model_fetch.py`, `gefs_fetch.py`, `gefs_probability_matched_mean.py`, `weather_model_forcing_jan_aug.py`, `weather_model_issue_layout.py`) |
| `src/evaluation/` | `score_fixed_leads.py` (RMSE, correlation, NSE, KGE at fixed leads, all and event-only origins), `score_forecast_windows.py` (24 h windows, crests), the analyses behind Sects. 3 and 4, and `run/` (the drivers; `run/run_all.sh` runs the full evaluation in order) |
| `src/figures/` | One script per paper figure (`fig01_study_area.py` to `fig15_gauge_networks.py`, `figA1_...`, `figB1_B2_C1_appendix.py`, `figC2_C3_event_hydrographs.py`), the shared plotting recipes they call (`recipe_*.py`) and helpers (`_mapbase.py`, `_units.py`, `_card.py`, `_spatialmap.py`); `run_all_figures.sh` |
| `src/tables/` | `table4_hindcast.py`, `tables_A1_B1_B2.py`, and the scripts for the numbers quoted in the text |
| `config/` | `rain_rise_caps.json` (rise-cap parameters of the postprocessing), `station_coordinates.csv` |
| `docs/` | Training protocol, its locked settings, and the protocol for the alternative graphs |
| `weights/` | Final weights (`gnn.weights.h5`), metadata (`gnn_meta.json`: node order, normalization statistics, graph hash, training period) and training history for 20 models × 3 seeds; `WEIGHTS_INDEX.csv` gives each file's SHA-256 |
| `environment/` | The evaluation environment (`environment_evaluation.yml`, `requirements_evaluation.txt`) |
| `archive/` | Superseded versions (the evaluators before the anchor fix), one-off generators that wrote copies now in `src/`, run helpers, and earlier trainers the models derive from. Kept for provenance; not needed to reproduce the results. |
| `PROVENANCE.csv` | Every file with its component and the SHA-256 of the file as it was run |
| `SCRUB_REPORT.csv` | Every place where a server or HPC account name was replaced by a placeholder |
| `FILE_MANIFEST.csv` | Size and SHA-256 of every file in this record |

## Figure and table map (manuscript numbering)

| Item | Script |
|---|---|
| Fig. 1 | `src/figures/fig01_study_area.py` (map layers: `fig01_fetch_map_layers.py`) |
| Fig. 2, 3, 4, 5 | `src/figures/fig02_operational_lifecycle.py`, `fig03_candidate_graphs.py`, `fig04_workflow.py`, `fig05_lstm_step.py` |
| Fig. 6 | Screenshot of the public forecast interface; no script |
| Fig. 7 | `src/figures/fig07_selected_graph.py` |
| Fig. 8, 9, 10 | `src/figures/fig08_hindcast_skill.py`, `fig09_hindcast_examples.py`, `fig10_simulated_forecast_skill.py` |
| Fig. 11 | `src/figures/fig11_event_only_model_comparison.py`, then `fig11_event_only_labels.py` |
| Fig. 12 | `src/figures/fig12_simulated_forecast_examples.py` |
| Fig. 13, 14, 16 | `src/figures/fig13_fig14_fig16_rainfall_crest_gauge_count.py` |
| Fig. 15 | `src/figures/fig15_gauge_networks.py` |
| Fig. A1 | `src/figures/figA1_terrain_and_mesh.py` |
| Fig. B1, B2, C1 | `src/figures/figB1_B2_C1_appendix.py` |
| Fig. C2, C3 | `src/figures/figC2_C3_event_hydrographs.py` |
| Table 3 | `src/evaluation/score_fixed_leads.py` on the six graph runs (`run/run_alternative_graphs.sh`) |
| Table 4 | `src/tables/table4_hindcast.py` |
| Table 5, Sect. 3.2.2 | `src/evaluation/score_fixed_leads.py`, event-only simulated operational forecasts |
| Table 6, Sect. 3.3 | `src/tables/section41_rainfall_numbers.py`, from `src/evaluation/rainfall_forcing_skill.py` |
| Tables A1, B1, B2 | `src/tables/tables_A1_B1_B2.py` |

The published Figs. 1 to 5, 7 and A1 were checked against these scripts' outputs (byte-identical or pixel-identical).

## Paths

- **Run everything from the repository root.** References between files of this repository are written as repository paths: `src/...` in Python and `${REPO_ROOT}/src/...` in shell scripts. Each shell script sets `REPO_ROOT` from its own location.
- **Data paths use a neutral project root.** The scripts read and write data under `project/` (for example `project/Experiments/<experiment>/runs`), and `project/hpc/` is the storage of the HPC cluster where training ran. These folders are not part of the deposit. Record A holds the data needed to check every published number.
- **Provenance and corrections.** The corrected LSTM uses the shared chronological trainer; its earlier random-validation trainer is retained only in `archive/`. `weights/MODEL_LINEAGE.csv` binds the evaluated LSTM to its three checkpoints. `src/evaluation/reproduce_paper_bootstrap.py` reproduces the paired h+24 RMSE analysis from Record A without TensorFlow and reuses one result for identical GRU/identity inputs.
- **Station coordinates.** The rainfall fetchers in `src/forcing/` read the station coordinates from the parish's operational downloader, which is not included. The same coordinates are in `config/station_coordinates.csv`.

## Not included

- **Operational credentials and server addresses.** They are replaced by `<SERVER>`, `<HPC_HOST>` and `<HPC_ALLOCATION>`, as listed in `SCRUB_REPORT.csv`.

- **The Ascension Parish Government HEC-RAS model files.** These are internal Parish data on the Parish SharePoint. They are read by:
  - `src/graphs/hecras_graph.py` (simulated water surface);
  - `src/graphs/dem_graph.py` (terrain raster; `dem_within_basin_graph.py` derives from that graph);
  - `src/figures/fig01_study_area.py` (terrain raster);
  - `src/figures/figA1_terrain_and_mesh.py` (terrain raster and mesh geometry).

  The graphs they produce are in Record A.
- **Raw training telemetry.** Record A contains paired forecasts and observations for scoring, not the complete feature-engineered training matrix. It supports reproducing scores and paired uncertainty calculations. Retraining and rainfall-forcing reconstruction require the original telemetry and issue-time rainfall inputs described by the scripts.

## Environment

- **Evaluation:** Linux with Python 3.13, TensorFlow 2.20, NumPy 2.2 and pandas 2.3 (`environment/`).
- **Training:** on the LSU HPC cluster, one GPU, in the cluster's TensorFlow 2.16.1 container.
- **Weights:** written by Keras `save_weights` (`src/stgnn/train.py`).
