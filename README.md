# River ST-GNN Forecast

Research code and trained weights for river-stage forecasting in Ascension Parish, Louisiana. The study compares a spatiotemporal graph neural network (ST-GNN), a GRU with identity adjacency, and a nodewise LSTM. The models use 72 hours of historical inputs to predict the next 24 hours at 15-minute intervals. The full network contains 68 gauges, including 51 within the parish and 17 outside its boundary.

## Quick start: reproduce the paired uncertainty analysis

The standalone resampling analysis needs Python 3.10 or later and NumPy. It does not require TensorFlow, a GPU, operational credentials, or raw training telemetry.

```bash
git clone https://github.com/awesomemfg/River_ST_GNN_Forecast.git
cd River_ST_GNN_Forecast
python -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements_resampling.txt
python src/evaluation/reproduce_paper_resampling.py --help
python src/evaluation/reproduce_paper_resampling.py \
  --data /absolute/path/to/evaluation-data \
  --output results/paired_resampling.csv
```

Set `--data` to the extracted evaluation-data directory containing `bootstrap_inputs/`. The script repeats the paired comparison 10,000 times, sampling gauges and 72-hour blocks of forecast origins with replacement. Each repeat uses the same sampled observations for both configurations. The middle 95% of the resulting RMSE differences gives the reported interval. See [reproducing results](docs/reproducing_results.md) for the inputs and output fields.

Archive identifiers: [evaluation data and derived graphs](https://doi.org/10.5281/zenodo.23046776) and [software and trained weights](https://doi.org/10.5281/zenodo.23063618). The software and weights use Apache-2.0; the evaluation data use CC BY 4.0. Archive availability is managed separately from this repository.

## Repository contents

| Directory or file | Contents |
| --- | --- |
| `src/data/` | Gauge, rainfall, wind and structure-record preparation, datum correction, and feature-matrix construction |
| `src/graphs/` | Observation, hydraulic, terrain, within-basin and identity graphs, network subsets, and graph checksums |
| `src/stgnn/` | ST-GNN and GRU models, chronological training, hindcasts, simulated operational forecasts, and HPC job scripts |
| `src/lstm/` | Nodewise LSTM model, training and inference |
| `src/forcing/` | Archived issue-time HRRR and other rainfall-forcing preparation |
| `src/evaluation/` | Fixed-lead and forecast-window scoring, paired resampling, and evaluation drivers |
| `src/figures/`, `src/tables/` | Original figure recipes and table calculations; see the [figure and table guide](docs/figures_and_tables.md) |
| `config/` | Rainfall-dependent rise caps and station coordinates |
| `weights/` | Checkpoints, node order, normalization statistics, training histories, model lineage and weight checksums for 20 models and three seeds |
| `environment/` | Minimal resampling requirements and snapshots of the original evaluation environment |
| `docs/` | Installation, reproduction instructions and scientific protocols |
| `archive/` | Earlier scripts retained to document model development and corrections |
| `PROVENANCE.csv` | Source files and their checksums as originally run |
| `SCRUB_REPORT.csv` | Locations where server and account identifiers were replaced |
| `FILE_MANIFEST.csv` | Current file sizes and SHA-256 checksums, excluding the manifest itself and the compressed code bundle |
| `Code_Ascension_Parish_ST_GNN_Model.tar.gz` | Compressed copy of the current repository files, excluding the bundle itself |

## Scientific protocol

Epoch selection uses 2023-2024 training origins and 2025 validation origins. Final weights are fitted from a fresh initialization using all pre-2026 origins, then held fixed throughout the January-August 2026 evaluation. Seeds are 101, 202 and 303. The evaluation contains 5,832 hourly forecast origins, including 927 event-only origins defined by parish pump operation at issue time.

The observed-rainfall hindcast has postprocessing off. Simulated operational forecasts use archived issue-time HRRR rainfall with continuity blending, stale-data handling and rainfall-dependent rise caps. Scores are calculated per gauge and seed, averaged over seeds, and summarized by the median across gauges. See [training protocol](docs/training_protocol.md) and the other protocol documents in `docs/`.

## Running the original workflow

Run scripts from the repository root. Most original training, inference and plotting scripts expect the study's data layout under `project/`, including `project/Experiments/` and HPC storage under `project/hpc/`. Those data directories are not included here. See [installation](docs/installation.md) before using the original environment snapshots.

The companion evaluation data support scoring and uncertainty calculations. They do not contain the complete raw training matrix. Retraining requires the original telemetry and meteorological inputs. The parish's private HEC-RAS model, terrain and mesh files are not distributed; their derived graphs are supplied separately with the evaluation data. Some rainfall downloaders also refer to the parish operational downloader, which is not included; station coordinates are provided in `config/station_coordinates.csv`.

For another watershed, supply its observations, forcing channels and graph, then train and evaluate models for that network. The supplied checkpoints depend on Ascension's gauge order, graph and normalization statistics. This repository preserves the research workflow; it does not provide a general-purpose forecasting service or a complete operational deployment.

## Citation and licence

Use [CITATION.cff](CITATION.cff) for the software citation, and cite the associated paper when available. Code and trained weights are covered by [Apache-2.0](LICENSE), with attribution in [NOTICE](NOTICE).
