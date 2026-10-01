# Installation

## Standalone resampling analysis

Use Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements_resampling.txt
python src/evaluation/reproduce_paper_resampling.py --help
```

NumPy 2.2.6 is pinned to the original evaluation version. This entry point uses only NumPy and the Python standard library. See [reproducing results](reproducing_results.md) for the companion inputs.

## Original training and evaluation environment

The original evaluation environment used Linux, Python 3.13.9, TensorFlow 2.20.0, NumPy 2.2.6 and pandas 2.3.0. `environment/environment_evaluation.yml` and `environment/requirements_evaluation.txt` preserve that environment. The pip snapshot includes local Conda build paths and is not a portable installation requirements file. On a compatible Linux machine, the Conda snapshot is a starting point:

```bash
conda env create --name river-st-gnn --file environment/environment_evaluation.yml
```

Training used the cluster's TensorFlow 2.16.1 GPU container. Reproducing the original training also requires the full training matrix and the data layout described in the scripts. Creating an environment alone does not supply those inputs. HPC account and server placeholders must be configured for your own infrastructure before submitting any job.

Run original scripts from the repository root. Their `project/` paths refer to study inputs and outputs that are supplied or prepared separately. Private parish hydraulic-model and terrain assets are not part of the release.
