# Reproducing the paired h+24 RMSE analysis

## Inputs

Extract the companion evaluation data and pass the directory containing `bootstrap_inputs/` as `--data`. The required files are:

- `bootstrap_inputs/comparisons.csv`: comparison names, baseline and candidate residual filenames, origin subsets and sampling designs.
- `bootstrap_inputs/origins_and_events.npz`: `origins_utc` and `event_origin` arrays.
- `bootstrap_inputs/network_units.csv`: physical-gauge identifiers and scoring-unit indices.
- The residual `.npz` files named in `comparisons.csv`: `squared_error_ft2`, `valid` and `units` arrays. Error and validity arrays have seed, origin and scoring-unit dimensions.

## Run

```bash
python src/evaluation/reproduce_paper_resampling.py \
  --data /absolute/path/to/evaluation-data \
  --output results/paired_resampling.csv
```

The default output is `results/paired_resampling.csv` under the current working directory. Reference evaluation data are left unchanged unless you explicitly choose an output path there. The original `reproduce_paper_bootstrap.py` entry point remains available and uses the same calculation.

## What the calculation does

The analysis retains 5,832 origins, 927 event-only origins, 10,000 repeated samples, 72-hour origin blocks and random seed 20260930. It samples blocks and gauges with replacement, using the same selections for the baseline and candidate. Gauge-network comparisons sample physical-gauge clusters so repeated scoring positions for one gauge remain together. Seeds are averaged before taking the median across gauges. Identical GRU/identity comparisons reuse the same gauge draws while retaining the original random-stream sequence for other comparisons.

For every comparison, `difference_m` is candidate RMSE minus baseline RMSE. A positive value means the candidate has higher error. `ci_low_m` and `ci_high_m` are the 2.5th and 97.5th percentiles of the repeated differences. An interval spanning zero contains repeated samples favouring either configuration, so this analysis does not establish a statistically significant difference at the nominal two-sided 5% level. It does not establish that the two configurations are equivalent. The script does not calculate a p-value.

The existing output names `bootstrap_median_m` and `interval_excludes_zero` are retained for compatibility with the deposited numerical results. Bootstrap is the statistical name for sampling the observed data repeatedly with replacement.

## Other results

The original scoring, model and figure scripts remain under `src/`. Their project paths require the corresponding paired predictions, graph inputs or original study assets. `src/evaluation/run/run_all.sh` describes the original evaluation sequence; it is not a standalone download-and-run command for a fresh checkout. See [installation](installation.md) and the [figure and table guide](figures_and_tables.md).
