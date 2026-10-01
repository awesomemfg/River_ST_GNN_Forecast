# Chronological training and frozen evaluation

All three architectures use the same pre-2026 training matrix, nine historical and nine forcing channels, stage-change targets, pinball loss, Adam optimizer, batch size 16, and seeds 101, 202 and 303. Training rainfall is observed.

## Epoch selection

Training origins: 4 January 2023 00:00 to 30 December 2024 23:00 UTC (17,448 origins). Validation origins: 4 January 2025 00:00 to 30 December 2025 23:00 UTC (8,664 origins). Each origin uses the preceding 288 fifteen-minute readings and targets the next 96. These origin limits prevent overlapping training targets and validation histories. Graph construction and normalization use only data through 2024 during epoch selection. Early stopping uses validation pinball loss, patience eight, with a maximum of 40 epochs.

Only the selected epoch count is retained. Final weights are fitted from a fresh initialization on 26,208 origins, 4 January 2023 00:00 to 30 December 2025 23:00 UTC. Graph construction and normalization then use data through 31 December 2025. No 2026 target participates in either stage.

The LSTM final epoch counts are 19, 12 and 21 for seeds 101, 202 and 303. The corrected evaluated weights are listed in `weights/MODEL_LINEAGE.csv`. The earlier random-validation LSTM trainer is preserved only under `archive/model_lineage/`.

## Models and inputs

The final ST-GNN keeps 68 nodes and the 471-edge observation graph. The GRU uses the same network with identity adjacency. The shared nodewise LSTM passes no information between gauges. The 51 in-parish gauges are the scoring mask, not a replacement graph.

Historical channels: standardized stage, rain6, rain24, API24, API48, API72, stage tendency, rainfall-wetness interaction, rainfall acceleration. Future channels replace stage and tendency with hour-of-day sine and cosine. Their definitions are identical across architectures.

Hard invariants:
- rain6 accumulates six fifteen-minute steps, or 1.5 h.
- rain24 accumulates 24 fifteen-minute steps, or 6 h.
- The decoder produces 96 fifteen-minute steps, or 24 h.
- The causal flatline mask uses the preceding 96 readings.

Five quantiles, 0.5, 0.6, 0.7, 0.8 and 0.9, are fitted. The paper scores P80.

## Frozen January to August evaluation

All 5,832 hourly origins from 1 January through 31 August 2026 are evaluated without changing weights. The observed-rainfall hindcast has postprocessing off. The simulated operational forecast uses archived issue-time HRRR rainfall and the identical continuity blend, stale-data handling and rainfall-dependent rise cap for all trained architectures. Raw HRRR runs are also retained for the forcing analysis. All 96 steps remain in the reevaluation archives.

Event-only means at least one parish pump was running at the issue time, giving 927 origins. Scores are calculated per gauge and seed, averaged over seeds, then summarized by the median across gauges. The paired uncertainty statistic is the h+24 RMSE difference, with 10,000 bootstrap replicates and 72-hour origin blocks. Gauge-network comparisons use physical-gauge clusters.

Ten graph nodes lacked sufficient development-stage observations. Their stage inputs and direct target loss were masked; nine later-supported in-parish gauges enter the final fit. Development directly supports 58 of 68 nodes and 42 of 51 score gauges; final fitting supports 67 of 68 nodes and all 51 score gauges. The full node order remains unchanged.

## Relation to Liu et al. (2025)

The protocol adopts chronological training, validation and testing, validation-based configuration, an established LSTM baseline, and hydrologic skill metrics. It does not reproduce Liu's datasets, dates or daily forecast experiment. The gap and final refit are this study's explicit choices, not a claim that Liu prescribes this exact sequence.

## Release limits

Record A supplies paired evaluation data and full-precision residual inputs for the bootstrap. It does not supply the complete raw training matrix. Retraining requires the original telemetry, routing assets and meteorological inputs.
