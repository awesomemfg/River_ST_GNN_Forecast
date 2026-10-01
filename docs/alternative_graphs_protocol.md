# System B matched comparison protocol lock

Locked before submission of the matched Section 3.1 graph and baseline jobs.

## Mandatory arms

1. Observation-derived lead-lag ST-GNN.
2. Basin-restricted observation-derived ST-GNN.
3. HEC-RAS connectivity ST-GNN.
4. DEM downslope connectivity ST-GNN.
5. Basin-restricted DEM connectivity ST-GNN.
6. Identity-adjacency ST-GNN.
7. Shared nodewise unidirectional LSTM.
8. Persistence.

The first observation-derived ST-GNN and persistence results already exist. All remaining mandatory learned arms will be fitted now.

## Matched controls

Every learned arm uses:

- the same 68-node order;
- the same nine historical and nine forcing channels;
- the same 72-hour encoder history;
- the same 96-step, 24-hour decoder trajectory;
- the same five training quantiles and P80 scored output;
- the same pinball loss, optimizer, batch size, and early-stopping rule;
- seeds 101, 202, and 303;
- chronological epoch selection using 2023 through 2024 for fitting and 2025 for validation;
- a fresh final fit using all eligible pre-2026 origins for the selected epoch count;
- the same 4,344 hourly test origins from January through June 2026;
- the same fixed 51-gauge in-parish scoring mask;
- raw predictions with postprocessing disabled;
- observed-rain and archived issue-time HRRR evaluation arms.

For each ST-GNN comparison, adjacency is the intended independent variable. The nodewise LSTM receives the identical per-node tensors and loss but exchanges no information between gauges.

## Graph assets

The observation-derived graphs use cutoff-specific assets. The unrestricted graph has 332 edges for epoch selection and 471 edges for final fitting. The basin-restricted observation graph has 196 and 279 edges, respectively.

HEC-RAS, DEM, basin-restricted DEM, and identity graphs are fixed definitions that do not use the validation or test observations for discovery. Their epoch-selection and final-fit copies are byte-identical.

Exact paths, SHA256 hashes, node order, and edge counts are recorded in `frozen_assets/comparisons/alternative_graph_manifest.json`.

## Bi-LSTM decision

The legacy Bi-LSTM is not part of the mandatory matched comparison because its historical multi-output training protocol differs from System B. A new nodewise Bi-LSTM may be added later as a secondary architecture sensitivity, but it must not replace the nodewise unidirectional LSTM or be described as the historical operational Bi-LSTM unless it exactly reproduces that architecture and data flow.

## Manuscript interpretation

The graph-selection result must be based on matched System B arms only. The old random-validation graph ranking may be discussed as superseded provenance but must not supply the primary Section 3.1 ranking.

The identity arm answers whether cross-gauge message passing adds value. The nodewise LSTM answers whether the graph-recurrent architecture improves on an established sequence-model baseline under the same data protocol.
