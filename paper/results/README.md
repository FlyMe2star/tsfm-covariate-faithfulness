# Paper-result provenance

The populated primary P1-SHAPE table is derived only from:

- `evidence/p1_shape/results/p1_shape_decision.json`;
- `evidence/p1_shape/results/complete_cell_matrix.csv`; and
- the frozen configuration and scientific-code hashes recorded in those artifacts.

The decision JSON has SHA-256
`077322e808fffe62af99feda3363220670eac96ba3f43eb6ef0758da2a7373a7`.
The primary table rounds estimates to three decimals for display; the CSV retains the
archived precision.

The construct-validation table is derived from
`evidence/p1_shape/construct/metric_construct_validation.yaml`. It reports selected
deterministic analytic fixtures only; the archived construct report remains the
authoritative record for all 11 blocking checks and the 768-scenario generator
preflight.

The separate fitted-reference table is derived from
`evidence/p1_shape/references/p1_reference_report.json`, whose archived SHA-256 is
`c3ed3ebfb62a36413b59094d1bf158bc0b78f55e3a4038d2beadead48e2f543c`.
Its source run contains 12 verified units, 1,536 series-model records, and eight
summary cells. The table is explicitly post-primary and descriptive and must never be
used to recompute or reinterpret the frozen P1 gate.
