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

P2 robustness material is archived verbatim under
`evidence/p2_robustness/results/`. Its report records configuration SHA-256
`7f7c27bc37ffc57ce3b2455877dab2a72d7d0d3255484042377cdaf90c27a284`
and scientific-code SHA-256
`7178cf9bfa72d63a261765fdad333a30abd34ee0014c8bb082d160e6e7eebbe4`.
The report checks all 24 primary source-unit hashes and the immutable P1 decision
SHA-256 above. P2 performed no model inference and did not modify the primary
arrays, registered metrics, thresholds, or decision.

The WQL column comes from `wql_cell_summary.csv`; the fixed-fine-normalization
figure from `monotone_cell_summary.csv`; the threshold map from
`threshold_sensitivity.csv`; the response-path figure from
`representative_responses.json`; and Table IV from
`metric_fixture_comparison.csv`. All are post-primary descriptive analyses.
`paper/figures/plot_p2.py` verifies each archived artifact hash and count against
the report before regenerating the vector PDF/SVG figures. The original P1 pass
column is unchanged. The P2 ZIP supplied by the author is retained privately
outside Git; these small derived artifacts contain no licensed dataset images.
