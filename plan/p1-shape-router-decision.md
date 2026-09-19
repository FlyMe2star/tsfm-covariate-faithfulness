# P1-SHAPE router decision

Route: **empirical evaluation paper**, with construct validation before checkpoint
inference.

## Why this route remains viable

- It asks a new, function-level question rather than weakening the failed P0 gate.
- Structural replay supplies exact response trajectories and removes the need for human
  annotation.
- The two verified model adapters and Colab execution path are reusable engineering,
  while P0 scientific outputs remain excluded from design choices.
- Four new mechanisms cover sign transition, timing dispersion, path dependence, and
  interaction, which is stronger than adding more variants of a linear response.
- The gate supports a benchmark paper without requiring a learned mitigation.

## Ordered work plan

1. Implement signed block aggregation, shape metrics, and analytic fixtures.
2. Implement and CPU-preflight the four untouched mechanisms.
3. Produce a construct-validation report and finalize numeric bounds using only fixture
   evidence.
4. Refresh the closest-work search and pin checkpoint revisions and licenses.
5. Ask the owner for `APPROVE P1-SHAPE FREEZE`.
6. Freeze config, code hash, dependency lock, and inference manifests.
7. Run one-cell-per-mechanism smoke on T4 without decision code.
8. Run the complete two-backbone matrix, resumably; use A100 only if T4 throughput is
   inadequate.
9. Compute the frozen decision once and archive all cells before drafting result claims.

## Stop conditions

- Construct fixtures do not show metric selectivity or stable severity ordering.
- A mechanism cannot guarantee paired replay or nondegenerate oracle effects.
- A checkpoint interface cannot be revision-pinned or reproducibly invoked.
- After freeze, the minimum cell, diversity, or accuracy-complement rule fails.

No stop condition permits post-hoc threshold relaxation.
