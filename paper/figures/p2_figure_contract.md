# P2 figure contract

Status: post-primary robustness and visualization support. Source artifacts are
hash-verified under `evidence/p2_robustness/results/`.

## Figure A: representative response paths

- Results question: What does a fine response-shape mismatch look like in the three
  registered passing cells, and what remains after width-8 blocking?
- Claim: the frozen deterministic representatives have visible fine-path differences
  even though their separately normalized width-8 profiles are closer.
- Archetype: quantitative grid, double-column (183 mm).
- Panels: three columns are the three registered passing cells; upper row shows
  signed, L1-normalized 24-step oracle and model paths; lower row shows separately
  normalized signed sums for three width-8 blocks.
- Evidence hierarchy: the upper trajectories are the hero evidence; lower bars show
  the registered coarse-resolution comparison. Report each representative's D1 and
  registered G in the column heading or caption.
- Source: `representative_responses.json`, selected by the frozen MAD rule, one
  series per passing cell. No uncertainty interval is appropriate for a single
  representative; the aggregate cell uncertainty remains in Table II.
- Reviewer risks: a chosen example could look hand-picked; the deterministic rule
  and all three passing cells must be visible. Never call these paths an average.

## Figure B: fixed-fine resolution curves

- Results question: Does the supplementary fixed-fine-normalization diagnostic
  decrease as width grows for every tested cell?
- Claim: all eight cell medians decrease from widths 1 to 8, while the contraction
  gap remains positive for every cell.
- Archetype: quantitative grid, double-column (183 mm).
- Panels: left Chronos-2, right TimesFM-3; four mechanism curves per backbone with
  paired-series stratified-bootstrap 95% confidence intervals at every width.
- Evidence hierarchy: complete mechanism coverage and intervals; the three primary
  passing cells are marked by solid rather than dashed lines, without hiding others.
- Source: `monotone_cell_summary.csv`, 192 series per cell and 5,000 bootstrap
  replicates. The analytic contraction proof is separate from these empirical values.
- Reviewer risks: do not use the registered G values for this curve or imply that
  the new metric replaced the frozen P1 endpoint.

## Figure C: registered-threshold sensitivity

- Results question: Across the declared D1 and G threshold neighborhood, where
  does the original paper-level condition continue to hold?
- Claim: the outcome is stable over the tested D1 thresholds and depends more
  sharply on the registered G lower bound.
- Archetype: compact quantitative grid, single-column (89 mm).
- Panel: vector cells encode the number of passing cells, with the original
  `(D1=0.08, G=0.03)` position outlined. The caption states that 28/56 grid
  points pass and that this descriptive scan cannot retrospectively validate
  the original threshold choice.
- Source: `threshold_sensitivity.csv`, all 56 grid points.
- Reviewer risks: avoid a red-green-only pass palette and avoid calling all
  56 grid points independent experiments.

## Table IV: analytic metric comparison

- Results question: What distinct failures do common pointwise and alignment
  diagnostics detect on the frozen analytic fixtures?
- Form: a small raw-value table is more faithful than color-normalizing metrics
  with different natural scales. The full 8-fixture matrix remains in the public CSV.
- Source: `metric_fixture_comparison.csv`.
- Reviewer risks: do not claim universal superiority from eight constructed
  examples. DTW's zero cost for temporal shifts and transport's zero for a
  sign flip illustrate different invariances rather than broken implementations.

All figures use the approved Matplotlib style in `paper/figures/STYLE.md`, export
PDF and editable SVG, and require final-size font, alignment, collision, and
panel-level visual QA.
