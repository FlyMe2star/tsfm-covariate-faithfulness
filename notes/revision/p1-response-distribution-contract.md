# Post-primary response-distribution diagnostic

Purpose: address the limitation of the three illustrative response curves without
changing the frozen P1 decision, thresholds, selected representatives, or figure.
This analysis was specified after seeing P1/P2 results and is descriptive only.

## Source and unit

- Read all 24 private P1 full-screening units: two backbones, four mechanisms,
  three generator seeds, and 64 paired series per seed (192 per cell).
- Verify every unit manifest, config/scientific-code hash, array SHA-256, completion
  status, array shape, finite value, and series-ID uniqueness. Recompute the
  archived per-series RGR and timing diagnostics from raw responses.
- Reconcile each cell's frozen DSA mean and RGR, D1, and G estimates with the
  public P1 decision before writing any new output. The frozen point estimator
  averages three within-seed means for DSA or three within-seed medians for RGR,
  D1, and G. It is not generally the pooled 192-series median.
- The independent descriptive unit is the paired synthetic series; generator seed
  is recorded as a stratum. No p-value, new confidence interval, or new gate is
  computed for this post-primary diagnostic.

## Complete-cell descriptive outputs

For **all eight cells**, report the pooled 192-series median, quartiles, and
90th percentile of:

1. archived per-series RGR on oracle-active support;
2. unnormalized full-horizon L1 gain ratio, `sum(abs(predicted)) /
   sum(abs(oracle))`, which is not the registered RGR;
3. absolute response-onset error in forecast steps, using the registered 10% of
   oracle-peak definition;
4. absolute peak-time error in forecast steps;
5. archived normalized absolute-mass transport distance.

Also report the number of series with onset or peak error of at most one step,
the number with no detected predicted onset, and the number with negligible
predicted response mass for peak detection. These are descriptive counts, not
pass/fail thresholds. Retain every series, including tails and nonpassing cells.
The registered no-onset error is 1.0 on the normalized 24-step horizon; after
conversion it appears as a 23-step sentinel, with its count reported separately.
The registered peak error uses the same sentinel for negligible predicted mass.
For the three already frozen representative series, report empirical percentile
positions within their full cells without reselecting examples.

## Interpretation and output boundary

The output is a source-ID-free CSV and a provenance-rich JSON stored outside the
private source archive. No forecast checkpoint is loaded or called. The diagnostic
cannot revise the P1 paper gate or turn the three-example plot into population
evidence. Manuscript numbers or a new supplementary figure may be added only after
the complete output is available, checked, and labeled post-primary.
