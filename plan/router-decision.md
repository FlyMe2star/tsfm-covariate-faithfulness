# Router decision

Route to **empirical-paper-writer** for an empirical benchmark and behavioral-audit paper.

The primary contribution requires new paired intervention experiments, uncertainty
estimation, metric validation, and cross-backbone comparisons. It therefore cannot be
supported as a literature-only review. The default paper is evaluation-first; a new
method is optional rather than required for continuation.

## Why this route is comparatively low-risk

- P0 requires no training and can run on a T4.
- Synthetic mechanisms provide analytic response ground truth.
- The existing FEV, Chronos-2, and TimesFM-3 execution knowledge is reusable without
  reusing the old sealed outcomes.
- A verified benchmark finding can support the paper even if no safeguard is added.
- The gate can fail cheaply before A100-scale expansion.

## Downstream order

1. Design and unit-test generators and metrics without invoking a TSFM.
2. Freeze the P0 configuration and code hash.
3. Run the two-backbone P0 pilot on T4.
4. Archive the complete output before choosing an evaluation-only or safeguard branch.
5. Route verified evidence to the empirical paper drafting and QA workflow.

## Frozen P0 outcome — 2026-09-19

The original route stopped at its predeclared gate. Across six active
backbone-by-mechanism cells, the smallest DSA lower bound was `0.9803`; all RGR
intervals stayed within `[0.50, 1.50]`. The largest placebo PRR upper bound was
`0.0985`, below the frozen `0.20` ceiling. There were therefore zero violating
cells, so the minimum-count and diversity checks failed.

`complementary_accuracy_passed: false` does **not** mean forecast accuracy was poor.
That condition is an existential check over fidelity-violating cells; because the set
of violating cells was empty, it is false by construction. In fact, covariate-aware
relative SQL differences ranged from `-0.7304` to `-0.5293`, indicating lower SQL than
the matched target-only forecasts in every cell.

The failure-centered C1/C3 paper route is closed. The bounds must not be relaxed on
these observations. A defensible continuation requires owner approval of a new,
untouched contract. The recommended adjacent hypothesis is **multi-resolution
response-shape faithfulness**: coarse direction and gain can be correct while nonlinear
response shape and temporal localization remain imperfect. P0 may motivate that
hypothesis, but cannot test it. Any continuation must freeze new metrics and success
criteria before evaluating new mechanisms, parameters, seeds, or backbones.
