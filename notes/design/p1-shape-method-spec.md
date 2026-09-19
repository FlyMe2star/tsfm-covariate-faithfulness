# CovIntervene-SHAPE P1 method specification

Status: **owner-approved design; proposed numeric bounds; no model inference authorized**  
Last updated: 2026-09-19

## 1. Audit object

For series \(i\), structural replay creates factual and intervened future worlds with
identical target history, covariate history, non-intervened future covariates, model
inputs other than the named intervention, and future innovations. The oracle and model
responses are

\[
\Delta^{\star}_{i,h}=y^{(1)}_{i,h}-y^{(0)}_{i,h},\qquad
\widehat{\Delta}_{i,h}=\widehat y^{(1)}_{i,h}-\widehat y^{(0)}_{i,h}.
\]

The paired series is the independent unit. Horizon points are never bootstrapped as
independent observations. Median forecasts define the primary response; quantile-wise
responses are exploratory.

## 2. Untouched structural mechanisms

All use context length 192 and horizon 24. Generator seeds are `101`, `307`, and `911`.
Exact parameter ranges will be machine-locked after CPU preflight and before checkpoint
access. They may not reuse P0 ranges.

### M1: biphasic rebound

A signed covariate impulse enters through a finite response kernel with an early lobe
and an opposite-signed delayed lobe. Lobe amplitudes, separation, and widths are sampled
from frozen ranges while both lobes retain nontrivial oracle mass. The mechanism tests
rebound deletion, phase displacement, and cancellation hidden by aggregate summaries.

### M2: dispersed delayed pulse

A localized future-covariate pulse is transmitted through a positive, unimodal delay
kernel with randomized onset and dispersion. It is not the P0 fixed three-lag kernel.
The mechanism tests onset, peak, and response-width fidelity.

### M3: rate-asymmetric hysteresis

A latent response state updates differently for positive and negative covariate changes
and decays over time before entering the target equation. Rising and falling intervention
paths are matched in total variation but have different oracle trajectories. The
mechanism tests path dependence and asymmetric persistence.

### M4: two-covariate synergy

Two future covariates have main effects and a bounded joint interaction. Four worlds
are replayed with shared innovations: factual, A-only, B-only, and joint. The oracle
interaction response is

\[
\Delta^{\star,\mathrm{int}}=
\Delta^{\star,AB}-\Delta^{\star,A}-\Delta^{\star,B},
\]

with the analogous predicted contrast. Generator assertions reject negligible
interaction effects.

## 3. Coarse eligibility metrics

The P0 definitions of Directional Sign Agreement (DSA) and absolute Response Gain
Ratio (RGR) are retained as coarse eligibility checks, not searched outcomes. Oracle
active support is defined by the frozen 10% of peak absolute response threshold.

A cell is coarse-eligible only when:

- the series-cluster bootstrap two-sided 95% DSA lower bound is at least `0.90`; and
- the complete two-sided 95% interval for median RGR lies within `[0.75, 1.25]`.

These stricter proposed bounds reflect the claim that shape distortion is hidden behind
otherwise credible coarse behavior. Construct validation may change them only before
any model inference and with an explicit versioned rationale.

## 4. Signed multi-resolution shape distance

For block width \(b\in\{1,2,4,8\}\), partition the 24-step horizon into contiguous
blocks and sum signed response inside each block:

\[
B_b(v)_j=\sum_{h\in j\text{th block}} v_h.
\]

Normalize a nonzero blocked response by signed L1 mass,
\(z_b(v)=B_b(v)/(\|B_b(v)\|_1+\epsilon_i)\). The primary distance is

\[
D_{i,b}=\tfrac12\|z_b(\widehat\Delta_i)-z_b(\Delta_i^\star)\|_1.
\]

It is in `[0, 1]`, is invariant to common positive scaling, retains sign, and equals
zero for an exactly shape-faithful positive rescaling. If the oracle is nonzero and the
prediction is numerically zero, distance is one. Any block whose signed cancellation
makes the oracle block vector numerically zero is flagged; the absolute-mass transport
diagnostic must still be reported.

The primary hidden-distortion contrast is

\[
G_i=D_{i,1}-D_{i,8}.
\]

A positive value means temporal aggregation concealed part of the fine response error.
The multi-resolution distortion AUC over \(\log_2 b\) is secondary.

## 5. Diagnostic shape metrics

- **Temporal mass distance:** normalized one-dimensional Wasserstein distance between
  the L1-normalized absolute oracle and predicted response mass over horizons.
- **Onset error:** absolute difference between first active oracle and predicted
  horizons, divided by 23.
- **Peak-time error:** absolute difference between maximum-absolute-response horizons,
  divided by 23; deterministic earliest-index tie breaking.
- **Positive/negative mass errors:** relative errors in separate positive and negative
  response mass, required for biphasic rebound.
- **Interaction shape distance:** `D_1` applied to the joint-minus-marginal interaction
  contrast in M4.

Only signed shape distance and hidden-distortion gap enter the primary continuation
rule. Diagnostics cannot substitute for them.

## 6. Construct validation before model access

CPU-only fixtures must include:

1. exact oracle;
2. pure positive gain scaling;
3. sign flip;
4. one-step shift within a width-8 block;
5. shift across a width-8 boundary;
6. temporal smoothing with preserved signed mass;
7. deletion of the rebound lobe;
8. zero response; and
9. omission of the M4 interaction while preserving marginal responses.

Blocking assertions:

- oracle distances are zero within numerical tolerance;
- pure gain scaling changes RGR but leaves all shape distances zero;
- fine distance detects a within-bin shift that the corresponding coarse distance can
  conceal;
- cross-bin shift is no less severe than the matched within-bin shift at width 8;
- rebound deletion changes signed shape and the appropriate signed-mass diagnostic;
- interaction omission fails the interaction metric while leaving marginal fixtures
  unchanged;
- all metrics are finite and remain in their declared ranges; and
- distortion ordering is monotone over the frozen fixture-severity ladder.

The proposed numeric scientific bounds are reviewed only against these analytic
fixtures. No TSFM outputs may exist during this review.

## 7. Aggregation and primary decision

- Each backbone-by-mechanism cell contains 64 series under each of three seeds: 192
  paired series per cell.
- Report seed-level estimates, then aggregate seeds with equal weight.
- Use 5,000 deterministic percentile bootstrap replicates, resampling complete paired
  series within seed.
- Report every cell, including undefined diagnostics and null results.

A cell supports **resolution-hidden shape distortion** only if all four conditions hold:

1. coarse DSA bound passes;
2. coarse RGR interval passes;
3. the two-sided 95% lower bound of median `D_1` is at least `0.10`; and
4. the two-sided 95% lower bound of median `G = D_1 - D_8` is at least `0.05`.

P1 continues to a paper claim only if at least two cells pass, the cells span at least
two mechanisms or both backbones, and at least one passing cell has covariate-aware SQL
no worse than target-only by more than 2% on the point estimate.

This gate is proposed until construct validation is complete. Once the owner types
`APPROVE P1-SHAPE FREEZE`, it becomes immutable. Failure then means archive and stop;
it never authorizes relaxation.

## 8. Forecast-skill complement and references

Factual-world SQL and WQL are computed for covariate-aware and matched target-only calls.
Ridge-ARX and nonlinear dynamic regression are fitted only on each generated training
split and are labeled fitted references, never zero-shot peers. The analytic oracle and
deliberately broken fixtures validate measurement but are not forecasting competitors.

## 9. Leakage controls

- No P0 arrays, cell estimates, or confidence intervals enter generator parameters,
  fixture severities, metric thresholds, or model selection.
- P1 mechanism code, config, package lock, checkpoint revisions, and canonical hashes
  are frozen before the first model call.
- Smoke runs are structurally unable to compute the primary decision.
- Screening artifacts are append-only and stored outside Git; reports record hashes,
  runtime, checkpoint revision, GPU, and wall time.
- Any third backbone requires a contract amendment before P1 inference, not after
  observing P1 results.
