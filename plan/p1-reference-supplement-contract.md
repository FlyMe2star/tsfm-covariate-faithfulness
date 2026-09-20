# P1-REF transparent-reference supplement contract

Status: **draft awaiting owner freeze**  
Approval phrase: `APPROVE P1-REF FREEZE`

## 1. Evidence status

This is a retrospective supplementary analysis designed after the P1-SHAPE primary
decision was observed. It is not preregistered primary evidence, cannot change the
P1-SHAPE continuation decision, and cannot be used to tune or relax any primary
threshold. Its only purpose is to contextualize the two frozen zero-shot checkpoints
against transparent models fitted without future-target access.

The original P1 configuration listed `ridge_arx` and
`nonlinear_dynamic_regression`, but it did not freeze their feature bases,
regularization, probabilistic construction, or fitting unit. No separate generated
training split was materialized. Presenting a newly implemented result as if those
details had been frozen before P1 inference would be inaccurate. This contract closes
that gap explicitly and labels every resulting artifact `post_primary_supplementary`.

## 2. Isolation from the primary evidence

- Implementation lives outside `src/covfaith`, whose complete Python-file set is part
  of the archived P1 scientific-code hash.
- The frozen P1 config, lock, units, decision, and result table are read-only inputs.
- Supplement code, config, and outputs receive a new independent hash chain.
- No TSFM inference is rerun and no primary aggregate is recomputed.
- Every mechanism, seed, series, world, and failed/null reference result is retained.

## 3. Fitting unit and leakage boundary

Each generated scenario is one fitting unit. Only its 192-step target context and two
covariate-context paths may be used. Future target values, analytic parameters, oracle
responses, intervention masks, model outputs, and primary pass/fail labels are excluded
from fitting and model selection.

For a context time index `t`, the supervised target is `y[t]`. Rows begin at `t=24`.
All feature standardization, target standardization, coefficient fitting, and residual
collection use context rows only. A world is forecast recursively for 24 steps using
the already fitted coefficients and that world's known future covariates.

## 4. Frozen candidate models

### Ridge-ARX

- target lags: `1, 2, 3, 6, 12, 24`;
- each covariate's lags: `0, 1, ..., 12`;
- intercept is unpenalized;
- every nonconstant feature and the supervised target are standardized from context
  rows only;
- ridge penalty: `alpha = 1.0` with no validation or tuning.

### Nonlinear dynamic regression

Start from the complete Ridge-ARX bank and append:

- `tanh(x_i[t-l] / s_i)` and `(x_i[t-l] / s_i)^2` for covariates `i=1,2` and
  lags `l=0,...,12`, where `s_i` is the context standard deviation;
- same-lag interactions `(x_1[t-l]/s_1)(x_2[t-l]/s_2)` for
  `l in {0,1,2,4,8,12}`;
- positive and negative parts of first differences for each covariate at
  `l in {0,1,2,3}`.

The generated feature columns are then standardized from context rows. The intercept
is unpenalized and the fixed ridge penalty is `alpha = 1.0`. There is no feature
selection or hyperparameter search.

## 5. Probabilistic forecasts and paired worlds

Point forecasts are recursive conditional means. For SQL and WQL, centered one-step
context residuals are resampled into 256 recursive paths. Quantiles are
`0.1,0.2,...,0.9`. The bootstrap seed is fixed globally and deterministically mixed
with mechanism, generator seed, series index, and model ID.

The same residual-index paths are replayed across factual, intervened, `a_only`,
`b_only`, and `joint` worlds. Thus the paired response reflects only the supplied
future-covariate path, not Monte Carlo mismatch.

## 6. Outputs and reporting

For both reference models and every frozen P1 scenario, retain:

- median and nine quantile forecasts for every required world;
- DSA, RGR, signed shape distance at widths 1/2/4/8, hidden-distortion gap,
  multi-resolution AUC, timing diagnostics, and interaction distance where defined;
- factual-world SQL, WQL, MAE, and WAPE;
- finite-fit status, matrix condition diagnostic, residual scale, and wall time.

Aggregation and intervals reuse the frozen P1 paired-series bootstrap solely for
comparability. There is no continuation threshold. Table 4 must identify these models
as fitted, post-primary, and descriptive. A poor reference result is retained; a good
reference result does not upgrade the primary claim.

## 7. Freeze boundary

Before any reference outcome is computed:

1. implement and unit-test design-matrix construction on synthetic fixtures only;
2. record the supplement config hash and independent code hash;
3. confirm that the archived P1 scientific-code hash is unchanged; and
4. obtain the exact owner phrase `APPROVE P1-REF FREEZE`.

After approval, no feature, penalty, bootstrap, metric, or reporting-rule change is
allowed. A defect may be fixed only with a versioned amendment that preserves the
failed artifact and explains the impact.
