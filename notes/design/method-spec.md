# CovIntervene P0 method specification

Status: **draft for owner review; no model inference is authorized yet**  
Last updated: 2026-09-18

## 1. Audit unit and pairing

For series (i), a generator produces the same target history, covariate history,
future stochastic innovations, and two future covariate paths: factual
(x^{(0)}_{i,1:H}) and intervened (x^{(1)}_{i,1:H}). Only the named future
covariate may differ. Replaying the structural equation with shared innovations gives
the factual and intervened targets and the oracle response

\[
\Delta^{\star}_{i,h}=y^{(1)}_{i,h}-y^{(0)}_{i,h}.
\]

A frozen forecasting model is called twice with identical target history and all other
inputs. Its median-forecast response is

\[
\widehat{\Delta}_{i,h}=\widehat{y}^{(1)}_{i,h}-\widehat{y}^{(0)}_{i,h}.
\]

The series—not the horizon point—is the independent evaluation unit. Independently
sampled future noise is forbidden because it would contaminate the intervention effect.

## 2. Structural mechanisms

All mechanisms use a 192-step context and a 24-step horizon. Parameters are sampled
once per series from frozen ranges and retained in the private run artifact.

1. **Contemporaneous linear.** The target follows an autoregression with a signed
   current-time covariate effect. A smooth future block intervention tests immediate
   direction and recursive decay.
2. **Delayed distributed lag.** The covariate enters through a normalized three-lag
   kernel whose first nonzero lag is two steps. The oracle response therefore has a
   known onset and distributed temporal profile.
3. **Threshold saturation.** The covariate enters through a centered hyperbolic-tangent
   response. Factual paths are sampled around the transition region and interventions
   are clipped to the predeclared covariate envelope.
4. **Irrelevant placebo.** The target depends on an active covariate, while an
   independently generated placebo covariate has exactly zero structural coefficient.
   The audited intervention changes only the placebo. A matched active intervention on
   the same series supplies a nonzero reference scale.

For active mechanisms, intervention direction is balanced across series. Structural
coefficients are also sign-balanced, which prevents a predictor that always moves in
one direction from receiving a high aggregate score.

## 3. Primary response metrics

For active mechanisms, define the primary oracle-active support

\[
A_i=\{h:|\Delta^{\star}_{i,h}|\geq
0.10\max_j|\Delta^{\star}_{i,j}|\}.
\]

The 10% support threshold is primary. Thresholds of 5% and 20% are sensitivity
analyses and cannot replace it in the continuation decision.

### Directional Sign Agreement (DSA)

\[
\mathrm{DSA}_i=\frac{1}{|A_i|}\sum_{h\in A_i}
\mathbf{1}[\operatorname{sign}(\widehat{\Delta}_{i,h})=
\operatorname{sign}(\Delta^{\star}_{i,h})].
\]

Numerical zero is defined using a tolerance of (10^{-8}\max(1,
\max_h|\Delta^{\star}_{i,h}|)). DSA is not computed for the zero-effect placebo.

### Response Gain Ratio (RGR)

\[
\mathrm{RGR}_i=
\frac{\sum_{h\in A_i}|\widehat{\Delta}_{i,h}|}
{\sum_{h\in A_i}|\Delta^{\star}_{i,h}|+\epsilon_i}.
\]

This separates magnitude from direction: a sign-flipped oracle response has RGR 1 but
DSA 0.

### Normalized Response Error (NRE)

\[
\mathrm{NRE}_i=
\frac{\sum_{h\in A_i}|\widehat{\Delta}_{i,h}-\Delta^{\star}_{i,h}|}
{\sum_{h\in A_i}|\Delta^{\star}_{i,h}|+\epsilon_i}.
\]

### Temporal Localization Score (TLS)

Let (p_h=|\Delta^{\star}_{i,h}|/\|\Delta^{\star}_i\|_1) and
(q_h=|\widehat{\Delta}_{i,h}|/\|\widehat{\Delta}_i\|_1) over the full horizon.
Then

\[
\mathrm{TLS}_i=1-\tfrac12\sum_h|p_h-q_h|.
\]

TLS is in ([0,1]), equals one for identical normalized temporal profiles, and is set
to zero when the oracle response is nonzero but the predicted response is numerically
zero.

### Placebo Response Ratio (PRR)

Because the placebo oracle response is exactly zero, it is not used as a denominator.
Instead,

\[
\mathrm{PRR}_i=
\frac{\|\widehat{\Delta}^{\mathrm{placebo}}_i\|_1}
{\|\Delta^{\star,\mathrm{matched-active}}_i\|_1+\epsilon_i}.
\]

For all ratio metrics, (epsilon_i=10^{-8}\max(1,
\|\Delta^{\star}_i\|_1)) is numerical protection only; generator assertions reject
active examples whose oracle effect is below the frozen minimum.

## 4. Forecast-skill complement

Factual-world probabilistic accuracy is measured with scaled quantile loss (SQL) and
weighted quantile loss (WQL); median MAE is secondary. Every covariate-aware call is
paired with the same backbone in target-only mode. The relative SQL difference is

\[
r=(\mathrm{SQL}_{cov}-\mathrm{SQL}_{target})/\mathrm{SQL}_{target}.
\]

The complementary condition is satisfied when a verified fidelity violation occurs in
a cell with (r\leq0.02). Its confidence interval is reported but the frozen condition
uses the point estimate exactly as stated in the contribution contract.

## 5. Aggregation and uncertainty

- Each backbone-by-mechanism cell contains 64 series under each of three independent
  generator seeds, for 192 paired series.
- Report per-seed summaries and the pooled series median or mean specified by metric.
- Use 5,000 deterministic percentile bootstrap replicates, resampling complete series
  within generator seed and then aggregating seeds equally.
- For low-is-failure DSA, a violation requires the two-sided 95% interval upper bound
  below 0.80.
- For RGR, a violation requires the interval upper bound below 0.50 or lower bound
  above 1.50.
- For high-is-failure PRR, a violation requires the interval lower bound above 0.20.
- NRE and TLS are diagnostic in P0 and are not continuation-gate degrees of freedom.

## 6. Frozen P0 continuation rule

Continue only if at least two backbone-by-mechanism cells satisfy an applicable
violation rule, and those cells span at least two mechanisms or both backbones. At
least one such cell must also satisfy the complementary accuracy condition. All eight
backbone-by-mechanism cells, including null results, will be retained.

Failure means stopping this formulation or writing a bounded null-result artifact. It
does not authorize lowering thresholds. A safeguard or third-backbone branch requires
a new contract and untouched evidence.

## 7. Leakage and reproducibility controls

- The config file and implementation commit are hashed before the first model call.
- Model checkpoints and revisions are pinned in the run manifest.
- No fitting, prompt selection, covariate selection, or threshold selection uses P0
  outputs.
- Metric code must first pass exact oracle and deliberately broken-predictor tests.
- Raw series-level results are append-only; summaries are reproducible from them.
- Runtime metadata records Python, package, CUDA, GPU, model revision, wall time, and
  every random seed.
