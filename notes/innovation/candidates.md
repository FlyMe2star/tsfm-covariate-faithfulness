# Candidate contribution framings

## A. Covariate-response faithfulness audit — selected

- **Claim:** Low forecast error does not guarantee that a frozen TSFM responds
  correctly to a controlled known-future covariate intervention.
- **Why it matters:** Forecasts are used for scenario planning, where response to a
  proposed promotion or weather path matters independently of observational accuracy.
- **Existing coverage:** Covariate-aware accuracy, model adaptation, single-domain
  explanations, generic synthetic TSFM behavior, and corruption robustness.
- **Gap:** No established cross-backbone protocol jointly measuring response sign,
  magnitude, temporal localization, placebo response, and forecast skill.
- **Evidence:** Analytic structural generators, paired interventions, oracle fixtures,
  two frozen backbones in P0 and at least three in the paper-stage benchmark.
- **Falsifier:** Both P0 backbones satisfy the frozen fidelity envelope across the
  active and placebo mechanisms.

## B. Realized-versus-forecast future-covariate gap — backup

- **Claim:** Evaluation with realized future covariates overstates deployable TSFM
  performance relative to genuinely forecast-time-available covariate paths.
- **Gap:** Benchmark availability semantics are often underspecified.
- **Evidence burden:** Requires archived weather/load forecasts aligned with outcomes,
  not reanalysis values.
- **Decision:** Keep as a later real-world extension. Do not select now because data
  acquisition and license validation threaten the 3–4 month schedule.

## C. Corruption-robust covariate forecasting — rejected

- **Claim:** Train or gate models to withstand noisy, missing, or misaligned covariates.
- **Decision:** Reject as the core topic. Exogenous Dropout (arXiv:2607.05452) already
  establishes a benchmark and strong simple baseline for this exact framing.

## D. Semantics-preserving corruption benchmark for FER — rejected for this cycle

- **Claim:** FER corruption ladders should be human-audited for expression-label
  preservation before robustness scores are interpreted.
- **Assets:** Existing RAF-DB and FER-C audit infrastructure.
- **Decision:** Scientifically plausible but requires more datasets and independent
  raters to support a CCF-B benchmark paper. The annotation bottleneck is less favorable
  than the selected no-annotation TSFM direction.

## Selection rationale

Candidate A offers the best combination of a narrow novelty hypothesis, falsifiability,
existing model/data adapters, low P0 cost, and two viable paper exits. It avoids both
previous projects' failed assumptions: it does not predict which input policy will win,
and it does not require a new robustness architecture to outperform a strong baseline.
