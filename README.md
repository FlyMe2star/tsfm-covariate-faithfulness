# TSFM Covariate Faithfulness

Research repository for auditing whether zero-shot time-series foundation models
respond faithfully to known-future covariates under controlled interventions.

Working title: **Right Forecast, Wrong Reason? Auditing Covariate Response
Faithfulness in Zero-Shot Time-Series Foundation Models**.

The project is currently at the pre-inference P0 design gate. All contribution claims
are hypotheses until backed by frozen experiment artifacts. The prior
`covariate-safe-tsfm` project and its sealed outcomes are not reused for model or
threshold selection here.

## Planned workflow

1. Freeze controlled structural data-generating processes and response metrics.
2. Run a small Chronos-2 and TimesFM-3 phenomenon-existence pilot on Colab T4.
3. Stop if response-faithfulness failures are not reproducible across mechanisms or
   backbones.
4. If the gate passes, expand to a paper-eligible benchmark and optionally test one
   lightweight safeguard under a separately frozen contract.

See [`brief/topic-brief.md`](brief/topic-brief.md) and
[`brief/contribution-map.yaml`](brief/contribution-map.yaml) for the research contract.

## Research-integrity rule

P0 thresholds are frozen before the first model inference. A failed gate may motivate
a genuinely new experiment on untouched evidence, but it may not be lowered on the
observed P0 outputs.
