# Topic brief

## Working title

**Right Forecast, Wrong Reason? Auditing Covariate Response Faithfulness in
Zero-Shot Time-Series Foundation Models**

## Research question

When a zero-shot time-series foundation model receives known-future covariates,
does its forecast respond in the correct direction, at the correct horizon, and with
an appropriate magnitude—or can ordinary forecast accuracy conceal unfaithful use of
the covariate path?

## Motivation

Chronos-2 and TimesFM-3 natively accept known-future covariates. Existing benchmarks
mainly rank predictive loss, while recent explanation and causal-analysis work does not
yet provide a cross-model audit focused on controlled interventions to future covariate
paths. This leaves a deployment-relevant measurement gap: a forecast can be accurate
on an observational test set while responding incorrectly to a planned promotion,
weather path, or other scenario input.

## Scope

- Frozen zero-shot forecasting backbones; no large-model pretraining.
- Numerical known-future covariates in the primary study.
- Controlled synthetic structural mechanisms as the source of response ground truth.
- Real FEV tasks as secondary plausibility evidence, not causal ground truth.
- Primary backbones: `amazon/chronos-2` and `google/timesfm-3.0-pytorch`.
- Optional paper-stage backbone: CITRAS-FM, subject to reproducible public weights and
  license review.

## Non-goals

- Estimating real-world causal effects from observational data.
- Claiming that one forecasting backbone is universally superior.
- Reusing the sealed outcomes of `covariate-safe-tsfm` for selection or tuning.
- Training a new time-series foundation model.
- Treating SHAP or attention weights alone as response ground truth.

## Constraints

- **Target**: empirical paper for a CCF-B data-mining / machine-learning conference.
- **Assumed length**: 8–10 pages excluding references and appendix.
- Completion window: 3–4 months.
- Compute: Colab T4 for P0; A100 only after a frozen continuation decision.
- Engineering: local VS Code/Codex development with reproducible Colab execution.
- Human annotation: none required for the primary claim.

## P0 decision philosophy

The gate is deliberately attainable but not retrospective. The project may continue as
an evaluation paper without a new mitigation method if controlled response failures are
replicable and scientifically informative. A method contribution is optional and must
be evaluated under a new frozen contract on untouched generators or tasks.

## Planned deliverable

An open `CovIntervene` protocol containing parameterized mechanisms, paired covariate
interventions, response-faithfulness metrics, model adapters, and reproducible reports.
