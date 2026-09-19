# P1-SHAPE topic brief

Status: **frozen, construct-validated, and owner-authorized for model inference**

Approval phrase: `APPROVE P1-SHAPE DESIGN`

Approval date: 2026-09-19

Freeze phrase: `APPROVE P1-SHAPE FREEZE`

Frozen config hash: `27826ad34bfe0acd8ee90ff4d0fbafb011a6b00aeea276dacaa9f9fe2a261918`

## Working title

**Beyond Direction and Gain: Multi-Resolution Covariate-Response Faithfulness in
Time-Series Foundation Models**

## Research question

When a covariate-aware time-series foundation model gets the direction and total
magnitude of a controlled future-covariate response broadly right, does it also place
that response at the correct forecast horizons and preserve its signed temporal shape?

## Primary hypothesis

For at least two predeclared backbone-by-mechanism cells, a frozen TSFM will satisfy a
coarse response envelope while exhibiting a practically meaningful fine-resolution
shape distortion that is hidden by temporal aggregation. This is a hypothesis, not a
result.

## Why this is a new contract

The completed P0 tested direction, total gain, and placebo response on four mechanisms.
It found no cell outside its frozen envelope. P1-SHAPE does not reinterpret that result,
lower its thresholds, or reuse its model outputs for selection. It introduces:

- four untouched structural mechanisms;
- new parameter ranges and generator seeds;
- a signed multi-resolution shape metric and construct-validation fixtures;
- a new primary contrast between fine and coarse temporal resolution; and
- a separately frozen success and stop rule.

P0 supplies motivation only: coarse metrics can be passed, so a genuinely different
question about within-horizon response shape is worth testing.

## Scope

- Frozen zero-shot covariate-aware checkpoints; no model fine-tuning.
- Primary backbones: Chronos-2 and TimesFM-3.
- Transparent fitted references: Ridge-ARX and a compact nonlinear dynamic-regression
  baseline.
- Horizon 24 with response aggregation widths 1, 2, 4, and 8.
- Exact paired structural replay with shared innovations.
- Synthetic structural evidence is primary; observational data cannot establish the
  structural-response claim.

## Untouched mechanism families

1. **Biphasic rebound** — an early response is followed by an opposite-signed rebound.
2. **Dispersed delayed pulse** — a localized intervention is transmitted through a
   unimodal delay kernel with variable onset and width.
3. **Rate-asymmetric hysteresis** — rising and falling covariate paths have different
   state-dependent effects.
4. **Two-covariate synergy** — joint intervention response contains a known interaction
   component that cannot be recovered by adding the two marginal responses.

These families do not reuse the P0 contemporaneous-linear, fixed distributed-lag,
threshold-saturation, or irrelevant-placebo cells.

## Non-goals

- Estimating causal effects from real observational data.
- Claiming universal TSFM behavior beyond the frozen backbones and mechanisms.
- Generating counterfactual inputs that achieve a desired forecast.
- Inspecting or steering internal representations.
- Developing a mitigation before a response-shape failure is independently verified.
- Treating a failed P1 gate as permission to relax the gate.

## Three-stage authorization

1. **Design — complete:** claim, metric, mechanism, and evidence contracts approved.
2. **Construct validation — complete:** 11/11 blocking checks passed without loading a
   TSFM; 768 frozen generator scenarios passed deterministic paired preflight.
3. **Inference freeze — complete:** the owner supplied `APPROVE P1-SHAPE FREEZE` before
   any P1 checkpoint call. Threshold changes are now forbidden.

## Planned deliverable

An auditable `CovIntervene-SHAPE` extension that reports response distortion across
temporal resolutions, complete null cells, uncertainty intervals, observational forecast
skill, and a bounded claim about the two tested model families.
