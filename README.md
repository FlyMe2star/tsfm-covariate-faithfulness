# TSFM Covariate Faithfulness

Research repository for auditing whether zero-shot time-series foundation models
respond faithfully to known-future covariates under controlled interventions.

Working title: **Right Forecast, Wrong Reason? Auditing Covariate Response
Faithfulness in Zero-Shot Time-Series Foundation Models**.

The owner-approved P0 protocol has completed. Both frozen backbones produced all
screening units, but the first-run decision found zero predeclared fidelity violations;
the original failure-centered paper gate therefore did not pass. The result is retained
as a verified bounded screening outcome, and no threshold will be lowered post hoc. The prior
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
The reviewable P0 protocol is in
[`notes/design/method-spec.md`](notes/design/method-spec.md), with the machine-readable
configuration in [`configs/p0/covintervene_p0.yaml`](configs/p0/covintervene_p0.yaml).

The config is protected by an external canonical hash lock. The first Colab notebook
runs one series per mechanism and seed for one selected backbone. It writes a smoke-only
manifest and is structurally unable to compute the P0 continuation gate.

## Local verification

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
py -3.14 -m pytest -q
py -3.14 -m ruff check src tests
```

## Colab smoke

Open [`notebooks/01_p0_backbone_smoke.ipynb`](notebooks/01_p0_backbone_smoke.ipynb),
select a T4 GPU, choose one frozen backbone in the parameter cell, and run all cells.
Run it once for `chronos_2` and once for `timesfm_3`. Smoke artifacts are saved to
Google Drive outside Git.

## Frozen P0 screening

After both smoke receipts pass, open
[`notebooks/02_p0_backbone_screen.ipynb`](notebooks/02_p0_backbone_screen.ipynb).
Run it once for `chronos_2` and once for `timesfm_3`; each run is resumable at the
mechanism-by-seed unit boundary and cannot compute the continuation gate. When both
backbones report 12/12 complete full-screening units, run
[`notebooks/03_p0_analyze.ipynb`](notebooks/03_p0_analyze.ipynb) on CPU. The analysis
notebook verifies every stored array against its manifest before computing the frozen
decision, and reuses an existing decision instead of recomputing it.

## Research-integrity rule

P0 thresholds are frozen before the first model inference. A failed gate may motivate
a genuinely new experiment on untouched evidence, but it may not be lowered on the
observed P0 outputs.
