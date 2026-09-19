# TSFM Covariate Faithfulness

Research repository for auditing whether zero-shot time-series foundation models
respond faithfully to known-future covariates under controlled interventions.

Current working title: **Beyond Direction and Gain: Multi-Resolution
Covariate-Response Faithfulness in Time-Series Foundation Models**.

The owner-approved P0 protocol has completed. Both frozen backbones produced all
screening units, but the first-run decision found zero predeclared fidelity violations;
the original failure-centered paper gate therefore did not pass. The result is retained
as a verified bounded screening outcome, and no threshold will be lowered post hoc.

The owner has now approved the **P1-SHAPE design direction**. P1 asks a new question:
whether direction and aggregate gain can look correct while the signed response is
misplaced within the forecast horizon. It uses untouched mechanisms, parameters,
seeds, and metrics. P0 is motivation only and cannot be used for P1 selection.

## P1-SHAPE workflow

1. Implement and validate multi-resolution metrics and four untouched mechanisms on
   CPU without loading a TSFM.
2. Review the construct-validation report and proposed numeric bounds.
3. Require the explicit phrase `APPROVE P1-SHAPE FREEZE` before model inference.
4. Freeze configs, code, dependencies, checkpoint revisions, and hashes.
5. Smoke Chronos-2 and TimesFM-3 on T4, then execute the resumable full matrix.
6. Compute the frozen decision once, retain every cell, and either write the bounded
   result or stop without threshold relaxation.

See [`brief/p1-shape-topic-brief.md`](brief/p1-shape-topic-brief.md),
[`brief/p1-shape-contribution-map.yaml`](brief/p1-shape-contribution-map.yaml), and
[`notes/design/p1-shape-method-spec.md`](notes/design/p1-shape-method-spec.md) for the
current contract. Original P0 documents and evidence remain unchanged and auditable.

## Archived P0 reproduction

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

P0 thresholds remain frozen. P1 may use P0 only as motivation, not as evidence or a
source of thresholds. P1 model inference is not authorized until construct validation
passes and the owner explicitly freezes the new contract.
