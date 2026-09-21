# TSFM Covariate Faithfulness

Research repository for auditing whether zero-shot time-series foundation models
respond faithfully to known-future covariates under controlled interventions.

Current working title: **Beyond Direction and Gain: Multi-Resolution
Covariate-Response Faithfulness in Time-Series Foundation Models**.

The owner-approved P0 protocol has completed. Both frozen backbones produced all
screening units, but the first-run decision found zero predeclared fidelity violations;
the original failure-centered paper gate therefore did not pass. The result is retained
as a verified bounded screening outcome, and no threshold will be lowered post hoc.

The owner has now frozen the **P1-SHAPE protocol**. P1 asks a new question:
whether direction and aggregate gain can look correct while the signed response is
misplaced within the forecast horizon. It uses untouched mechanisms, parameters,
seeds, and metrics. P0 is motivation only and cannot be used for P1 selection.

The frozen P1-SHAPE decision subsequently **passed**. Three of eight complete cells
satisfied the predeclared direction, gain, fine-shape, and resolution-gap conditions,
while spanning two mechanisms and both audited backbones. This supports a bounded paper
claim for the tested checkpoints and structural mechanisms; it is not a universal TSFM
or real-world causal claim. See
[`evidence/p1_shape/results/result-summary.md`](evidence/p1_shape/results/result-summary.md).
An evidence-constrained manuscript scaffold and the complete eight-cell primary table
are under [`paper/`](paper/). The literature gate passed on 2026-09-20 with 30
primary-source-verified entries. The owner-facing 13-paper review list is
[`notes/innovation/core-review-shortlist.md`](notes/innovation/core-review-shortlist.md).

A transparent Ridge-ARX/nonlinear reference supplement is now implemented and tested
outside the frozen P1 code package. The owner supplied `APPROVE P1-REF FREEZE` on
2026-09-21, and the independent supplement config is now locked. The supplement is
explicitly post-primary and descriptive; it cannot change the archived P1 decision. See
[`plan/p1-reference-supplement-contract.md`](plan/p1-reference-supplement-contract.md).

Run the frozen supplement on CPU with
[`notebooks/07_p1_reference_supplement.ipynb`](notebooks/07_p1_reference_supplement.ipynb).
It stores each mechanism-by-seed unit independently and safely resumes after a Colab
disconnect. [Open the notebook directly in Colab](https://colab.research.google.com/github/FlyMe2star/tsfm-covariate-faithfulness/blob/main/notebooks/07_p1_reference_supplement.ipynb).

## P1-SHAPE workflow

1. Multi-resolution metrics and four untouched mechanisms passed CPU construct
   validation without loading a TSFM.
2. The owner supplied `APPROVE P1-SHAPE FREEZE`; config and code hashes are locked.
3. Smoke Chronos-2 and TimesFM-3 on T4, then execute the resumable full matrix.
4. Compute the frozen decision once, retain every cell, and either write the bounded
   result or stop without threshold relaxation.

See [`brief/p1-shape-topic-brief.md`](brief/p1-shape-topic-brief.md),
[`brief/p1-shape-contribution-map.yaml`](brief/p1-shape-contribution-map.yaml), and
[`notes/design/p1-shape-method-spec.md`](notes/design/p1-shape-method-spec.md) for the
current contract. Original P0 documents and evidence remain unchanged and auditable.

## Frozen P1-SHAPE execution

1. Open [`notebooks/04_p1_shape_backbone_smoke.ipynb`](notebooks/04_p1_shape_backbone_smoke.ipynb)
   on a T4 and run once for `chronos_2`, then once for `timesfm_3`.
2. After both smoke reports show 12/12 units, run
   [`notebooks/05_p1_shape_backbone_screen.ipynb`](notebooks/05_p1_shape_backbone_screen.ipynb)
   once per backbone. Full units are resumable in Google Drive.
3. Only after both full reports show 12/12 units, run
   [`notebooks/06_p1_shape_analyze.ipynb`](notebooks/06_p1_shape_analyze.ipynb) on CPU.

The smoke and full-screen notebooks cannot compute the aggregate decision. The analysis
notebook refuses incomplete or hash-mismatched units.

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
source of thresholds. P1 is now inference-authorized under config hash
`27826ad34bfe`; no threshold may change after this authorization.
