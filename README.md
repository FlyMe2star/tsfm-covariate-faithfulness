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
An evidence-constrained manuscript draft, construct-validation table, complete
eight-cell primary table, and post-primary reference table are under
[`paper/`](paper/). The literature gate passed on 2026-09-20 with 30
primary-source-verified entries. The owner-facing 13-paper review list is
[`notes/innovation/core-review-shortlist.md`](notes/innovation/core-review-shortlist.md),
with a direct claim-to-prose guide in
[`notes/writing/core-review-to-prose-map.md`](notes/writing/core-review-to-prose-map.md).

A transparent Ridge-ARX/nonlinear reference supplement has completed outside the
frozen P1 code package. The owner supplied `APPROVE P1-REF FREEZE` on 2026-09-21;
all 12 units and eight model-by-mechanism summaries subsequently passed the frozen
hash and completeness checks. The supplement is explicitly post-primary and
descriptive; it cannot change the archived P1 decision. See the
[`contract`](plan/p1-reference-supplement-contract.md) and
[`verified result summary`](evidence/p1_shape/references/result-summary.md).

Ridge-ARX nearly recovered both additive temporal mechanisms but not the registered
hysteresis and synergy behavior. The nonlinear feature expansion partially recovered
the synergy interaction without uniformly improving shape or forecast error. These
results contextualize mechanism difficulty and do not establish a preregistered
reference-versus-TSFM comparison.

The verified P2 analysis-only robustness supplement is now archived under
[`evidence/p2_robustness/results/`](evidence/p2_robustness/results/), with
Matplotlib vector figures in [`paper/figures/generated/`](paper/figures/generated/).
It added fixed-fine-normalization curves, analytic metric comparisons, WQL, and
threshold sensitivity without new model inference or any change to the P1 gate.

The proposed external-validity extension, P3, has an owner-approved design:
[`notes/design/p3_semisynthetic_design.md`](notes/design/p3_semisynthetic_design.md)
and [`configs/p3_semisynthetic/p3_design_candidate.yaml`](configs/p3_semisynthetic/p3_design_candidate.yaml).
The data-only audit found enough eligible original traffic, cloud-workload, and
solar series at one pinned public revision. Its subsequent CPU construct preflight
failed the registered 5%-maximum exclusion gate in three dispersed-delay cells;
see the [adverse receipt](evidence/p3_semisynthetic/construct/p3_v1_construct_preflight.json)
and [decision note](notes/design/p3_v1_construct_failure.md). No P3 checkpoint
call or manuscript result claim is authorized. The failed pilot remains archived;
the frozen P1/P2 evidence is unchanged.

The explicitly post-pilot, construct-held-out [P3-v2 design](notes/design/p3_v2_design.md)
retains the v1 failure record and original thresholds, excludes v1's source IDs,
and places the delayed pulse fully inside the forecast horizon. Its [CPU construct
receipt](evidence/p3_semisynthetic/v2_construct/p3_v2_construct_preflight.json)
passes all six source-family gates: 5 exclusions among 576 attempted scenarios,
with no source-ID overlap. This is **not** forecasting evidence. Model inference
was separately approved on 2026-09-25 in the [model-freeze receipt](configs/p3_semisynthetic/p3_v2_model_freeze_approval.json).
The [resumable Colab notebook](notebooks/09_p3_v2_backbone_units.ipynb) is ready,
but no P3-v2 checkpoint output or aggregate result is claimed yet. The runner
reconstructs frozen source selection and construct exclusions before forecasting.

For P3-v2, [open the notebook in Colab](https://colab.research.google.com/github/FlyMe2star/tsfm-covariate-faithfulness/blob/main/notebooks/09_p3_v2_backbone_units.ipynb)
on T4. Run `MODE='checkpoint_smoke'` once with `BACKBONE='chronos_2'` and once
with `BACKBONE='timesfm_3'`. Then set `MODE='full_units'` and repeat for each
backbone; switch to A100 only if T4 is insufficient. The 18 full units per
backbone resume individually in Google Drive. The notebook writes no aggregate
paper decision; private Parquet and forecasts stay outside Git. Keep both smoke
and full completion reports for the next read-only analysis stage.

The owner has reported both backbones at 18/18 units and 571 valid scenarios
each, but those private arrays have not yet undergone the independent archive
audit. Run the CPU-only [P3-v2 analysis notebook](notebooks/10_p3_v2_analyze.ipynb)
([open in Colab](https://colab.research.google.com/github/FlyMe2star/tsfm-covariate-faithfulness/blob/main/notebooks/10_p3_v2_analyze.ipynb))
to verify all 36 arrays and produce the twelve-cell descriptive matrix. The
registered lower-link SQL cannot be computed from the archived median-only
lower-link forecasts and is marked missing, never approximated.

Reproduce the frozen supplement on CPU with
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
