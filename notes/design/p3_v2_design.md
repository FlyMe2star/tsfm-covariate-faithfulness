# P3-v2 semi-synthetic extension: candidate for owner review

Original proposal status: **candidate, not approved or frozen**. At the time of
this design, no v2 oracle/construct preflight or checkpoint inference had been
run. The v1 adverse receipt remains immutable; this is one explicitly
post-pilot redesign, not a retroactive correction.

Execution addendum (2026-09-25): the owner subsequently approved this candidate
with `APPROVE P3-V2 DESIGN`. The [CPU construct report](../../evidence/p3_semisynthetic/v2_construct/p3_v2_construct_preflight.json)
passed the registered gate. This addendum does not change the original design
or retroactively authorize model inference. The owner then separately approved
the [model freeze](../../configs/p3_semisynthetic/p3_v2_model_freeze_approval.json)
with `APPROVE P3-V2 MODEL FREEZE`. The new checkpoint runner lives outside the
locked construct package. Its smoke and full outputs remain pending, so this
approval adds no model-result claim.

## Why another attempt is scientifically defensible

The [P3-v1 construct receipt](../../evidence/p3_semisynthetic/construct/p3_v1_construct_preflight.json)
reports 38/864 scenarios below the predeclared oracle-mass criterion, with all
three dispersed-delay cells above the 5% exclusion ceiling. A data-only
cross-tab of the private v1 unit records shows 17/144 exclusions when the
dispersed pulse starts at step 13, versus 9/144 and 5/144 at starts 1 and 7.
This association does **not** establish timing as the sole cause. The v1
selection, thresholds, and failed decision must be disclosed if v2 appears in
the paper.

For the 12-lag dispersed kernel and pulse lengths up to 7, a start no later
than step 5 lets the full finite response support fall within the 24-step
forecast horizon. V2 therefore uses dispersed starts `[1,3,5]`, retaining the
biphasic `[1,7,13]` starts. A one-context-SD bounded covariate pulse, up from
v1's 0.65 SD, is an interpretable single change in intervention magnitude;
the original 0.5%–99.5% context-quantile clipping remains. These choices were
informed by the failed *construct* pilot, not by any model outcome. They are
registered as **one candidate**, with no severity grid and no choice after
v2 oracle results. The multiplicative target link and all kernel parameter
ranges remain unchanged. The 0.005 minimum oracle-to-factual $L_1$ ratio and
the 5% maximum exclusion fraction remain unchanged.

## Construct-held-out series

Reconstruct and verify the exact v1 selection manifest from the pinned source
files, then exclude every one of its 48 selected original IDs per source.
A [data-only feasibility receipt](../../evidence/p3_semisynthetic/v2_design/holdout_feasibility.json)
found 275 / 34 / 89 eligible remaining IDs for traffic / workload / solar.
V2 deterministically selects **32 previously
unused IDs per source** with a new salt, one origin per ID, and three new seeds
`[1801,2203,3001]`. Selection uses only target context and known timestamps,
never future targets or model outputs. The source-ID overlap between v1 and v2
must be exactly zero. The whole source collections were seen during v1's
data-only audit, so “held-out” refers specifically to construct scenarios,
not untouched datasets or independent domains.

The selected ID/origin manifest must be written before any v2 oracle is
computed. If fewer than 32 eligible unused IDs remain for any source, or the
v1 manifest hash cannot be reconstructed, stop rather than substitute data.
Future-target finiteness and effect mass are checked *after* selection;
failures are retained without replacement.

## Evaluation contract and stopping rule

Each source-by-family cell has 32 ID clusters × 3 seeds = 96 attempted
scenarios. At most 4 exclusions are allowed per cell; 5/96 exceeds 5%.
The CPU preflight checks paired target-context identity, exact shared-background
oracle subtraction, placebo structural zero with a nonzero placebo perturbation,
deterministic replay, finite nonnegative targets, nominal cadence, disjoint
source IDs, and the unchanged oracle-mass criterion. It computes no forecast
accuracy or model-selection metric. **If any cell fails, stop P3** and finish
the paper with P1/P2; do not make P3-v3 on these sources to chase a pass.

Only after the v2 construct receipt passes and the owner separately approves
a model-inference freeze may the same pinned Chronos-2 and TimesFM-3 checkpoints
be run. The complete 3 sources × 2 mechanisms × 2 backbones matrix, all nulls,
source-ID-clustered 5,000-replicate intervals, target-only SQL/WQL complement,
12-ID placebo control, and 12-ID lower-link sensitivity remain as in v1.
The P1 cell rule is applied descriptively without threshold changes; evidence
for transfer requires at least one passing complete cell in at least two
sources. V2 cannot alter the P1 primary decision.

This remains a **semi-synthetic** test: the real series supply target
backgrounds, while the covariates and their causal response are synthetic.
Possible pretraining overlap with public sources, activity-regime selection,
the small 32-ID source clusters, and source/model license restrictions must
be stated. No P3 result belongs in `paper/main.tex` until a verified aggregate
report exists. The selected baselines and their roles are inherited unchanged
from `notes/design/p3_semisynthetic_baselines.csv` (SHA-256 pinned in the v2
candidate config); v2 components and experiments are listed separately.
