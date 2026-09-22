# P2 metric-robustness supplement contract

Status: **frozen and owner-approved on 2026-09-22; no outcomes computed at freeze**

Approval phrase: `APPROVE P2 ROBUSTNESS FREEZE`

## 1. Purpose and evidence status

This is a retrospective, post-primary robustness analysis prompted by external review.
It cannot alter the frozen P1-SHAPE endpoint, thresholds, pass/fail decision, or primary
claim. Its purposes are to:

1. separate the registered, per-resolution-normalized resolution contrast from a
   mathematically monotone aggregation diagnostic;
2. compare the registered shape metric with common trajectory diagnostics on the
   pre-model analytic fixtures;
3. report the already-stored WQL results; and
4. show how the registered paper-level count changes over a declared neighborhood of
   the two shape thresholds.

No model inference is performed. Only hash-verified archived P1 unit files are read.

## 2. Registered metric clarification

The P1 quantity

\[
G_i=D_{i,1}-D_{i,8}
\]

is named **resolution contrast**. Because each resolution is normalized separately,
it need not be nonnegative and is not presented as a pure contraction gap. This is a
terminology and interpretation correction only; its implementation and frozen result
remain unchanged.

## 3. Monotone supplementary diagnostic

For nonzero fine-resolution responses, define

\[
p=\widehat\Delta/\|\widehat\Delta\|_1,\qquad
o=\Delta^\star/\|\Delta^\star\|_1,
\]

and

\[
C_b=\frac12\|B_b(p-o)\|_1,
\qquad H_{1\to b}=C_1-C_b.
\]

For a zero predicted response, all widths receive the same maximal distance one and
the gap is zero. Oracle-zero cases remain undefined and excluded under the same
numerical tolerance as P1.

For any block-sum operator, the triangle inequality gives

\[
\|B_b x\|_1\leq\|x\|_1,
\]

so (C_b\leq C_1) and (H_{1\to b}\geq0). This proposition is analytic; the archived
model outputs are used only to measure its empirical magnitude.

## 4. Frozen analyses

### 4.1 Analytic-fixture comparison

Evaluate every existing construct-validation fixture with:

- registered (D_1) and resolution contrast (G);
- monotone (C_1) and (H_{1\to8});
- normalized RMSE;
- cosine distance;
- Pearson distance;
- dynamic-time-warping distance normalized by horizon;
- one-dimensional mass-transport distance;
- onset error;
- peak-time error; and
- response gain ratio.

Undefined values are retained as missing rather than imputed. The comparison is
descriptive: no metric is declared universally superior from these fixtures.

### 4.2 Archived-cell summaries

For each of the eight frozen backbone--mechanism cells, report paired-series-bootstrap
means and 95% intervals for (C_1,C_2,C_4,C_8), (H_{1\to8}), SQL, and WQL. Reuse the
P1 deterministic bootstrap recipe and 5,000 replicates.

### 4.3 Threshold sensitivity

Recompute only the descriptive number of registered passing cells over the Cartesian
grid

- (D_1) lower bound: 0.04, 0.06, ..., 0.16;
- registered (G) lower bound: 0.01, 0.02, ..., 0.08.

DSA, RGR, bootstrap intervals, and all other registered rules remain fixed. This grid
does not redefine practical significance or retrospectively select a threshold.

### 4.4 Representative trajectories

The eligible set is the three registered P1 passing cells. Within each cell, select
the series minimizing the sum of median-absolute-deviation standardized distances to
the cell medians of (D_1) and registered (G). Break exact ties by series ID. Store
the selected oracle and predicted trajectories and their selection diagnostics.

## 5. Outputs

- `monotone_cell_summary.csv`
- `wql_cell_summary.csv`
- `metric_fixture_comparison.csv`
- `threshold_sensitivity.csv`
- `representative_responses.json`
- `p2_robustness_report.json`

All outputs must record the P2 config hash, supplement-code hash, frozen P1 config and
scientific-code hashes, source artifact hashes, and the post-primary evidence label.

## 6. Freeze and execution boundary

Before analysis:

1. unit tests for contraction, zero-mass behavior, fixtures, and hash checks must pass;
2. the archived P1 scientific-code hash must remain unchanged;
3. the draft config must be locked without changes; and
4. the owner must provide the exact approval phrase above.

After approval, analysis rules may not be changed in response to results. Any defect
requires a versioned amendment that preserves the failed artifact and explains impact.
