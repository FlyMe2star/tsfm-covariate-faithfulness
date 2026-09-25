# P3 semi-synthetic extension: design for owner review

Status: **candidate, not frozen**. No P3 checkpoint inference is authorized by this
document. The P1 decision, P2 supplement, and their hashes remain immutable.
The paper target remains an approximately eight-page CCF-B empirical submission;
P3 will be a separately labeled secondary result, not a replacement for P1.

## Question and evidence boundary

Does the P1 response-shape finding persist when the background target trajectory
comes from public traffic, cloud-workload, and solar series instead of a fully
synthetic autoregression? The causal covariate and response mechanism are still
synthetic. Therefore a positive result would show transfer to **realistic target
backgrounds**, not causality of a real observed covariate. A null result is retained
and reported without changing P1.

The data-only source audit is in `notes/data/p3_source_audit.md`. Use the exact
Parquet revision and SHA-256 values in `configs/p3_semisynthetic/p3_design_candidate.yaml`.
The sources have 323, 82, and 137 eligible original series, respectively, under
the candidate data-only window rule. The plan selects 48 distinct original series
from each source; all three synthetic seeds share one selected background window
per source ID. Thus the independent bootstrap unit is the original source ID,
**not** a seed, horizon, or overlapping window.

## Frozen-after-approval scenario construction

Let $b_{i,t}\geq0$ be a real 216-step background window; $L=192$, $H=24$. Generate
two independent AR(1) synthetic known-future covariates, `active_synthetic` and
`placebo_synthetic`, with 96 warm-up steps. The active covariate drives either the
P1 biphasic-rebound or dispersed-delayed-pulse filter with the **same numeric
kernel ranges** as P1, but the response innovations are zero: the real background
supplies the non-mechanistic variation. For both families,

```text
r_t = phi * r_(t-1) + beta * filter_family(active_x_<=t)
s   = max(Q75(abs(r_context)), 1e-6)        # context only, shared across worlds
y_t(world) = b_t * exp(log(1.4) * tanh(r_t(world) / s))
oracle_delta_h = y_h(intervened) - y_h(factual)
```

The same real $b_t$, initial response state, unperturbed placebo, and active
covariate history are used in both worlds. Only the active **future** covariate
receives the smooth pulse. This guarantees identical target history and an exact
oracle response while preserving nonnegativity and much of the original target
scale/seasonality. The link is multiplicative, so the oracle is not assumed to
have the same normalized shape as the P1 additive generator; it is computed by
paired replay. Where $b_t=0$, the response is zero. Selection does not inspect
future target values: workload windows require positive recent context and solar
origins use a fixed clock-hour set. Residual idle horizons and the resulting
effect-mass exclusions must be disclosed in the paper.

Source and origin selection are deterministic and independent of checkpoint
outputs: rank eligible source IDs by SHA-256 of the specified salt, source name,
and ID; retain the first 48. Candidate origins are in the final 40% of each
source series, on a 24-step grid, and pass the finite, cadence, context-variance,
source-specific context-activity, and known-timestamp rules. Choose the minimum
SHA-256-ranked eligible origin for each ID. Three P3-specific seeds generate
paired scenarios on the same window. Direction, coefficient sign, pulse start,
and pulse length are balanced by the frozen source/seed ranks. The implementation
must write a selection manifest **before** model loading; it may not replace an
ID or origin after model results are available.

The CPU construct validation must check exact paired target-context identity,
unchanged background and placebo in the active intervention, finite bounded
targets, deterministic replay, a nonzero oracle, and an oracle $L_1$-mass ratio
of at least 0.005. It must report every excluded scenario. If more than 5% of
planned scenarios fail in any source-by-family cell, stop before checkpoint
inference and submit a revised candidate design for owner review. A result-driven
change to source, link strength, or thresholds is forbidden.

## Comparisons, uncertainty, and reporting

The two P1 checkpoint revisions and native covariate APIs are reused without
fine-tuning. Each paired scenario receives matched `target_only`, `factual`, and
`active_intervention` calls. The exact oracle is the measurement reference.
Context-fitted Ridge-ARX and nonlinear dynamic regression remain optional
**descriptive** comparators, never mixed with zero-shot model rankings. A
placebo-only future perturbation is run on the first 12 selected source IDs
per dataset; its oracle response is identically zero and its model-response
mass is normalized by the active oracle mass. The same 12 IDs also receive a
lower link strength, $\log(1.2)$ instead of $\log(1.4)$, without changing any
other scenario parameter. Four control axes are registered in the experiment
matrix: paired identity, target-only complement, placebo sham, and link-strength
sensitivity. These are audit controls, not invented neural architecture
ablations.

For every one of the 3 sources × 2 families × 2 backbones = **12 complete cells**,
report DSA, RGR, $D_1,D_2,D_4,D_8$, registered $G=D_1-D_8$, relative SQL and WQL,
and all exclusions. Use 5,000 percentile bootstrap replicates, resampling the
48 original source IDs within each cell while keeping all three synthetic seeds
for an ID together. Show source-wise results and equal-source macro summaries;
do not pool source-by-seed observations as independent units. Include at least one
predefined representative per source only if the corresponding cell is reported
in full; use the P2 median-proximity rule, never manual case selection.

The **secondary descriptive transfer rule** applies the P1 complete cell rule
and SQL complement unchanged to each of the 12 P3 cells. Support for transfer
requires at least one complete cell in at least two of the three sources.
Failure yields a bounded non-replication and the complete matrix, not another
threshold revision. This P3 label cannot retroactively modify the P1 gate.
No unverified P3 result should be written as fact in `paper/main.tex`.

## Implementation and decision gates

1. Owner reviews this design, source/licensing caveats, and candidate config.
2. Implement a separate `covfaith_p3` package and tests without editing frozen
   P1 metric/generator code. The same model adapters may be reused through a
   compatible scenario interface.
3. Run CPU-only data/construct preflight. Record exact source hashes, selected
   IDs/origins, exclusions, oracle range, and source/code/config hashes. No
   forecast accuracy or model-selection metric is computed.
4. Owner approves a **separate P3 freeze** after the preflight. Only then create
   Colab smoke/full notebooks and call the pinned checkpoints. Start with T4;
   use A100 only for runtime, not scientific redesign.
5. Archive all 12 cells and null/adverse results. Backfill the manuscript only
   after a verified aggregate report and run the approved Matplotlib vector-figure
   workflow. Original dataset arrays stay outside public Git.

Risks to highlight in the paper: artificial covariate semantics; selection of
active historical regimes and midday solar origins; potential checkpoint
pretraining overlap with public
background series; Redset's biased fleet sample; research/noncommercial data and
model terms; and only 48 independent IDs per source in the main analysis.
