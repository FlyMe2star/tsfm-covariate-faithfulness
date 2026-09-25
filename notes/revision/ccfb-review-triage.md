# CCF-B pre-submission review triage

Status: active pre-submission revision plan

This is an internal author document, not a reviewer-facing response. The review is
useful overall, but its suggestions differ in evidential value and in whether they can
be added without violating the frozen P1 boundary.

## Decision summary

The current paper has a credible bounded finding, but it is not yet maximally
competitive. The highest-return path is:

1. correct the registered metric's mathematical interpretation;
2. add a post-primary, analysis-only monotone coarsening diagnostic;
3. compare common trajectory metrics on the same analytic fixture ladder;
4. expose representative response paths and complete WQL/sensitivity reporting; and
5. run a separately frozen semi-synthetic realistic-context extension.

The original P1 arrays, thresholds, passing cells, and decision remain immutable.

## Comment tracker

| ID | Concern | Assessment | Severity | Action | Status | Evidence boundary |
|---|---|---|---|---|---|---|
| C1 | Per-resolution L1 normalization complicates the interpretation of $G=D_1-D_8$ | Valid. $G$ is a resolution contrast, not a general monotonicity theorem. The review also exposed a manuscript/code mismatch: epsilon was printed in the denominator, whereas code used it only as a zero-mass tolerance. | major | ACCEPT_TEXT + ACCEPT_ANALYSIS | P2 analysis and text done | Registered $G$ unchanged; separate fixed-fine-normalization diagnostic and contraction proof added. |
| C2 | Evidence is entirely synthetic | Valid and the main external-validity weakness. | major | ACCEPT_EXPERIMENT | P3-v1 CPU gate failed; P3-v2 held-out CPU construct gate passed; model freeze pending | Preserve the adverse pilot and every v2 exclusion. No P3 model result exists; neither semi-synthetic test is real-covariate causal validation. |
| C3 | Common trajectory metrics are not compared | Valid. Existing fixture validation shows selectivity but does not establish comparative necessity. | major | ACCEPT_ANALYSIS | P2 fixture table done | Same analytic fixtures; report values and invariances without outcome-driven metric selection. |
| C4 | No response-path figure | Valid and unusually costly for a response-shape paper. The frozen outline already required response and resolution figures, so this is an omitted planned deliverable rather than a new reviewer-driven hypothesis. | major presentation | ACCEPT_FIGURE | P2 figure done | All three passing cells shown with deterministic median-proximity selection. |
| C5 | Thresholds and binary Pass appear arbitrary | Partly valid. Pre-inference freezing addresses researcher degrees of freedom, not universal practical significance. A sensitivity map can show stability but cannot retroactively justify the original values. | medium | ACCEPT_ANALYSIS + SOFTEN_CLAIM | P2 map done; interpretation bounded | Preserve effect sizes and intervals as primary; keep Pass as the registered decision only. |
| C6 | Quantile robustness, WQL, and epsilon are underreported | Epsilon is a real reporting error and is corrected. WQL already exists in stored P1 arrays and should be reported. Quantile-wise response distances require new inference because quantile trajectories were not archived. Their interpretation also differs from a structural individual response. | mixed | ACCEPT_TEXT + ACCEPT_ANALYSIS + PARTIAL | epsilon and WQL done; quantile analysis deferred | Do not present differences between marginal forecast quantiles as individual causal effects. |
| C7 | TimesFM-3 patching may explain temporal distortion | Plausible analysis hypothesis. The official architecture uses 32-step patches, but a single-model association is not causal evidence. | medium/high | ACCEPT_ANALYSIS | optional P4; not addressed by P2 | A frozen resampling-by-resolution study is needed before making an architectural claim. |
| C8 | Add more backbones | Useful breadth, but cannot be inserted into the frozen P1 primary matrix after observing outcomes. | medium | PARTIAL | optional secondary extension | Any new checkpoint is a separately frozen secondary study. |
| C9 | Re-freeze transparent references | Already addressed. P1-REF was separately frozen before its execution and is explicitly post-primary descriptive evidence. | resolved | CLARIFY_EXISTING | VERIFIED_DONE | Keep the distinction prominent in Methods and Results. |
| C10 | Add complete appendix and artifact details | Valid and mostly an assembly task because hashes, seed-level units, model revisions, WQL arrays, and construct checks already exist. | medium | ACCEPT_TEXT + ACCEPT_FIGURE | TODO_TEXT | Supplementary material should preserve complete adverse and null cells. |

## What should not be done

- Do not replace the registered primary metric or recompute the P1 decision with a more
  favorable definition.
- Do not relax the frozen thresholds after seeing outcomes.
- Do not describe semi-synthetic contexts as real-world causal evidence.
- Do not add one attractive hand-picked trajectory. Use a deterministic rule and retain
  every registered passing cell.
- Do not claim that TimesFM-3 patch length causes the observed pattern without a frozen
  resolution experiment.
- Do not spend the next iteration mainly polishing prose. The main acceptance risk is
  evidence breadth and metric necessity.

## Ordered revision program

### P2: analysis-only robustness supplement, no model inference

- fixed-fine-normalization signed distance and its coarsening contraction proposition;
- common metric comparison on analytic fixtures;
- complete WQL complement from stored arrays;
- threshold-neighborhood map based on the immutable cell intervals;
- deterministic representative trajectories for the three registered passing cells;
- explicit audit proving that P1 source arrays and decisions were not modified.

### P3: semi-synthetic realistic-context extension

- select three public context sources with different frequency/domain structure;
- inject two frozen response families into the full context and future while retaining
  realistic baseline variation;
- predeclare windows, scaling, intervention magnitudes, exclusions, checkpoints,
  uncertainty, and a descriptive success rule before inference;
- evaluate the same two backbones first; treat any additional backbone as optional.

### P4: optional mechanism analyses

- sampling-resolution sensitivity motivated by temporal patching;
- quantile-level response analysis with an explicit distributional estimand;
- one or two additional covariate-native checkpoints under a new frozen contract.

P2 and P3 are the acceptance-critical increments. P4 should not delay submission unless
the selected venue or subsequent review specifically requires it.
