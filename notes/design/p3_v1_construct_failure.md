# P3-v1 CPU construct preflight: registered gate failed

Status: **verified adverse construct result, no forecasting model inference**.
The owner approved the design on 2026-09-25. The config was then locked by
`configs/p3_semisynthetic/p3_design_approval.json`; the implementation does not
modify P1 code or change the approved P3 config. The public aggregate receipt is
`evidence/p3_semisynthetic/construct/p3_v1_construct_preflight.json`. The exact
source-ID/origin selection and unit diagnostics remain in ignored
`tmp/p3_construct_preflight/`, linked by SHA-256 in that receipt.

| Background | Biphasic exclusions / 144 | Dispersed-delay exclusions / 144 |
|---|---:|---:|
| Traffic | 0 (0.00%) | 10 (6.94%) |
| Workload | 2 (1.39%) | 10 (6.94%) |
| Solar | 5 (3.47%) | 11 (7.64%) |

All 38 exclusions had the same reason: exact oracle $L_1$ response was below
0.005 of factual future $L_1$. The predeclared maximum was 5% per background-by-
family cell, so three dispersed-delay cells fail. This is a construct signal
criterion, not a model performance result. It does not imply those backgrounds
or counterfactuals are semantically invalid. No other exclusion reason was
observed. The 48 original IDs per source were fixed before scenario generation;
none was replaced after an oracle check.

The tested implementation uses the approved 0.65 context-SD covariate pulse,
the unchanged P1 kernel parameter ranges, multiplier cap 1.4, and the original
0.005 / 5% construct gate. The failure is spread over multiple seeds and source
IDs, especially for the dispersed-delay mechanism, rather than a single bad
record. Unit tests verify future-target-blind selection, timestamp cadence,
paired context identity, deterministic replay, exact oracle subtraction,
placebo structural zero, and rejection of nonfinite selected futures. There is
no evidence of an implementation defect that would justify correcting this run
in place.

**Decision:** do not create a P3 model-inference freeze or run the checkpoints.
Do not lower the 0.005 threshold, relax the 5% rule, change source IDs, or
increase the intervention/link strength within P3-v1 to force a pass. Preserve
this attempt as a failed construct pilot. A new P3-v2 mechanism and evaluation
contract would require its own owner review and an explicit account of this
pilot; alternatively, report the bounded P3-v1 failure as a limitation and
prioritize the complete P1/P2 appendix. Neither choice changes the P1 result.

Reproduction on a machine with the three pinned Parquet files:

```text
py -3.14 -m pip install -r requirements/p3-preflight.txt
py -3.14 -m covfaith_p3.preflight --repo . --data-root tmp/p3_data_preflight --private-root tmp/p3_construct_preflight
```

The command intentionally exits with code 2 when the construct gate fails.
