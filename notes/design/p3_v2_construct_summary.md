# P3-v2 construct preflight: verified, not a model result

The single owner-approved P3-v2 CPU run passed its unchanged construct gate.
The public [report](../../evidence/p3_semisynthetic/v2_construct/p3_v2_construct_preflight.json)
matches the private unit receipt and source-ID selection hashes. It contains
no Chronos-2 or TimesFM-3 predictions and cannot support a response-faithfulness
or forecasting-performance claim.

| Real target background | Biphasic exclusions / 96 | Dispersed-delay exclusions / 96 |
|---|---:|---:|
| Traffic | 0 | 1 |
| Workload | 1 | 0 |
| Solar | 1 | 2 |

The total is 5 exclusions among 576 attempted scenarios, all for oracle
$L_1$/factual $L_1$ below the unchanged 0.005 minimum. Every source-by-family
cell stays below the unchanged 5% ceiling. The other 571 scenarios passed
paired-context identity, exact oracle subtraction, finite nonnegative targets,
placebo structural zero with a nonzero perturbation, and deterministic replay.
The 96 selected original source IDs have zero overlap with the 144 IDs in the
failed v1 pilot: 32 new IDs per source, each with three new synthetic seeds.
No source ID or origin was replaced after an oracle-mass check.

The v1 pilot failure and its influence on v2 timing and intervention amplitude
remain part of the evidence record. “Construct-held-out” does not mean the
three original data collections were unseen in the data-only v1 audit.
Passing this gate merely permits a **separate owner review of the model-inference
freeze**. Until that freeze is approved and the run harness passes its smoke
checks, the P3-v2 checkpoint calls remain unauthorized. P1/P2 results and
thresholds are unchanged.

The private selection and unit files are in ignored
`tmp/p3_v2_construct_preflight/`; only their SHA-256 values are public.
Reproduction with the pinned local Parquet files:

```text
py -3.14 -m pip install -r requirements/p3-preflight.txt
py -3.14 -m covfaith_p3_v2.preflight --repo . --data-root tmp/p3_data_preflight --private-root tmp/p3_v2_construct_preflight
```
