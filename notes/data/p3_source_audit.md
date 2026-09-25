# P3 public-background source audit (data only)

Status: candidate source audit; no checkpoint loaded and no P3 model outcome inspected.
Audit date: 2026-09-25. Raw Parquet files are under ignored `tmp/p3_data_preflight/`
locally and must not be committed or copied into a public artifact.

The candidate source is the `autogluon/fev_datasets` repository at immutable revision
`aee7d2c576582040339379984cc777abc7181919`. The [repository card](https://huggingface.co/datasets/autogluon/fev_datasets/blob/aee7d2c576582040339379984cc777abc7181919/README.md)
documents the unified format and source lineage. The three exact Parquet files were
downloaded from that revision and checked locally:

| Configuration / target | Domain / cadence | Original series | Length range | Raw SHA-256 | Source |
|---|---|---:|---:|---|---|
| `LOOP_SEATTLE_1D` / `target` | road traffic / daily | 323 | 365–365 | `2f2e49f88c31f10e1e9040e7e0791ec179a953820fb67f3e284264a1f7aa1657` | [GIFT-Eval](https://huggingface.co/datasets/Salesforce/GiftEval) |
| `redset_5T` / `target` | database workload / 5 min | 118 | 5,518–26,210 | `022d25f70c67d08935ad08069ac9326c0bc43921030d4e8721ffee4d0aa9cd19` | [Redset](https://github.com/amazon-science/redset) |
| `solar_10T` / `target` | solar production / 10 min | 137 | 52,560–52,560 | `0c3e80f5ac9edadaea8799abd79bd88cb5752599af5d5d4ef264c2a188551772` | [GIFT-Eval](https://huggingface.co/datasets/Salesforce/GiftEval) |

All inspected target values are finite and nonnegative. Median whole-series zero
fractions are 0, 0.114, and 0.552, respectively. Solar therefore cannot be treated
as an always-active target, and sparse Redset instances need a predeclared coverage
filter. No native dataset covariates are fed to the model in this extension: the
background target is real, while the covariate and its causal response are injected.

CPU-only feasibility scan: candidate origins are `range(max(192, floor(0.60*N)),
N-24+1, 24)` for each source series. Selection requires finite target values in
the 192-step **context only**, exact nominal timestamp cadence across the 216-step
window, context target standard deviation above
`1e-6`, and context positive fraction at least 0.95 / 0.90 / 0.30 for traffic /
Redset / solar. Redset additionally requires its last 24 context values to be
positive; solar uses the origin timestamp's clock hour in `{10,11,12}`. These
rules use **only historical target values and known timestamps**, never future
target values. Future-target finiteness is checked only after selection, without
replacement of an ID or origin. At least one eligible origin was found for 323 / 82 / 137 original
series, respectively. No candidate window failed the cadence check. Thus, 48
source-ID clusters per dataset are feasible without seeing any model outputs.

For risk assessment only, we inspected the 24 future baseline values **after**
applying the deterministic ID/origin selection: all 48 selected traffic windows
were positive throughout; 7/48 workload and 2/48 solar windows contained at
least one zero, and only one workload window had fewer than half its future values
positive. These future values did **not** affect selection. Exact response-mass
eligibility remains a separate CPU construct check before any checkpoint call.
These counts are descriptive data-only preflight evidence, not performance results.

Licensing boundary: the [FEV card](https://huggingface.co/datasets/autogluon/fev_datasets)
labels this converted collection `other` and directs users to original terms; it
describes the datasets as research-purpose material unless otherwise specified.
[Redset states CC BY-NC 4.0](https://github.com/amazon-science/redset).
The original terms for the GIFT-Eval traffic and solar derivatives still need
source-level confirmation before any raw-data redistribution. Until then, publish
only code, source IDs/hashes as permitted, aggregate metrics, and derived figures;
never upload the Parquet files or reconstructed raw target paths to GitHub.

The older immutable revision is intentional: later collection revisions removed
the high-frequency `solar_10T` copy. A dataset-availability or terms failure before
freeze requires a new candidate design and owner review, not a silent source swap.
