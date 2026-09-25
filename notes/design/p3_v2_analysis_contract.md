# P3-v2 sealed-array analysis contract

The owner pasted completion receipts for both frozen backbones on 2026-09-25:
18 of 18 full units and 571 valid scenarios per backbone, with identical
configuration, selection, construct-report, and runner-code hashes. These are
completion claims, not yet verified model outcomes. No P3-v2 result is added
to the manuscript at this stage.

The separate CPU-only analysis replays source-ID/origin selection and all 576
construct scenarios from pinned private Parquet, then verifies each of the 36
full prediction arrays and its manifest against its SHA-256 and exact scenario
order, factual target, paired oracle, sham subset, and lower-link oracle. It
refuses incomplete or modified archives and never calls either checkpoint.

For each of the twelve backbone × source × family cells, it computes DSA,
RGR, signed shape distances at widths 1/2/4/8, registered gap G=D1−D8,
relative SQL/WQL against target-only, and fixed first-12-ID sham and lower-link
controls. It uses the frozen P1 thresholds for a **descriptive** complete-cell
flag and reports every cell regardless of outcome. Uncertainty resamples
original source IDs with all valid synthetic seeds together (5,000 percentile
replicates, seed 250926). Transfer support requires at least one complete cell
in at least two distinct sources. The P1 primary decision is never rerun.

The inference archive contains lower-link medians and the exact lower-link
oracle, but not lower-link quantile arrays. Therefore its preregistered
lower-link relative SQL is **unavailable**, not zero or imputed. The main
12-cell SQL/WQL complement and sham control remain fully archived. The
analysis report and CSV must explicitly retain this omission, the failed
P3-v1 construct pilot, and all null/adverse P3-v2 cells. Private source IDs,
windows, and forecast arrays remain outside public Git.
