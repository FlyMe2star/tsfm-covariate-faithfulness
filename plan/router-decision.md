# Router decision

Route to **empirical-paper-writer** for an empirical benchmark and behavioral-audit paper.

The primary contribution requires new paired intervention experiments, uncertainty
estimation, metric validation, and cross-backbone comparisons. It therefore cannot be
supported as a literature-only review. The default paper is evaluation-first; a new
method is optional rather than required for continuation.

## Why this route is comparatively low-risk

- P0 requires no training and can run on a T4.
- Synthetic mechanisms provide analytic response ground truth.
- The existing FEV, Chronos-2, and TimesFM-3 execution knowledge is reusable without
  reusing the old sealed outcomes.
- A verified benchmark finding can support the paper even if no safeguard is added.
- The gate can fail cheaply before A100-scale expansion.

## Downstream order

1. Design and unit-test generators and metrics without invoking a TSFM.
2. Freeze the P0 configuration and code hash.
3. Run the two-backbone P0 pilot on T4.
4. Archive the complete output before choosing an evaluation-only or safeguard branch.
5. Route verified evidence to the empirical paper drafting and QA workflow.
