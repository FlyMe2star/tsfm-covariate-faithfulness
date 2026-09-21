# Verified P1-REF result summary

Status: **verified post-primary supplementary descriptive evidence**

Execution time: 2026-09-21T10:06:45.338900Z

Supplement config hash: `330e2317ffb7dcead45e4ef83bf830388ed835b1eb887ad59ae54eb7e5a22bb5`

Supplement scientific-code hash: `a921a7858c663bb17e93c06398393ab4a5c3193b71c76dbaccd4f75d0bcf0ed5`

Archived report SHA-256: `c3ed3ebfb62a36413b59094d1bf158bc0b78f55e3a4038d2beadead48e2f543c`

## Integrity and scope

All 12 frozen mechanism-by-seed units completed, producing 1,536
series-model records and eight model-by-mechanism summaries. The report matches the
frozen supplement and primary config/code hashes. It records that no continuation
gate was computed, no primary decision was recomputed, no primary threshold was
revised, and no TSFM inference was performed.

This analysis was designed and frozen after the primary result. It is descriptive
context only: it cannot change the P1-SHAPE decision or establish a preregistered
reference-versus-TSFM comparison.

## Bounded findings

- Ridge-ARX closely recovered the two additive temporal mechanisms. For biphasic
  rebound and dispersed delayed pulse, DSA was 0.999 and 0.993, RGR was 0.995 and
  1.002, and fine-resolution signed shape distance was 0.038 and 0.057.
- The same linear reference was less faithful on path-dependent hysteresis
  (`D_1=0.243`) and could not represent the registered synergy interaction
  (interaction distance `1.000`). This confirms that the audit does not impose a
  uniformly unattainable shape criterion while retaining genuinely harder mechanisms.
- The frozen nonlinear feature expansion reduced synergy interaction distance from
  `1.000` to `0.300`, with DSA/RGR changing from `0.784/0.724` to `0.871/1.042`.
  It did not dominate Ridge-ARX: on the two additive mechanisms its `D_1` rose from
  `0.038--0.057` to `0.117--0.151`, and its SQL was also higher.
- The references therefore contextualize mechanism difficulty, but they do not
  support a universal model ranking or invalidate the bounded primary finding.

The complete bootstrap intervals, unit hashes, and private artifact hashes are in
[`p1_reference_report.json`](p1_reference_report.json).
