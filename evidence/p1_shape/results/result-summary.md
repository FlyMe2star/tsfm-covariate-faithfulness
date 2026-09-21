# Verified P1-SHAPE result summary

Status: **eligible for a bounded paper claim**

Decision time: 2026-09-19T13:07:43.545451Z

Config hash: `27826ad34bfe0acd8ee90ff4d0fbafb011a6b00aeea276dacaa9f9fe2a261918`

Scientific code hash: `fc1cf6d73a59a4200deaa964f341a17b72d8fae75fa40b1086ed1c8c714f5925`

Decision artifact SHA-256: `077322e808fffe62af99feda3363220670eac96ba3f43eb6ef0758da2a7373a7`

## Frozen decision

The predeclared P1-SHAPE gate passed. Three of eight backbone-by-mechanism
cells satisfied all four cell-level conditions:

1. Chronos-2 / biphasic rebound;
2. Chronos-2 / dispersed delayed pulse; and
3. TimesFM-3 / biphasic rebound.

The cells span two structural mechanisms and both backbones. The minimum cell-count,
diversity, and complementary forecast-accuracy conditions all passed.

## Bounded primary finding

Within the three passing cells, covariate-aware SQL was 51.4%--52.3% lower than the
matched target-only call, DSA estimates were 0.973--0.982, and the complete median RGR
intervals remained inside the frozen `[0.65, 1.35]` coarse envelope. Nevertheless,
fine-resolution signed shape distance estimates were 0.185--0.198, with lower 95%
bounds of 0.173--0.189. The hidden-distortion gap estimates were 0.040--0.066, with
lower bounds of 0.032--0.061. Thus, under these tested mechanisms and checkpoints,
temporal aggregation concealed part of a reproducible horizon-wise response-shape
error despite strong forecast skill and credible coarse response behavior.

## Complete-matrix boundary

- All eight cells crossed the frozen fine-shape-distance bound.
- Six cells crossed the hidden-gap bound; both hysteresis cells did not.
- TimesFM-3 / dispersed delayed pulse missed the complete rule only because its RGR
  lower bound was `0.615`, below the frozen `0.65` coarse-eligibility boundary.
- Both synergy cells failed coarse DSA and RGR eligibility and therefore cannot support
  the primary “hidden behind coarse correctness” claim.
- Neither model is claimed to be universally better.

## Evidence limits

The primary evidence covers two frozen public TSFM checkpoints, four synthetic
structural mechanisms, a 24-step horizon, and three generator seeds. It does not
establish real-world causal effects. Transparent fitted dynamic-regression references
have since completed under an independently frozen, post-primary protocol; they are
descriptive context and cannot modify this decision. Any observational plausibility
study remains unverified.
