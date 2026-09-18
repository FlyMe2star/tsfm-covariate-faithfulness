# Frozen model-interface source note

Verification date: 2026-09-18

## Chronos-2

- Official repository: <https://github.com/amazon-science/chronos-forecasting>
- Package: `chronos-forecasting==2.2.2`
- Checkpoint: `amazon/chronos-2`
- Checkpoint revision: `29ec3766d36d6f73f0696f85560a422f50e8498c`
- Interface: `Chronos2Pipeline.predict` accepts a list of mappings containing a
  target, past covariates, and the future values of known-future covariates. Its
  output is target-by-quantile-by-horizon. The P0 adapter disables cross-learning.
- Code license: Apache-2.0. The checkpoint model card must also be retained with the
  run artifact.

## TimesFM-3

- Official repository: <https://github.com/google-research/timesfm>
- Package: `timesfm==2.0.2`
- Checkpoint: `google/timesfm-3.0-pytorch`
- Checkpoint revision: `43046b85ec22d584a13f8098c2ed39c889e129c2`
- Interface: `TimesFM3Evaluator.predict_batch` accepts target contexts and
  past-and-future covariate arrays with shape covariates-by-context-plus-horizon.
- Code license: Apache-2.0.
- Weight restriction: `timesfm-non-commercial-license-v1.0`; the P0 study is academic,
  non-commercial research and must not represent the checkpoint as commercially usable.

## Reproducibility boundary

The revisions above were previously resolved from successful isolated model smoke
runs. P0 must fail closed if the requested revision cannot be downloaded or if the
runtime interface does not match the adapter tests. No fallback to a moving `main`
revision is allowed.
