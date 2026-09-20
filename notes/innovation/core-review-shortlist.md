# Core literature review shortlist

Search frozen for drafting on 2026-09-20. All links below were checked against a
primary paper, proceedings page, or official model page. This is the small set for
owner review; the full 30-item ledger is in `submission-literature-ledger.csv`.

## A. Novelty boundary: read these first

| Priority | Work | What to check | Owner decision |
|---|---|---|---|
| 1 | [Causal Analysis for Time Series Foundation Models](https://arxiv.org/abs/2608.24303) | Confirm that its intervention is on univariate generator parameters rather than a known-future exogenous path and that it does not test multi-resolution response distortion. | pending |
| 2 | [Interventional Time Series Priors for Causal Foundation Models](https://arxiv.org/abs/2603.11090) | Confirm that it trains causal PFNs for effect estimation rather than auditing frozen forecasting checkpoints. | pending |
| 3 | [Certified Interventional Fidelity](https://proceedings.mlr.press/v337/asiaee26d.html) | Check that the intervention object is internal model state/components and that our use of “interventional fidelity” remains carefully distinguished. | pending |
| 4 | [Exploring Representations and Interventions in TSFMs](https://proceedings.mlr.press/v267/wilinski25a.html) | Confirm that it studies internal representations and steering rather than external future-covariate response paths. | pending |

## B. Covariate-aware forecasting: verify positioning

| Priority | Work | What to check | Owner decision |
|---|---|---|---|
| 5 | [Chronos-2](https://arxiv.org/abs/2510.15821) | Native interface, group attention, and claims made for covariate-aware accuracy. | pending |
| 6 | [TimesFM-3 official release](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) | Native past-future covariate interface and lookahead token construction. | pending |
| 7 | [ChronosX](https://proceedings.mlr.press/v258/arango25a.html) | Synthetic covariate benchmark and fitted adapter boundary. | pending |
| 8 | [COSMIC](https://arxiv.org/abs/2506.03128) | Whether any evaluation directly compares a signed forecast response path to analytic ground truth. | pending |
| 9 | [TFMAdapter](https://arxiv.org/abs/2509.13906) | Evidence for covariate underuse and how it differs from our multi-resolution audit. | pending |
| 10 | [ApolloPFN](https://arxiv.org/abs/2603.15802) | Synthetic-prior training and exogenous support; confirm different method/evaluation target. | pending |

## C. Explanation and testing precedents: skim

| Priority | Work | What to check | Owner decision |
|---|---|---|---|
| 11 | [TIMING](https://proceedings.mlr.press/v267/jang25a.html) | Signed temporal attribution metrics versus a causal response trajectory. | pending |
| 12 | [Counterfactual Explanation for Forecasting with Exogenous Variables](https://arxiv.org/abs/2511.06906) | Optimized explanatory input changes versus fixed generator-defined paired worlds. | pending |
| 13 | [CheckList](https://aclanthology.org/2020.acl-main.442/) | Black-box directional testing as methodological precedent, without claiming a forecasting predecessor. | pending |

## Review rule

For each item, replace `pending` with `must cite`, `background only`, or `exclude`, and
add a one-line correction only if the boundary statement is wrong. Full-paper reading
is unnecessary unless a boundary appears wrong; abstracts, methods, and evaluation
sections are sufficient for this pass.
