# Literature map

Search date: 2026-09-18. This map records the nearest work used to bound the
novelty hypothesis; it is not yet a submission bibliography.

| Work | What it establishes | Boundary relative to this project |
|---|---|---|
| [Chronos-2](https://arxiv.org/abs/2510.15821) | A frozen TSFM can natively handle targets, related series, and past/future covariates through group attention. | Establishes the capability being audited; it primarily reports forecast performance rather than controlled response fidelity. |
| [TimesFM-3](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) | Adds native multivariate and past-future covariate support with lookahead token construction. | Provides a recently released architecture with a distinct covariate pathway and a natural cross-model comparison. |
| [fev-bench](https://arxiv.org/abs/2509.26468) | Provides 100 forecasting tasks including 46 with covariates and principled accuracy aggregation. | Supplies real-task infrastructure; does not provide structural ground truth for how forecasts should respond to a covariate intervention. |
| [ChronosX](https://proceedings.mlr.press/v258/arango25a.html) | Adapts pretrained forecasters to past and future exogenous variables and evaluates synthetic and real tasks. | Focuses on effective covariate incorporation and accuracy, not audit metrics for response direction/localization/placebos. |
| [TFMAdapter](https://arxiv.org/abs/2509.13906) | Shows that nominal covariate support may underuse even an oracle target-as-covariate and proposes a local adapter. | Strong motivation for auditing effective use, but not a cross-backbone structural response benchmark. |
| [CITRAS-FM](https://arxiv.org/abs/2606.10798) | Uses synthetic covariates for covariate-informed zero-shot pretraining in a small model. | Candidate third backbone and evidence that synthetic covariate structures matter; not itself a faithfulness audit. |
| [Explainable Load Forecasting with Covariate-Informed TSFMs](https://arxiv.org/abs/2604.28149) | Uses masking-based SHAP explanations for Chronos-2 and TabPFN-TS on one load task. | Explains observational predictions in one domain; does not compare model response with known interventional ground truth. |
| [Causal Analysis for Time Series Foundation Models](https://arxiv.org/abs/2608.24303) | Intervenes on six synthetic target-series generators to expose generic TSFM pattern biases. | Closest conceptual neighbor. The present project must remain narrower: interventions act on future covariate paths while target history is fixed. |
| [Exogenous Dropout](https://arxiv.org/abs/2607.05452) | Benchmarks noisy, missing, and misaligned exogenous inputs and proposes a strong training baseline. | Occupies corruption robustness; this project excludes generic corruption as its primary novelty. |
| [TSFM Benchmarking Challenges](https://arxiv.org/abs/2510.13654) | Identifies leakage, representativeness, and out-of-sample validity risks in TSFM benchmarks. | Supports the need for controlled evaluation but does not define covariate-response metrics. |

## Gap statement

The defensible gap is not “TSFMs need covariates,” “covariates can be noisy,” or
“TSFMs require explanations.” It is the narrower measurement question: when a known
future covariate is changed under a data-generating mechanism with a known response,
does a frozen TSFM produce the correct counterfactual forecast change? The primary
evidence must jointly report forecast skill and response fidelity so that neither can
stand in for the other.

## Search clusters for the next pass

- future covariate intervention / response fidelity / scenario forecasting
- TSFM feature attribution / masking / counterfactual explanation
- synthetic structural time-series generators with known exogenous effects
- metamorphic testing and behavioral testing of forecasting models
- invariance and equivariance tests for normalized multivariate forecasters
