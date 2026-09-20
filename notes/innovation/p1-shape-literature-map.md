# P1-SHAPE literature and novelty map

Search date: 2026-09-20. Sources below are primary paper, proceedings, journal, or
official model pages. The submission bibliography and per-claim ledger are now in
`paper/ref.bib` and `notes/innovation/submission-literature-ledger.csv`.

## Nearest work

| Work | What it covers | Remaining boundary for P1-SHAPE |
|---|---|---|
| [Causal Analysis for Time Series Foundation Models](https://arxiv.org/abs/2608.24303) | Intervenes on parameters of six univariate synthetic generators and tests whether Chronos-2 and TimesFM-2.5 preserve scalar time-series patterns. | Closest threat. It does not audit a target forecast's signed horizon-wise response to an intervened known-future exogenous path or compare that response across temporal resolutions. |
| [Interventional Time Series Priors for Causal Foundation Models](https://arxiv.org/abs/2603.11090) | Generates paired observational and interventional temporal structural causal models for training causal prior-fitted networks. | Its target is in-context causal-effect estimation and model training; P1 is a black-box behavioral audit of frozen forecasters with analytic response trajectories. |
| [Certified Interventional Fidelity](https://proceedings.mlr.press/v337/asiaee26d.html) | Defines causal estimands and anytime-valid inference for interventions on internal model components. | It supplies useful language for interventional fidelity but studies mechanistic-interpretability interventions, not future exogenous inputs or forecast response shape. |
| [Chronos-2](https://arxiv.org/abs/2510.15821) | Introduces a zero-shot universal forecaster with native target, related-series, and covariate handling through group attention. | Establishes one audited interface and reports forecasting performance; it does not provide a structural response-shape audit. |
| [TimesFM-3 official release](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) | Provides a second public zero-shot architecture with future-covariate support. | Establishes the second audited interface; the release is accuracy-centered rather than an intervention-faithfulness benchmark. |
| [fev-bench](https://arxiv.org/abs/2509.26468) | Evaluates 100 forecasting tasks, including covariate-rich tasks, with standardized accuracy metrics. | Real-task breadth does not supply the structural target response to a controlled covariate intervention. |
| [ChronosX](https://proceedings.mlr.press/v258/arango25a.html) | Adapts pretrained forecasting models to past and future exogenous variables. | Optimizes covariate-informed forecast accuracy and requires adaptation; P1 audits frozen native interfaces and response functions. |
| [COSMIC](https://arxiv.org/abs/2506.03128) | Trains a zero-shot covariate-aware forecaster using informative covariate augmentation and in-context learning. | Demonstrates effective covariate use through accuracy and qualitative analysis, not analytic intervention-response fidelity. |
| [UniCA](https://arxiv.org/abs/2506.22039) | Adapts TSFMs to homogeneous and heterogeneous covariates through representation and attention modules. | It is an adaptation method evaluated for forecast accuracy; P1 does not train an adapter and asks a different behavioral question. |
| [TFMAdapter](https://arxiv.org/abs/2509.13906) | Proposes lightweight instance-level covariate adaptation and shows that covariates can be underused. | Motivates effective-use tests but does not compare a predicted response trajectory with structural ground truth. |
| [CITRAS-FM](https://arxiv.org/abs/2606.10798) | Pretrains a compact model for univariate, multivariate, and covariate-informed zero-shot forecasting. | Expands the covariate-aware model class but evaluates predictive accuracy rather than the signed temporal response to controlled interventions. |
| [ApolloPFN](https://arxiv.org/abs/2603.15802) | Uses synthetic priors and time-aware architecture changes for zero-shot forecasting with exogenous variables. | It is a trained covariate-aware forecaster; P1 instead evaluates whether fixed public checkpoints reproduce known structural response shape. |
| [Counterfactual Explanation for Multivariate Time Series Forecasting with Exogenous Variables](https://arxiv.org/abs/2511.06906) | Searches exogenous-variable changes that produce desired forecasts and analyzes variable influence. | It generates explanatory counterfactuals; P1 supplies controlled inputs and audits whether the resulting response is structurally faithful. |
| [ForecastCF](https://arxiv.org/abs/2310.08137) | Generates counterfactual examples for forecasting models and evaluates validity and plausibility. | Counterfactual generation and manifold proximity are different from known-ground-truth response-shape fidelity. |
| [CounTS](https://proceedings.mlr.press/v202/yan23d.html) | Builds self-interpretable time-series prediction with counterfactual explanations. | Its contribution is prediction explanation, not a frozen TSFM behavioral audit under exogenous interventions. |
| [Exploring Representations and Interventions in TSFMs](https://proceedings.mlr.press/v267/wilinski25a.html) | Studies internal representations, redundancy, learned concepts, and representation steering. | Internal concept analysis does not test the external structural response to a future-covariate path. |
| [TIMING](https://proceedings.mlr.press/v267/jang25a.html) | Develops temporality-aware integrated gradients for time-series explanation. | Attribution faithfulness is not equivalent to fidelity of the model's intervention-response function. |
| [Framework for Evaluating Faithfulness of Local Explanations](https://proceedings.mlr.press/v162/dasgupta22a.html) | Formalizes consistency and sufficiency criteria for local explanation faithfulness. | Useful evaluation precedent, but it neither addresses forecasting nor provides temporal structural-response metrics. |
| [Exogenous Dropout](https://arxiv.org/abs/2607.05452) | Studies robustness to corrupted, missing, and misaligned exogenous inputs and proposes a training method. | Robustness to bad inputs is not fidelity to a valid, controlled intervention; generic corruption remains out of scope. |

## Defensible gap

The novelty claim is intentionally narrow:

> Existing work studies covariate-aware accuracy, generated counterfactual
> explanations, internal representation interventions, univariate generator-parameter
> responses, or robustness to corrupted covariates. P1-SHAPE asks whether the full
> signed target-response trajectory to a known-future exogenous intervention matches
> structural ground truth, and whether temporal aggregation hides its distortion.

The project must not claim that no prior work studies TSFM interventions, covariate use,
or interventional fidelity. It may claim, subject to a final pre-submission search, that
the verified set does not provide this exact combination: a black-box audit of frozen
covariate-aware TSFMs, paired future-exogenous worlds, analytic signed response paths,
and a test of distortion hidden by temporal aggregation.

## Novelty threats and controls

1. **Overlap with causal TSFM analysis.** Keep the intervention object, response object,
   and metric object explicit: exogenous future path, target forecast trajectory, and
   signed multi-resolution distance.
2. **Overlap with counterfactual explanation.** Do not optimize an input to reach a
   desired output. Use frozen, generator-defined paired worlds.
3. **Metric proliferation.** One primary fine metric and one primary resolution-gap
   contrast; onset, peak, and transport distances remain diagnostics.
4. **Synthetic toy criticism.** Include mechanisms with rebound, path dependence, and
   interaction; report parameter envelopes and transparent fitted references.
5. **Universal-claim criticism.** Name the tested checkpoints and restrict conclusions
   to them.

## Search refresh checkpoints

- The implementation-freeze refresh was completed on 2026-09-20. It added
  CausalTimePrior, Certified Interventional Fidelity, COSMIC, UniCA, CITRAS-FM, and
  ApolloPFN; none invalidated the narrow boundary above.
- Refresh again before manuscript submission.
- Search clusters: `future covariate intervention response trajectory`, `TSFM causal
  analysis covariate`, `forecast counterfactual exogenous`, `temporal response shape
  faithfulness`, and `metamorphic testing forecasting`.
