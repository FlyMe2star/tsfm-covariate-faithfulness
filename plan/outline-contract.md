# Outline contract

Status: topic approved; contribution hypotheses frozen for P0 design; no model inference yet

## Provisional empirical story

1. **Problem and distinction** — Covariate-aware TSFMs are evaluated mainly by
   observational forecast error, but scenario use also requires faithful response to
   changes in known-future covariates.
2. **Audit formulation** — Hold target history and all non-intervened inputs fixed;
   change one future covariate under a structural generator with a known target response.
3. **CovIntervene protocol** — Cover contemporaneous, delayed, nonlinear saturation,
   and placebo mechanisms with scale-aware paired metrics.
4. **Behavioral evidence** — Compare Chronos-2, TimesFM-3, and a third reproducible
   covariate-aware model across response fidelity and forecast skill.
5. **Practical boundary** — Use real FEV tasks only for plausible sensitivity checks;
   do not label observational changes as causal effects.
6. **Optional safeguard** — Include only if separately authorized and verified on
   untouched evidence after P0.
7. **Limitations and deployment guidance** — Bound conclusions to tested mechanisms,
   covariate types, horizons, and public frozen backbones.

## Planned sections and evidence roles

1. **Introduction** — distinguish forecast accuracy, attribution, robustness, and
   response faithfulness; state only verified findings in the final version.
2. **Related Work** — TSFMs, covariate-aware forecasting, behavioral/causal model
   audits, feature attribution, and forecast robustness.
3. **Problem Formulation** — define factual and intervened covariate paths, paired
   forecast responses, oracle structural effects, and non-causal real-data probes.
4. **CovIntervene** — generators, intervention families, metrics, aggregation,
   uncertainty, invariance tests, and leakage controls.
5. **Experimental Protocol** — models, horizons, compute, P0/P1 separation, baselines,
   and frozen decision rules.
6. **Results** — skill-versus-fidelity matrix, mechanism profiles, model differences,
   and all negative or null cells.
7. **Analysis** — normalization, scale, intervention support, horizon localization,
   and optional safeguard ablations.
8. **Limitations and Conclusion** — synthetic-to-real boundary and precise deployment
   implications.

## Required figures and tables

- Figure 1: factual versus intervened future-covariate paths and expected response.
- Figure 2: mechanism families and metric definitions.
- Figure 3: skill-versus-faithfulness scatter or quadrant plot.
- Figure 4: horizon-wise response profiles by backbone and mechanism.
- Table 1: models, covariate interfaces, licenses, and compute.
- Table 2: complete mechanism-by-model fidelity matrix with uncertainty.
- Table 3: invariance tests and optional safeguard ablations.

## Citation quotas

- Introduction and motivation: 7–10 verified citations.
- Related work: 18–24 verified citations across five clusters.
- Method/evaluation rationale: 6–10 verified citations.
- Target total: 30–40 verified references.

## Drafting and threshold rule

No abstract result sentence or contribution is written in factual tense until the
corresponding evidence row is `verified`. P0 continuation thresholds are fixed before
the first backbone invocation. Failure triggers a stop or a new untouched contract,
not a lower retrospective threshold.
