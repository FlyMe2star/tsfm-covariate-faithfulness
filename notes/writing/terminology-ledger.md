# Manuscript terminology ledger

| Canonical term | First-use definition | Variants avoided | Decision |
|---|---|---|---|
| time-series foundation model (TSFM) | Pretrained forecasting model evaluated without task-specific fitting | time series FM; forecasting foundation model | Expand once, then use TSFM. |
| CovIntervene-SHAPE | Paired multi-resolution audit introduced in this work | P1-SHAPE; shape audit | Use the method name in manuscript prose; retain P1-SHAPE only for artifact names. |
| known-future covariate | Exogenous input whose forecast-horizon path is supplied to the model | future covariate; exogenous path | Use the canonical term unless contrasting past covariates. |
| paired structural replay | Factual and intervened worlds with shared history and innovations | paired replay; counterfactual replay | Use the full term on first use. |
| oracle response | Structural target difference between intervened and factual worlds, $\Delta^\star$ | true response; ground-truth effect | Use oracle response to avoid implying observational causal identification. |
| predicted response | Difference between median forecasts under intervened and factual worlds, $\widehat{\Delta}$ | model effect; forecast effect | Use predicted response. |
| Directional Sign Agreement (DSA) | Sign agreement on oracle-active horizons | sign accuracy | Expand once, then DSA. |
| Response Gain Ratio (RGR) | Predicted-to-oracle absolute response mass on active horizons | gain ratio | Expand once, then RGR. |
| signed shape distance $D_b$ | Signed $L_1$-normalized response distance after width-$b$ block aggregation | shape error; TV distance | Use $D_b$; specify $D_1$ and $D_8$ when relevant. |
| hidden-distortion gap $G$ | $D_1-D_8$ | resolution gap; hidden gap | Expand once, then $G$. |
| scaled quantile loss (SQL) | Quantile loss scaled by context first-difference MAE | scaled pinball loss | Expand once, then SQL. |
| weighted quantile loss (WQL) | Quantile loss normalized by target absolute mass | normalized quantile loss | Expand once, then WQL. |
| fitted transparent reference | Ridge-ARX or nonlinear dynamic regression fitted from context rows | baseline model; zero-shot baseline | Always mark as fitted and post-primary. |

No unresolved terminology collision remains. Model names follow their official forms:
Chronos-2, TimesFM-3, Ridge-ARX, and nonlinear dynamic regression.

