"""Thin frozen-model adapters for the common CovIntervene scenario schema."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from covfaith.generators import PairedScenario

Float32Array = NDArray[np.float32]
Variant = Literal["target_only", "factual", "intervened"]


@dataclass(frozen=True)
class ForecastBundle:
    median: Float32Array
    quantiles: Float32Array | None
    quantile_levels: tuple[float, ...]


class ScenarioAdapter(Protocol):
    backbone_id: str

    def forecast(self, scenarios: Sequence[PairedScenario], variant: Variant) -> ForecastBundle: ...


def _validate_scenarios(scenarios: Sequence[PairedScenario]) -> int:
    if not scenarios:
        raise ValueError("at least one scenario is required")
    horizons = {scenario.target_future_factual.size for scenario in scenarios}
    if len(horizons) != 1:
        raise ValueError("all scenarios in one call must share a horizon")
    return horizons.pop()


def _future_covariates(scenario: PairedScenario, variant: Variant) -> Float32Array:
    if variant == "factual":
        values = scenario.covariate_future_factual
    elif variant == "intervened":
        values = scenario.covariate_future_intervened
    else:
        raise ValueError("target_only has no future covariates")
    return values.astype(np.float32, copy=False)


class Chronos2Adapter:
    """Chronos-2 adapter using its official list-of-dictionaries API."""

    backbone_id = "chronos_2"

    def __init__(self, pipeline: Any, *, batch_size: int = 128) -> None:
        self.pipeline = pipeline
        self.batch_size = batch_size

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        revision: str,
        *,
        device: str = "cuda",
        batch_size: int = 128,
    ) -> Chronos2Adapter:
        import torch
        from chronos import BaseChronosPipeline

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the frozen Chronos-2 P0 run")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        pipeline = BaseChronosPipeline.from_pretrained(
            checkpoint,
            revision=revision,
            device_map=device,
            dtype=dtype,
        )
        return cls(pipeline, batch_size=batch_size)

    def _inputs(self, scenarios: Sequence[PairedScenario], variant: Variant) -> list[Any]:
        if variant == "target_only":
            return [
                scenario.target_context.astype(np.float32, copy=False)
                for scenario in scenarios
            ]
        records: list[dict[str, Any]] = []
        for scenario in scenarios:
            past = {
                name: scenario.covariate_context[index].astype(np.float32, copy=False)
                for index, name in enumerate(scenario.covariate_names)
            }
            future_values = _future_covariates(scenario, variant)
            future = {
                name: future_values[index]
                for index, name in enumerate(scenario.covariate_names)
            }
            records.append(
                {
                    "target": scenario.target_context.astype(np.float32, copy=False),
                    "past_covariates": past,
                    "future_covariates": future,
                }
            )
        return records

    def forecast(self, scenarios: Sequence[PairedScenario], variant: Variant) -> ForecastBundle:
        horizon = _validate_scenarios(scenarios)
        if variant not in {"target_only", "factual", "intervened"}:
            raise ValueError(f"unknown variant {variant!r}")
        predictions = self.pipeline.predict(
            self._inputs(scenarios, variant),
            prediction_length=horizon,
            batch_size=self.batch_size,
            cross_learning=False,
        )
        levels = tuple(float(level) for level in self.pipeline.quantiles)
        median_index = int(np.argmin(np.abs(np.asarray(levels) - 0.5)))
        if abs(levels[median_index] - 0.5) > 1e-8:
            raise RuntimeError("Chronos-2 output does not include the median quantile")
        quantiles: list[Float32Array] = []
        medians: list[Float32Array] = []
        for prediction in predictions:
            array = np.asarray(prediction.detach().float().cpu(), dtype=np.float32)
            if array.shape != (1, len(levels), horizon):
                raise RuntimeError(f"unexpected Chronos-2 prediction shape {array.shape}")
            quantiles.append(np.transpose(array[0], (1, 0)))
            medians.append(array[0, median_index])
        return ForecastBundle(
            median=np.stack(medians),
            quantiles=np.stack(quantiles),
            quantile_levels=levels,
        )


class TimesFM3Adapter:
    """TimesFM-3 adapter using its native past-and-future covariate interface."""

    backbone_id = "timesfm_3"
    default_quantiles = tuple(float(level) for level in np.arange(0.1, 1.0, 0.1))

    def __init__(self, evaluator: Any) -> None:
        self.evaluator = evaluator

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        revision: str,
        *,
        device: str = "cuda",
        per_core_batch_size: int = 16,
    ) -> TimesFM3Adapter:
        import torch
        from huggingface_hub import snapshot_download
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the frozen TimesFM-3 P0 run")
        checkpoint_path = snapshot_download(repo_id=checkpoint, revision=revision)
        evaluator = TimesFM3Evaluator(
            ModelConfig(
                checkpoint_path=checkpoint_path,
                per_core_batch_size=per_core_batch_size,
                device=device,
            )
        )
        return cls(evaluator)

    def forecast(self, scenarios: Sequence[PairedScenario], variant: Variant) -> ForecastBundle:
        horizon = _validate_scenarios(scenarios)
        if variant not in {"target_only", "factual", "intervened"}:
            raise ValueError(f"unknown variant {variant!r}")
        contexts = [
            scenario.target_context.astype(np.float32, copy=False)
            for scenario in scenarios
        ]
        kwargs: dict[str, Any] = {
            "contexts": contexts,
            "horizon": horizon,
            "return_quantiles": True,
            "use_symmetric_averaging": False,
        }
        if variant != "target_only":
            kwargs["past_future_covariates"] = [
                np.concatenate(
                    [
                        scenario.covariate_context.astype(np.float32, copy=False),
                        _future_covariates(scenario, variant),
                    ],
                    axis=1,
                )
                for scenario in scenarios
            ]
        outputs = list(self.evaluator.predict_batch(**kwargs))
        medians: list[Float32Array] = []
        quantiles: list[Float32Array] = []
        for output in outputs:
            median = np.asarray(output.forecast, dtype=np.float32)
            quantile = np.asarray(output.quantiles, dtype=np.float32)
            if median.shape == (1, horizon):
                median = median[0]
            if quantile.shape == (1, horizon, len(self.default_quantiles)):
                quantile = quantile[0]
            if median.shape != (horizon,):
                raise RuntimeError(f"unexpected TimesFM-3 forecast shape {median.shape}")
            if quantile.shape != (horizon, len(self.default_quantiles)):
                raise RuntimeError(f"unexpected TimesFM-3 quantile shape {quantile.shape}")
            medians.append(median)
            quantiles.append(quantile)
        return ForecastBundle(
            median=np.stack(medians),
            quantiles=np.stack(quantiles),
            quantile_levels=self.default_quantiles,
        )
