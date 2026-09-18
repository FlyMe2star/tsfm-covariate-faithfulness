from dataclasses import dataclass

import numpy as np

from covfaith.adapters import Chronos2Adapter, TimesFM3Adapter
from covfaith.generators import generate_scenario


class FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self.value


class FakeChronosPipeline:
    quantiles = [0.1, 0.5, 0.9]

    def __init__(self) -> None:
        self.last_inputs = None

    def predict(self, inputs, prediction_length, **kwargs):
        self.last_inputs = inputs
        values = np.zeros((1, 3, prediction_length), dtype=np.float32)
        values[0, 1] = 2.0
        return [FakeTensor(values) for _ in inputs]


@dataclass
class FakeTimesFMOutput:
    forecast: np.ndarray
    quantiles: np.ndarray


class FakeTimesFMEvaluator:
    def __init__(self) -> None:
        self.last_kwargs = None

    def predict_batch(self, **kwargs):
        self.last_kwargs = kwargs
        horizon = kwargs["horizon"]
        for _ in kwargs["contexts"]:
            yield FakeTimesFMOutput(
                forecast=np.full(horizon, 3.0, dtype=np.float32),
                quantiles=np.zeros((horizon, 9), dtype=np.float32),
            )


def test_chronos_adapter_builds_paired_covariate_records() -> None:
    scenario = generate_scenario("contemporaneous_linear", 11, 0)
    pipeline = FakeChronosPipeline()
    adapter = Chronos2Adapter(pipeline)
    result = adapter.forecast([scenario], "intervened")
    assert result.median.shape == (1, 24)
    assert result.quantiles.shape == (1, 24, 3)
    assert pipeline.last_inputs[0]["target"].shape == (192,)
    assert set(pipeline.last_inputs[0]["future_covariates"]) == {"active", "placebo"}
    np.testing.assert_array_equal(
        pipeline.last_inputs[0]["future_covariates"]["active"],
        scenario.covariate_future_intervened[0].astype(np.float32),
    )


def test_chronos_target_only_omits_covariates() -> None:
    scenario = generate_scenario("contemporaneous_linear", 11, 0)
    pipeline = FakeChronosPipeline()
    adapter = Chronos2Adapter(pipeline)
    adapter.forecast([scenario], "target_only")
    assert isinstance(pipeline.last_inputs[0], np.ndarray)


def test_timesfm_adapter_concatenates_context_and_selected_future() -> None:
    scenario = generate_scenario("irrelevant_placebo", 29, 1)
    evaluator = FakeTimesFMEvaluator()
    adapter = TimesFM3Adapter(evaluator)
    result = adapter.forecast([scenario], "intervened")
    assert result.median.shape == (1, 24)
    assert result.quantiles.shape == (1, 24, 9)
    supplied = evaluator.last_kwargs["past_future_covariates"][0]
    assert supplied.shape == (2, 216)
    np.testing.assert_array_equal(
        supplied[:, 192:],
        scenario.covariate_future_intervened.astype(np.float32),
    )


def test_timesfm_target_only_omits_covariates() -> None:
    scenario = generate_scenario("threshold_saturation", 47, 2)
    evaluator = FakeTimesFMEvaluator()
    adapter = TimesFM3Adapter(evaluator)
    adapter.forecast([scenario], "target_only")
    assert "past_future_covariates" not in evaluator.last_kwargs
