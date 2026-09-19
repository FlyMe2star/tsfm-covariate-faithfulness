import numpy as np
import pytest

from covfaith.shape_generators import SHAPE_MECHANISMS, generate_shape_scenario
from covfaith.shape_metrics import interaction_response


@pytest.mark.parametrize("mechanism", SHAPE_MECHANISMS)
def test_shape_generation_is_deterministic_and_paired(mechanism: str) -> None:
    first = generate_shape_scenario(mechanism, 101, 7)
    second = generate_shape_scenario(mechanism, 101, 7)
    np.testing.assert_array_equal(first.target_context, second.target_context)
    np.testing.assert_array_equal(first.oracle_response, second.oracle_response)
    np.testing.assert_array_equal(
        first.oracle_response,
        first.target_future_intervened - first.target_future_factual,
    )
    assert first.target_context.shape == (192,)
    assert first.covariate_context.shape == (2, 192)
    assert first.covariate_future_factual.shape == (2, 24)
    assert np.sum(np.abs(first.oracle_response)) > 1e-4


@pytest.mark.parametrize(
    "mechanism",
    ["biphasic_rebound", "dispersed_delayed_pulse", "rate_asymmetric_hysteresis"],
)
def test_single_covariate_mechanisms_change_only_covariate_a(mechanism: str) -> None:
    scenario = generate_shape_scenario(mechanism, 307, 4)
    changed = np.any(
        scenario.covariate_future_intervened != scenario.covariate_future_factual, axis=1
    )
    np.testing.assert_array_equal(changed, [True, False])


def test_synergy_exposes_four_worlds_and_nonzero_interaction() -> None:
    scenario = generate_shape_scenario("two_covariate_synergy", 911, 3)
    assert set(scenario.covariate_future_worlds) == {"factual", "a_only", "b_only", "joint"}
    interaction = interaction_response(
        scenario.oracle_response_worlds["joint"],
        scenario.oracle_response_worlds["a_only"],
        scenario.oracle_response_worlds["b_only"],
    )
    assert np.sum(np.abs(interaction)) > 1e-4
