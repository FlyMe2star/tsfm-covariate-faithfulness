import numpy as np
import pytest

from covfaith.generators import MECHANISMS, GeneratorSpec, generate_scenario


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_generation_is_deterministic(mechanism: str) -> None:
    first = generate_scenario(mechanism, 11, 7)
    second = generate_scenario(mechanism, 11, 7)
    assert first.series_id == second.series_id
    np.testing.assert_array_equal(first.target_context, second.target_context)
    np.testing.assert_array_equal(
        first.covariate_future_intervened,
        second.covariate_future_intervened,
    )
    np.testing.assert_array_equal(first.oracle_response, second.oracle_response)


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_pairing_changes_exactly_one_future_covariate(mechanism: str) -> None:
    scenario = generate_scenario(mechanism, 29, 4)
    assert scenario.target_context.shape == (192,)
    assert scenario.covariate_context.shape == (2, 192)
    assert scenario.covariate_future_factual.shape == (2, 24)
    assert scenario.covariate_future_intervened.shape == (2, 24)
    changed_rows = np.any(
        scenario.covariate_future_factual != scenario.covariate_future_intervened,
        axis=1,
    )
    expected = (
        np.array([True, False])
        if mechanism != "irrelevant_placebo"
        else np.array([False, True])
    )
    np.testing.assert_array_equal(changed_rows, expected)
    changed_index = int(np.flatnonzero(expected)[0])
    np.testing.assert_array_equal(
        scenario.covariate_future_factual[changed_index, ~scenario.intervention_mask],
        scenario.covariate_future_intervened[changed_index, ~scenario.intervention_mask],
    )


@pytest.mark.parametrize(
    "mechanism",
    ["contemporaneous_linear", "delayed_distributed_lag", "threshold_saturation"],
)
def test_active_oracle_matches_structural_target_difference(mechanism: str) -> None:
    scenario = generate_scenario(mechanism, 47, 10)
    np.testing.assert_allclose(
        scenario.oracle_response,
        scenario.target_future_intervened - scenario.target_future_factual,
        atol=0.0,
        rtol=0.0,
    )
    assert np.linalg.norm(scenario.oracle_response, ord=1) > 0.0


def test_placebo_has_zero_truth_and_nonzero_matched_active_reference() -> None:
    scenario = generate_scenario("irrelevant_placebo", 11, 5)
    np.testing.assert_array_equal(scenario.oracle_response, np.zeros(24))
    np.testing.assert_array_equal(scenario.target_future_intervened, scenario.target_future_factual)
    assert np.linalg.norm(scenario.matched_active_oracle_response, ord=1) > 0.0


def test_distributed_lag_has_predeclared_onset() -> None:
    spec = GeneratorSpec(first_lag=2)
    scenario = generate_scenario("delayed_distributed_lag", 11, 0, spec)
    np.testing.assert_allclose(scenario.oracle_response[:2], 0.0, atol=1e-12)
    assert np.any(np.abs(scenario.oracle_response[2:]) > 1e-10)
