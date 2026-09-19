from covfaith.shape_constructs import evaluate_construct_fixtures, evaluate_generator_preflight


def test_construct_fixture_checks_pass() -> None:
    report = evaluate_construct_fixtures()
    assert all(report["checks"].values())


def test_small_generator_preflight_passes() -> None:
    report = evaluate_generator_preflight(seeds=(101,), series_per_mechanism_per_seed=2)
    for summary in report.values():
        assert summary["deterministic"]
        assert summary["paired_replay_valid"]
        assert summary["oracle_l1_minimum"] > 1e-4
