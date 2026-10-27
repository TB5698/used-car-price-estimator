"""The training settings come from config.yaml, not from the code."""
import pytest
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from src.train import build_regressor, load_experiments, load_final_spec

REQUIRED_KEYS = {"name", "model", "params", "log_target", "use_condition", "min_frequency"}


def test_config_has_at_least_five_different_experiments():
    experiments = load_experiments()
    assert len(experiments) >= 5
    names = [spec["name"] for spec in experiments]
    assert len(set(names)) == len(names)
    settings = {repr((spec["model"], sorted(spec["params"].items()), spec["log_target"],
                      spec["use_condition"], spec["min_frequency"])) for spec in experiments}
    assert len(settings) == len(experiments)  # no two runs with identical settings


def test_every_experiment_has_all_its_settings():
    for spec in load_experiments():
        assert REQUIRED_KEYS <= set(spec), spec["name"]


def test_build_regressor_uses_the_hyperparameters_from_the_config():
    ridge = build_regressor("ridge", {"alpha": 3.0}, seed=1)
    assert isinstance(ridge, Ridge) and ridge.alpha == 3.0

    lgbm = build_regressor("lightgbm", {"n_estimators": 7, "num_leaves": 5}, seed=1)
    assert isinstance(lgbm, LGBMRegressor)
    assert lgbm.n_estimators == 7 and lgbm.num_leaves == 5 and lgbm.random_state == 1


def test_unknown_model_type_is_rejected():
    with pytest.raises(ValueError):
        build_regressor("random_forest", {}, seed=1)


def test_every_configured_experiment_can_be_built():
    for spec in load_experiments():
        build_regressor(spec["model"], spec["params"], seed=42)


def test_final_model_uses_the_settings_of_the_experiment_it_names():
    final = load_final_spec()
    source = {spec["name"]: spec for spec in load_experiments()}["lgbm_log_price_deeper"]
    assert final["name"] == "final_lgbm_full_data"
    assert final["model"] == source["model"] and final["params"] == source["params"]
    assert final["log_target"] == source["log_target"]
