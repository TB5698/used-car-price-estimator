from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.error_stats import compute_error_stats
from src.estimate import (MODEL_DIR, STATS_PATH, PriceEstimator, build_facts,
                          load_stats, model_keys)
from src.preprocess import build_preprocessor
from src.validate import MODEL_COLUMNS

CFG = {"MIN_YEAR": 1990, "MAX_ODOMETER": 300000, "MIN_ODOMETER": 100}
STATS = {
    "level": 0.8,
    "common": {"low_ratio": 0.8, "high_ratio": 1.25},
    "other": {"low_ratio": 0.7, "high_ratio": 1.4},
}

CAR = dict(
    year=2016, manufacturer="toyota", model="camry", odometer=90000, condition=None,
    fuel="gas", title_status="clean", transmission="automatic", drive="fwd", type="sedan",
)


class FakeModel:
    """Prices a car from year, mileage and title with simple rules. No training needed."""

    def predict(self, frame):
        price = 30000 - 1500 * (2022 - frame["year"]) - 0.05 * frame["odometer"]
        return np.where(frame["title_status"] == "rebuilt", price * 0.7, price).astype(float)


def estimator(model=None, frequent=("toyota camry",), seen=("toyota camry", "ford rare")):
    return PriceEstimator(model or FakeModel(), STATS, CFG, set(frequent), set(seen))


def test_estimate_is_rounded_and_inside_its_range():
    result = estimator().estimate(CAR)
    # 30000 - 1500 * 6 - 0.05 * 90000 = 16500
    assert result.price == 16500
    assert result.low == 13200 and result.high == 20600
    assert result.low <= result.price <= result.high
    assert result.level == 0.8


def test_support_levels():
    est = estimator()
    assert est.support_for(CAR) == "common"
    assert est.support_for({**CAR, "manufacturer": "ford", "model": "rare"}) == "rare"
    assert est.support_for({**CAR, "model": "made up name"}) == "unseen"
    assert est.support_for({**CAR, "model": " Camry "}) == "common"


def test_rare_and_unseen_models_get_a_wider_range_and_a_caveat():
    common = estimator().estimate(CAR)
    unseen = estimator().estimate({**CAR, "model": "made up name"})
    assert unseen.model_support == "unseen"
    assert (unseen.high - unseen.low) / unseen.price > (common.high - common.low) / common.price
    assert unseen.caveats and not common.caveats
    rare = estimator().estimate({**CAR, "manufacturer": "ford", "model": "rare"})
    assert rare.model_support == "rare" and rare.caveats


def test_drivers_point_the_expected_way():
    drivers = {d["if_instead"]: d for d in estimator().estimate(CAR).drivers}
    assert drivers["20,000 more miles"]["price_change"] == -1000
    assert drivers["20,000 fewer miles"]["price_change"] == 1000
    assert drivers["1 model year newer"]["price_change"] == 1500
    assert drivers["1 model year older"]["price_change"] == -1500
    assert drivers["a rebuilt title"]["price_change"] == -4950
    assert drivers["a rebuilt title"]["new_estimate"] == 11600


def test_noisy_small_step_falls_through_to_a_bigger_one():
    class Noisy(FakeModel):
        # a one-year-newer car comes out cheaper, like a tree model sometimes does
        def predict(self, frame):
            price = super().predict(frame)
            return np.where(frame["year"] == 2017, price - 3000, price)

    labels = [d["if_instead"] for d in estimator(Noisy()).estimate(CAR).drivers]
    assert "1 model year newer" not in labels
    assert "2 model years newer" in labels


def test_changes_that_barely_move_the_price_are_left_out():
    class Flat(FakeModel):
        def predict(self, frame):
            return np.full(len(frame), 10000.0)

    assert estimator(Flat()).estimate(CAR).drivers == []


def test_steps_outside_the_data_are_not_tried():
    newest = [d["if_instead"] for d in estimator().estimate({**CAR, "year": 2022}).drivers]
    assert not any("newer" in label for label in newest)
    oldest = [d["if_instead"] for d in estimator().estimate({**CAR, "year": 1990}).drivers]
    assert not any("older" in label for label in oldest)
    high_miles = [d["if_instead"] for d in estimator().estimate({**CAR, "odometer": 299000}).drivers]
    assert not any("more miles" in label for label in high_miles)
    low_miles = [d["if_instead"] for d in estimator().estimate({**CAR, "odometer": 5000}).drivers]
    assert not any("fewer miles" in label for label in low_miles)


def test_title_driver_goes_the_other_way_for_a_rebuilt_title():
    drivers = {d["if_instead"]: d for d in estimator().estimate({**CAR, "title_status": "rebuilt"}).drivers}
    assert "a rebuilt title" not in drivers
    assert drivers["a clean title"]["price_change"] > 0
    # no title comparison for a lien
    lien = [d["if_instead"] for d in estimator().estimate({**CAR, "title_status": "lien"}).drivers]
    assert not any("title" in label for label in lien)


def test_build_facts_has_only_real_numbers_and_the_caveats():
    est = estimator()
    result = est.estimate({**CAR, "model": "made up name"})
    car = {**CAR, "model": "made up name"}
    facts = build_facts(car, result, assumptions=["Assuming the title status is clean."],
                        warnings=["AWD treated as 4wd."])
    assert facts["car"]["make"] == "toyota" and "condition" not in facts["car"]
    assert facts["estimated_price_dollars"] == result.price
    assert facts["likely_range_dollars"]["low"] == result.low
    assert "80 percent" in facts["likely_range_dollars"]["meaning"]
    assert facts["what_moves_the_price"] == result.drivers
    assert facts["assumptions"] == ["Assuming the title status is clean."]
    assert len(facts["caveats"]) == 2  # model not in the data, plus the validation warning
    assert "2021" in facts["price_basis"]


def tiny_pipeline():
    """A small real Pipeline, so model_keys and error stats run against sklearn objects."""
    rows = []
    for name, count in [("camry", 60), ("corolla", 60), ("oddball", 3)]:
        for i in range(count):
            rows.append({**CAR, "model": name, "year": 2005 + i % 15, "odometer": 20000 + 1000 * i})
    X = pd.DataFrame(rows, columns=MODEL_COLUMNS)
    y = 30000 - 1200 * (2022 - X["year"]) - 0.04 * X["odometer"]
    pipe = Pipeline([("preprocess", build_preprocessor(use_condition=True, min_frequency=5)),
                     ("model", Ridge())])
    return pipe.fit(X, y), X, y


def test_model_keys_read_from_a_fitted_pipeline():
    pipe, _, _ = tiny_pipeline()
    frequent, seen = model_keys(pipe)
    assert frequent == {"toyota camry", "toyota corolla"}
    assert seen == {"toyota camry", "toyota corolla", "toyota oddball"}


def test_estimator_with_a_real_pipeline_handles_an_unseen_model():
    pipe, _, _ = tiny_pipeline()
    frequent, seen = model_keys(pipe)
    est = PriceEstimator(pipe, STATS, CFG, frequent, seen)
    result = est.estimate({**CAR, "model": "never seen", "condition": None, "drive": None})
    assert result.model_support == "unseen"
    assert result.price > 0 and result.low <= result.price <= result.high


def test_error_stats_split_common_and_other_models():
    pipe, X, y = tiny_pipeline()
    stats = compute_error_stats(pipe, X, y)
    assert stats["level"] == 0.8
    assert stats["common"]["n"] + stats["other"]["n"] == len(X)
    assert stats["other"]["n"] == 3
    for group in ["common", "other"]:
        assert 0 < stats[group]["low_ratio"] <= stats[group]["high_ratio"]


def test_missing_stats_file_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="error_stats"):
        load_stats(tmp_path / "nope.json")


@pytest.mark.skipif(not (Path(MODEL_DIR).exists() and Path(STATS_PATH).exists()),
                    reason="needs models/final_model and models/error_stats.json")
def test_real_model_prices_the_demo_car():
    result = PriceEstimator.load().estimate(CAR)
    assert 8000 < result.price < 20000
    assert result.model_support == "common"
    assert result.low < result.price < result.high
