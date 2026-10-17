"""Price a validated car with the final model.

Python does all the math here. The LLM only phrases the numbers that
build_facts puts together.

    estimator = PriceEstimator.load()          # models/final_model + models/error_stats.json
    result = estimator.estimate(validation.car)
    facts = build_facts(validation.car, result, validation.assumptions, validation.warnings)

What you get back for one car:
    price         the model's estimate, rounded to the nearest 100 dollars
    low, high     a likely range (about 8 in 10 held-out test listings landed inside
                  a range built this way, see src/error_stats.py)
    drivers       what happens to the price if one thing changes (miles, year, title)
    model_support common, rare or unseen: how much the model saw of this exact model
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import mlflow.sklearn
import pandas as pd

from src.preprocess import load_config, prepare_features
from src.validate import LAST_DATA_YEAR, MODEL_COLUMNS

MODEL_DIR = Path("models/final_model")
STATS_PATH = Path("models/error_stats.json")

PRICE_STEP = 100   # prices are shown to the nearest 100 dollars
DIFF_STEP = 50     # price changes are shown to the nearest 50 dollars

# "what moves the price" tries these step sizes, smallest first. The model is a tree
# ensemble, so a one-year or 20,000-mile step can come out the wrong way round (a 2017
# priced below a 2016). A change is only reported if it points the expected way and
# moves the price by at least MIN_EFFECT, otherwise the next bigger step is tried.
MILES_STEPS = [20_000, 40_000, 60_000]
YEAR_STEPS = [1, 2, 3]
MIN_EFFECT = 0.03


@dataclass
class Estimate:
    price: int
    low: int
    high: int
    level: float                 # share of test listings inside the range (0.8)
    model_support: str           # "common", "rare" or "unseen"
    drivers: list = field(default_factory=list)
    caveats: list = field(default_factory=list)


def _round_to(value, step):
    return int(round(value / step) * step)


def _inner_model(model):
    """The Pipeline inside the TransformedTargetRegressor (or the model itself)."""
    return getattr(model, "regressor_", model)


def model_keys(model):
    """Return (frequent, seen): model keys the encoder kept, and every key it saw."""
    encoder = _inner_model(model).named_steps["preprocess"].named_steps["encode"]
    onehot = encoder.named_transformers_["model_key"]
    seen = set(onehot.categories_[0])
    infrequent = onehot.infrequent_categories_[0]
    frequent = seen - set(infrequent) if infrequent is not None else set(seen)
    return frequent, seen


def model_key_for(car):
    """The manufacturer + model text the Pipeline builds for this car."""
    return prepare_features(pd.DataFrame([car])).loc[0, "model_key"]


def load_stats(path=STATS_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.error_stats` once to create it "
            "(it is also written whenever you run `python -m src.train_final`)."
        )
    with open(path) as f:
        return json.load(f)


class PriceEstimator:
    def __init__(self, model, stats, cfg, frequent_keys, seen_keys):
        self.model = model
        self.stats = stats
        self.cfg = cfg                      # data section of config.yaml
        self.frequent_keys = frequent_keys
        self.seen_keys = seen_keys

    @classmethod
    def load(cls, model_dir=MODEL_DIR, stats_path=STATS_PATH, cfg=None):
        model = mlflow.sklearn.load_model(str(model_dir))
        frequent, seen = model_keys(model)
        return cls(model, load_stats(stats_path), cfg or load_config(), frequent, seen)

    def predict(self, cars):
        """Predicted prices (dollars, unrounded) for a list of car dicts."""
        frame = pd.DataFrame(cars, columns=MODEL_COLUMNS)
        return [float(p) for p in self.model.predict(frame)]

    def support_for(self, car):
        key = model_key_for(car)
        if key in self.frequent_keys:
            return "common"
        return "rare" if key in self.seen_keys else "unseen"

    def estimate(self, car):
        """Estimate for one validated car (every model column present, None for blanks)."""
        candidates = self._candidates(car)
        predictions = self.predict([car] + [changed for _, _, _, changed in candidates])
        base, others = predictions[0], predictions[1:]

        support = self.support_for(car)
        errors = self.stats["common" if support == "common" else "other"]
        low = _round_to(base * errors["low_ratio"], PRICE_STEP)
        high = _round_to(base * errors["high_ratio"], PRICE_STEP)
        price = _round_to(base, PRICE_STEP)

        return Estimate(
            price=price,
            low=min(low, price),
            high=max(high, price),
            level=self.stats["level"],
            model_support=support,
            drivers=_pick_drivers(candidates, others, base),
            caveats=_support_caveats(support),
        )

    def _candidates(self, car):
        """The same car with one thing changed: (group, label, expected sign, car)."""
        out = []

        def add(group, label, sign, **changes):
            out.append((group, label, sign, {**car, **changes}))

        miles, year = car["odometer"], car["year"]
        for step in MILES_STEPS:
            if miles + step <= self.cfg["MAX_ODOMETER"]:
                add("more_miles", f"{step:,} more miles", -1, odometer=miles + step)
            if miles - step >= self.cfg["MIN_ODOMETER"]:
                add("fewer_miles", f"{step:,} fewer miles", +1, odometer=miles - step)
        for step in YEAR_STEPS:
            plural = "model year" if step == 1 else "model years"
            if year + step <= LAST_DATA_YEAR:
                add("newer", f"{step} {plural} newer", +1, year=year + step)
            if year - step >= self.cfg["MIN_YEAR"]:
                add("older", f"{step} {plural} older", -1, year=year - step)

        title = car["title_status"]
        if title == "clean":
            add("title", "a rebuilt title", -1, title_status="rebuilt")
        elif title in ("rebuilt", "salvage"):
            add("title", "a clean title", +1, title_status="clean")
        return out


def _pick_drivers(candidates, predictions, base):
    """Per group, the smallest step that moves the price the expected way by enough."""
    chosen = {}
    for (group, label, sign, _), value in zip(candidates, predictions):
        change = value - base
        if group in chosen or change * sign <= 0 or abs(change) < MIN_EFFECT * base:
            continue
        chosen[group] = {
            "if_instead": label,
            "new_estimate": _round_to(value, PRICE_STEP),
            "price_change": _round_to(change, DIFF_STEP),
        }
    order = ["more_miles", "fewer_miles", "newer", "older", "title"]
    return [chosen[group] for group in order if group in chosen]


def _support_caveats(support):
    if support == "rare":
        return ["There are only a few listings of this exact model in the data, "
                "so this estimate is less reliable than usual."]
    if support == "unseen":
        return ["This exact model is not in the data, so the estimate leans on year, "
                "mileage, body type and the other details and is less reliable than usual."]
    return []


READABLE = {
    "year": "year", "manufacturer": "make", "model": "model", "odometer": "mileage",
    "type": "body type", "drive": "drive", "fuel": "fuel", "transmission": "transmission",
    "condition": "condition", "title_status": "title status",
}


def build_facts(car, estimate, assumptions=(), warnings=()):
    """Everything the LLM may mention, as plain data. It should add nothing else."""
    percent = round(estimate.level * 100)
    return {
        "car": {READABLE[name]: car[name] for name in MODEL_COLUMNS if car.get(name) is not None},
        "estimated_price_dollars": estimate.price,
        "likely_range_dollars": {
            "low": estimate.low,
            "high": estimate.high,
            "meaning": f"in testing, about {percent} percent of real listing prices "
                       "fell inside a range built this way",
        },
        "what_moves_the_price": estimate.drivers,
        "price_basis": "typical asking price in 2021 Craigslist listings, "
                       "not a guaranteed sale price",
        "assumptions": list(assumptions),
        "caveats": list(estimate.caveats) + list(warnings),
    }
