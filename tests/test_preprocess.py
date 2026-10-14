import pandas as pd

from src.preprocess import build_preprocessor, clean_data, prepare_features

CFG = {
    "MIN_PRICE": 500,
    "MAX_PRICE": 100000,
    "MIN_YEAR": 1990,
    "MAX_ODOMETER": 300000,
    "MIN_ODOMETER": 100,
    "NEW_CAR_YEAR": 2020,
    "DROP_TITLE_STATUS": ["parts only", "missing"],
}

BASE = dict(
    price=10000, year=2015, manufacturer="ford", model="f-150",
    condition="good", fuel="gas", odometer=80000.0, title_status="clean",
    transmission="automatic", drive="4wd", type="pickup",
)


def row(**overrides):
    d = dict(BASE)
    d.update(overrides)
    return d


def make_df(*rows):
    return pd.DataFrame(list(rows))


def test_price_limits():
    df = make_df(row(price=100), row(price=10000), row(price=500000))
    out = clean_data(df, CFG)
    assert out["price"].tolist() == [10000]


def test_year_rules():
    df = make_df(row(year=1985), row(year=2015), row(year=None))
    out = clean_data(df, CFG)
    assert out["year"].tolist() == [2015]


def test_odometer_rules():
    df = make_df(
        row(odometer=50, year=2012),    # old car, placeholder reading: dropped
        row(odometer=50, year=2021),    # new car, low mileage is real: kept
        row(odometer=400000),           # over the cap: dropped
        row(odometer=None),             # missing: dropped
        row(odometer=90000),            # normal: kept
    )
    out = clean_data(df, CFG)
    assert sorted(out["odometer"].tolist()) == [50, 90000]


def test_title_status_rules():
    df = make_df(
        row(title_status="parts only"),
        row(title_status="missing"),
        row(title_status=None),         # empty stays, it is optional
        row(title_status="rebuilt"),
    )
    out = clean_data(df, CFG)
    assert len(out) == 2
    assert set(out["title_status"].dropna()) == {"rebuilt"}


def test_truck_merged_and_year_is_int():
    out = clean_data(make_df(row(type="truck")), CFG)
    assert out["type"].iloc[0] == "pickup"
    assert out["year"].dtype.kind == "i"


def test_prepare_features_fills_and_builds_key():
    df = make_df(row(manufacturer="Ford", model=" F-150 ", fuel=None))
    out = prepare_features(df)
    assert out["model_key"].iloc[0] == "ford f-150"
    assert out["fuel"].iloc[0] == "unknown"


def test_pipeline_handles_unseen_model_and_missing_fields():
    train = make_df(
        *[row(manufacturer="toyota", model="camry") for _ in range(3)],
        *[row() for _ in range(3)],
        row(model="rare"),
    )
    pre = build_preprocessor(min_frequency=2)
    X_train = pre.fit_transform(train.drop(columns=["price"]))

    new = make_df(row(manufacturer="tesla", model="roadster", drive=None))
    X_new = pre.transform(new.drop(columns=["price"]))
    assert X_new.shape[1] == X_train.shape[1]