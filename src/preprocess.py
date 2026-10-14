import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

USE_COLS = [
    "price", "year", "manufacturer", "model", "condition", "fuel",
    "odometer", "title_status", "transmission", "drive", "type",
]

NUMERIC = ["year", "odometer"]
CATEGORICAL = ["manufacturer", "title_status", "fuel", "transmission", "drive", "type"]
TEXT_COLS = ["manufacturer", "model", "title_status", "fuel", "transmission",
             "drive", "type", "condition"]


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)["data"]


def clean_data(df, cfg):
    """Apply the row-cleaning rules from the EDA notebook."""
    df = df.copy()

    # required fields: a row without these is not useful
    df = df.dropna(subset=["price", "year", "odometer"])

    df = df[(df["price"] >= cfg["MIN_PRICE"]) & (df["price"] <= cfg["MAX_PRICE"])]
    df = df[df["year"] >= cfg["MIN_YEAR"]]
    df = df[df["odometer"] <= cfg["MAX_ODOMETER"]]

    # near-zero odometer on an older car is a placeholder, not a real reading
    placeholder = (df["odometer"] < cfg["MIN_ODOMETER"]) & (df["year"] < cfg["NEW_CAR_YEAR"])
    df = df[~placeholder]

    df = df[~df["title_status"].isin(cfg["DROP_TITLE_STATUS"])]

    df["type"] = df["type"].replace({"truck": "pickup"})
    df["year"] = df["year"].astype(int)
    return df


def prepare_features(df):
    """Fill gaps, normalize text, and build the manufacturer + model feature."""
    df = df.copy()
    for col in TEXT_COLS:
        df[col] = df[col].fillna("unknown").astype(str).str.lower().str.strip()
    df["type"] = df["type"].replace({"truck": "pickup"})
    df["model_key"] = df["manufacturer"] + " " + df["model"]
    return df


def build_preprocessor(use_condition=False, min_frequency=25):
    """Preprocessing steps as a Pipeline. The regressor gets added in train.py."""
    cat_cols = CATEGORICAL + (["condition"] if use_condition else [])
    encoder = ColumnTransformer([
        ("model_key",
         OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency),
         ["model_key"]),
        ("cats", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("nums", StandardScaler(), NUMERIC),
    ])
    return Pipeline([
        ("prepare", FunctionTransformer(prepare_features)),
        ("encode", encoder),
    ])


if __name__ == "__main__":
    cfg = load_config()
    raw = pd.read_csv(cfg["raw_path"], usecols=USE_COLS)
    clean = clean_data(raw, cfg)
    print(len(raw), "->", len(clean), "rows")

    pre = build_preprocessor()
    X = pre.fit_transform(clean.drop(columns=["price"]))
    print("feature matrix:", X.shape)