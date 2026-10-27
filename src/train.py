"""Train used car price models and log every run to MLflow."""
import numpy as np
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import load_config, clean_data, build_preprocessor

RAW_COLUMNS = [
    "price", "year", "manufacturer", "model", "condition", "cylinders",
    "fuel", "odometer", "title_status", "transmission", "drive", "type",
    "paint_color", "state", "posting_date",
]


def load_train_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)["train"]


def load_experiments(path="config.yaml"):
    """The list of experiments (name, model, params, ...) from config.yaml."""
    with open(path) as f:
        return yaml.safe_load(f)["experiments"]


def load_final_spec(path="config.yaml"):
    """The experiment the final model is trained from, plus the final run name."""
    with open(path) as f:
        final = yaml.safe_load(f)["final_model"]
    by_name = {spec["name"]: spec for spec in load_experiments(path)}
    if final["from_experiment"] not in by_name:
        raise KeyError(f"final_model.from_experiment {final['from_experiment']!r} "
                       "is not one of the experiments in config.yaml")
    return {**by_name[final["from_experiment"]], "name": final["run_name"]}


def build_regressor(model_type, params, seed):
    """Make the regressor named in config.yaml with its hyperparameters."""
    if model_type == "ridge":
        return Ridge(**params)
    if model_type == "lightgbm":
        return LGBMRegressor(**params, random_state=seed, verbose=-1)
    raise ValueError(f"unknown model type {model_type!r} (use ridge or lightgbm)")


def load_split(data_cfg, train_cfg):
    """Return X_train, X_test, y_train, y_test from the cleaned, sampled data."""
    df = pd.read_csv(data_cfg["raw_path"], usecols=RAW_COLUMNS)
    df = clean_data(df, data_cfg)

    sample_size = train_cfg["SAMPLE_SIZE"]
    seed = train_cfg["RANDOM_SEED"]
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)

    X = df.drop(columns="price")
    y = df["price"]
    return train_test_split(
        X, y, test_size=train_cfg["TEST_SIZE"], random_state=seed
    )


def evaluate(y_true, y_pred):
    """Metrics in dollars: MAE, RMSE, R2."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def run_experiment(run_name, regressor, data, train_cfg, log_target=False,
                   use_condition=False, min_frequency=25, params=None):
    """Fit one pipeline, log params, metrics and the model to MLflow.

    Returns the test metrics and the fitted model.
    """
    X_train, X_test, y_train, y_test = data

    # preprocessing and regressor live in one Pipeline
    pipe = Pipeline([
        ("preprocess", build_preprocessor(use_condition=use_condition,
                                          min_frequency=min_frequency)),
        ("model", regressor),
    ])
    # for log runs, wrap it so predictions still come out in dollars
    if log_target:
        model = TransformedTargetRegressor(
            regressor=pipe, func=np.log1p, inverse_func=np.expm1
        )
    else:
        model = pipe

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model_type": type(regressor).__name__,
            "log_target": log_target,
            "use_condition": use_condition,
            "min_frequency": min_frequency,
            "sample_size": train_cfg["SAMPLE_SIZE"],
            "n_train": len(X_train),
            **(params or {}),
        })

        model.fit(X_train, y_train)

        test_metrics = evaluate(y_test, model.predict(X_test))
        train_metrics = evaluate(y_train, model.predict(X_train))
        mlflow.log_metrics(test_metrics)
        mlflow.log_metric("train_mae", train_metrics["mae"])

        # cloudpickle, because skops rejects our custom prepare_features function
        mlflow.sklearn.log_model(
            model, name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    print(f"{run_name}: MAE {test_metrics['mae']:.0f}, "
          f"RMSE {test_metrics['rmse']:.0f}, R2 {test_metrics['r2']:.3f}")
    return test_metrics, model


def run_from_spec(spec, data, train_cfg):
    """Run one experiment described in config.yaml."""
    regressor = build_regressor(spec["model"], spec["params"], train_cfg["RANDOM_SEED"])
    return run_experiment(
        spec["name"], regressor, data, train_cfg,
        log_target=spec["log_target"],
        use_condition=spec["use_condition"],
        min_frequency=spec["min_frequency"],
        params=spec["params"],
    )


if __name__ == "__main__":
    data_cfg = load_config()
    train_cfg = load_train_config()
    data = load_split(data_cfg, train_cfg)

    mlflow.set_experiment(train_cfg["EXPERIMENT_NAME"])

    for spec in load_experiments():
        run_from_spec(spec, data, train_cfg)
