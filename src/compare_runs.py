"""Compare the finished MLflow runs for the used car experiment.

Run from the repo root with the venv active:
    python -m src.compare_runs
"""
import mlflow
import pandas as pd

from src.train import load_train_config

COLUMNS = {
    "tags.mlflow.runName": "run",
    "params.model_type": "model",
    "params.log_target": "log_price",
    "params.use_condition": "condition",
    "params.min_frequency": "min_freq",
    "metrics.mae": "test_mae",
    "metrics.rmse": "test_rmse",
    "metrics.r2": "test_r2",
    "metrics.train_mae": "train_mae",
}


def load_runs(experiment_name):
    """Return one row per finished run (the latest if a name was run twice)."""
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )
    if runs.empty:
        raise SystemExit(f"No finished runs found for experiment '{experiment_name}'.")

    runs = runs.drop_duplicates(subset="tags.mlflow.runName", keep="first")
    table = runs[list(COLUMNS)].rename(columns=COLUMNS)
    # share of extra error on the test set compared to the training set
    table["overfit_gap"] = (table["test_mae"] - table["train_mae"]) / table["train_mae"]
    return table.sort_values("test_mae").reset_index(drop=True)


def main():
    train_cfg = load_train_config()
    table = load_runs(train_cfg["EXPERIMENT_NAME"])

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(table.to_string(
        index=False,
        formatters={
            "test_mae": "{:,.0f}".format,
            "test_rmse": "{:,.0f}".format,
            "test_r2": "{:.3f}".format,
            "train_mae": "{:,.0f}".format,
            "overfit_gap": "{:.1%}".format,
        },
    ))

    best = table.iloc[0]
    print(f"\nBest run by test MAE: {best['run']} "
          f"(MAE {best['test_mae']:,.0f}, R2 {best['test_r2']:.3f})")


if __name__ == "__main__":
    main()
