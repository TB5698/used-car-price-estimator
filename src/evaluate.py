"""Load the best MLflow model and check it on the held-out test set.

Run from the repo root with the venv active:
    python -m src.evaluate
"""
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.preprocess import load_config
from src.train import evaluate, load_split, load_train_config


def best_run_id(experiment_name):
    """Run id of the finished run with the lowest test MAE."""
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.mae ASC"],
        max_results=1,
    )
    if runs.empty:
        raise SystemExit(f"No finished runs found for experiment '{experiment_name}'.")
    return runs.iloc[0]["run_id"], runs.iloc[0]["tags.mlflow.runName"]


def main():
    data_cfg = load_config()
    train_cfg = load_train_config()

    run_id, run_name = best_run_id(train_cfg["EXPERIMENT_NAME"])
    print(f"Best run: {run_name}")
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    # same seed and split as train.py, so this is the same held-out test set
    _, X_test, _, y_test = load_split(data_cfg, train_cfg)
    preds = model.predict(X_test)
    metrics = evaluate(y_test, preds)

    logged = mlflow.get_run(run_id).data.metrics
    print("\nMetric   loaded model   logged in MLflow")
    for name in ["mae", "rmse", "r2"]:
        print(f"{name:<8} {metrics[name]:>12,.3f}   {logged[name]:>14,.3f}")
    same = all(np.isclose(metrics[m], logged[m], rtol=1e-3) for m in ["mae", "rmse", "r2"])
    print("Matches the logged metrics." if same else "WARNING: metrics differ from the logged run.")

    # a few real test listings next to what the model predicts
    sample = X_test.head(5)[["year", "manufacturer", "model", "odometer"]].copy()
    sample["actual"] = y_test.head(5).to_numpy()
    sample["predicted"] = preds[:5].round(0)
    print("\nFirst 5 test listings (dollars):")
    print(sample.to_string(index=False))

    # the app will send messy input: an unseen model and missing optional fields
    car = pd.DataFrame([{
        "year": 2016, "manufacturer": "toyota", "model": "made up model name",
        "odometer": 90000, "condition": None, "fuel": None, "title_status": None,
        "transmission": None, "drive": None, "type": None,
    }])
    print(f"\nUnseen model with missing fields predicts: {model.predict(car)[0]:,.0f} dollars")


if __name__ == "__main__":
    main()
