"""Train the final model on the full data and save it for the app.

Run from the repo root with the venv active:
    python -m src.train_final
"""
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.error_stats import save_error_stats
from src.preprocess import load_config
from src.train import load_final_spec, load_split, load_train_config, run_from_spec

MODEL_DIR = Path("models/final_model")


def main():
    data_cfg = load_config()
    train_cfg = load_train_config()
    train_cfg["SAMPLE_SIZE"] = None  # use every cleaned row
    data = load_split(data_cfg, train_cfg)

    # separate experiment: the test split here is bigger than in the sample runs,
    # so these numbers should not be ranked against them in compare_runs
    mlflow.set_experiment(train_cfg["EXPERIMENT_NAME"] + "-final")

    # same settings as the best sample run, named in config.yaml (final_model section)
    _, model = run_from_spec(load_final_spec(), data, train_cfg)

    # plain folder for the Docker image (models/ is not tracked by Git)
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.parent.mkdir(exist_ok=True)
    mlflow.sklearn.save_model(
        model, str(MODEL_DIR),
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )
    print(f"Saved model to {MODEL_DIR}")

    # how far off the model is on the held-out cars, used for the price range in the app
    save_error_stats(model, data[1], data[3])

    # reload from disk and check it gives the same answer as the model in memory
    car = pd.DataFrame([{
        "year": 2016, "manufacturer": "toyota", "model": "camry",
        "odometer": 90000, "condition": None, "fuel": "gas",
        "title_status": "clean", "transmission": "automatic",
        "drive": None, "type": None,
    }])
    reloaded = mlflow.sklearn.load_model(str(MODEL_DIR))
    price = reloaded.predict(car)[0]
    assert np.isclose(price, model.predict(car)[0]), "reloaded model differs"
    print(f"Reloaded model OK. 2016 Toyota Camry, 90,000 miles: {price:,.0f} dollars")


if __name__ == "__main__":
    main()
