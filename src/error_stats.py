"""How far off is the final model on held-out cars? Saves what estimate.py needs.

estimate.py turns one prediction into a likely range by scaling it with the ratios
saved here (actual price divided by predicted price on the test set). Models the
Pipeline sees often get their own ratios, because rare and unseen models are
harder to price and need a wider range.

Run from the repo root with the venv active (also runs at the end of train_final):
    python -m src.error_stats
"""
import json
from pathlib import Path

import numpy as np

from src.estimate import MODEL_DIR, STATS_PATH, load_model, model_key_for, model_keys
from src.preprocess import load_config

LEVEL = 0.8   # the range covers the middle 80 percent of test errors


def _ratio_range(ratios):
    tail = (1 - LEVEL) / 2
    low, high = np.quantile(ratios, [tail, 1 - tail])
    return {"n": int(len(ratios)), "low_ratio": round(float(low), 4),
            "high_ratio": round(float(high), 4)}


def compute_error_stats(model, X_test, y_test):
    """Error ratios on the test set, split by how well the model knows the car."""
    frequent, _ = model_keys(model)
    predicted = np.asarray(model.predict(X_test), dtype=float)
    actual = np.asarray(y_test, dtype=float)
    ratios = actual / predicted

    keys = X_test.apply(lambda row: model_key_for(row.to_dict()), axis=1)
    common = keys.isin(frequent).to_numpy()

    return {
        "level": LEVEL,
        "n_test": int(len(actual)),
        "median_abs_pct_error": round(float(np.median(np.abs(ratios - 1))), 4),
        "common": _ratio_range(ratios[common]),
        "other": _ratio_range(ratios[~common]),
    }


def save_error_stats(model, X_test, y_test, path=STATS_PATH):
    stats = compute_error_stats(model, X_test, y_test)
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved error stats to {path}")
    for group in ["common", "other"]:
        g = stats[group]
        print(f"  {group:<7} n={g['n']:>6}  actual/predicted: "
              f"{g['low_ratio']:.2f} to {g['high_ratio']:.2f}")
    print(f"  median absolute percent error: {stats['median_abs_pct_error']:.1%}")
    return stats


def main():
    from src.train import load_split, load_train_config

    data_cfg = load_config()
    train_cfg = load_train_config()
    train_cfg["SAMPLE_SIZE"] = None  # same split as train_final
    _, X_test, _, y_test = load_split(data_cfg, train_cfg)
    model = load_model(MODEL_DIR)
    save_error_stats(model, X_test, y_test)


if __name__ == "__main__":
    main()
