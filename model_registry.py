"""
model_registry.py
------------------
Phase 7 of the IDS pipeline: a single place to load trained models, so
predict_live.py and the dashboard don't need to know joblib/torch details.

VERSIONING: every save call in train_ml.py / train_dl.py writes to a fixed
filename (e.g. models/sklearn/xgboost.joblib). For simple projects like this
one, filesystem overwrite-on-retrain is fine; if you need real versioning,
the cheap upgrade is to append a timestamp: f"{name}_{datetime.now():%Y%m%d_%H%M%S}.joblib"
and keep a `latest.joblib` symlink pointing at the newest one. That hook is
marked below with LOAD_LATEST so it's a one-line change later.
"""

import sys
from pathlib import Path
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config_loader import get_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_sklearn_model(name: str):
    """name: one of 'decision_tree', 'random_forest', 'xgboost',
    'logistic_regression', 'svm', 'knn'."""
    cfg = get_config()
    path = resolve_path(cfg["paths"]["sklearn_model_dir"]) / f"{name}.joblib"
    # LOAD_LATEST: swap this for "find newest {name}_*.joblib" if you add timestamps
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}. Run train_ml.py first.")
    model = joblib.load(path)
    logger.info(f"Loaded {name} from {path}")
    return model


def load_best_model():
    """Reads ml_model_comparison.csv (written by train_ml.py) and loads
    whichever model scored highest F1 -- avoids hard-coding 'xgboost' here."""
    import pandas as pd
    cfg = get_config()
    results_path = resolve_path(cfg["paths"]["processed_data_dir"]) / "ml_model_comparison.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"No comparison results at {results_path}. Run train_ml.py first.")
    df = pd.read_csv(results_path)
    best_name = df.sort_values("f1", ascending=False).iloc[0]["model"]
    return load_sklearn_model(best_name), best_name


def load_scaler_and_columns():
    cfg = get_config()
    processed_dir = resolve_path(cfg["paths"]["processed_data_dir"])
    scaler = joblib.load(processed_dir / "scaler.joblib")
    feature_columns = joblib.load(processed_dir / "feature_columns.joblib")
    return scaler, feature_columns


def load_autoencoder():
    """Loads the PyTorch autoencoder + its saved threshold. Requires torch
    to be installed (see train_dl.py docstring)."""
    import torch
    from src.models.train_dl import build_autoencoder

    cfg = get_config()
    model_dir = resolve_path(cfg["paths"]["pytorch_model_dir"])
    meta = joblib.load(model_dir / "autoencoder_meta.joblib")

    model = build_autoencoder(meta["input_dim"], cfg)
    model.load_state_dict(torch.load(model_dir / "autoencoder.pt"))
    model.eval()
    logger.info(f"Loaded autoencoder from {model_dir} (threshold={meta['threshold']:.6f})")
    return model, meta["threshold"]


if __name__ == "__main__":
    # Quick manual smoke test: `python src/models/model_registry.py`
    model, name = load_best_model()
    scaler, cols = load_scaler_and_columns()
    print(f"Best model: {name}, expects {len(cols)} features")
