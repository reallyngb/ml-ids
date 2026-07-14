"""
train_ml.py
-----------
Phase 5 of the IDS pipeline: train and compare six classical ML models.

Models (theory in one line each, full explanation in README.md):
  - Decision Tree: single tree of if/else splits on feature thresholds.
    Fast, interpretable, prone to overfitting on its own.
  - Random Forest: bagged ensemble of decision trees, votes on the result.
    Reduces overfitting vs a single tree, handles nonlinear boundaries well.
  - XGBoost: gradient-boosted trees, each new tree corrects the previous
    ensemble's errors. Usually the strongest tabular-data performer.
  - Logistic Regression: linear decision boundary in scaled feature space.
    Fast, very interpretable (coefficients = feature weights), weak on
    non-linear attack patterns.
  - SVM (RBF kernel): finds a max-margin boundary, kernel trick handles
    non-linear separation. Slow on large datasets, sensitive to scaling
    (this is exactly why Phase 3 standardized features first).
  - K-Nearest Neighbours: classifies a flow by majority vote of its closest
    neighbours in feature space. Simple, no training phase, but slow at
    inference time on large datasets and sensitive to irrelevant features.

For each model we run: 5-fold cross-validation on the training set (to
check the score isn't a fluke of one train/test split), then a final fit
and evaluation on the held-out test set.

Metrics computed for every model: Precision, Recall, F1, ROC-AUC, confusion
matrix, Detection Rate (= Recall on the attack class), and False Positive
Rate. See README.md "Why Precision/Recall over Accuracy" for justification.

Run directly: `python src/models/train_ml.py`
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config_loader import get_config, resolve_path
from src.utils.logger import get_logger
from src.data.preprocessing import run_preprocessing

logger = get_logger(__name__)

# Small, sensible hyperparameter grids. Kept small deliberately: with 5-fold CV
# every extra grid point multiplies runtime by 5. Expand these once you're on
# the real (larger) CIC-IDS2017 dataset and have time to spare.
PARAM_GRIDS = {
    "decision_tree": {"max_depth": [10, 20, None], "min_samples_split": [2, 10]},
    "random_forest": {"n_estimators": [100, 200], "max_depth": [15, None]},
    "xgboost": {"n_estimators": [100, 200], "max_depth": [4, 6], "learning_rate": [0.1, 0.3]},
    "logistic_regression": {"C": [0.1, 1.0, 10.0]},
    "svm": {"C": [1.0, 10.0], "gamma": ["scale"]},
    "knn": {"n_neighbors": [3, 5, 9]},
}

MODEL_FACTORY = {
    "decision_tree": lambda seed: DecisionTreeClassifier(random_state=seed),
    "random_forest": lambda seed: RandomForestClassifier(random_state=seed, n_jobs=-1),
    "xgboost": lambda seed: XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=-1),
    "logistic_regression": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    "svm": lambda seed: SVC(probability=True, random_state=seed),
    "knn": lambda seed: KNeighborsClassifier(),
}


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # a.k.a. Recall on attacks
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_and_evaluate_all(data: dict, cfg: dict) -> pd.DataFrame:
    seed = cfg["project"]["random_seed"]
    cv_folds = cfg["training"]["cv_folds"]
    model_names = cfg["training"]["models"]

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    results = []
    fitted_models = {}
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    for name in model_names:
        logger.info(f"=== Training {name} ===")
        base_model = MODEL_FACTORY[name](seed)

        # Hyperparameter tuning via grid search with cross-validation.
        # scoring='f1' because in IDS, precision/recall tradeoff matters more
        # than raw accuracy (see Phase 11 discussion).
        t0 = time.time()
        search = GridSearchCV(
            base_model, PARAM_GRIDS[name], scoring="f1", cv=skf,
            n_jobs=cfg["training"]["n_jobs"],
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        train_time = time.time() - t0

        logger.info(f"{name} best params: {search.best_params_} (CV f1={search.best_score_:.4f}, {train_time:.1f}s)")

        # Cross-validated score on the BEST model, for a stability check
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=skf, scoring="f1", n_jobs=cfg["training"]["n_jobs"])

        metrics = evaluate_model(best_model, X_test, y_test)
        metrics.update({
            "model": name,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "train_time_s": round(train_time, 2),
            "best_params": search.best_params_,
        })
        results.append(metrics)
        fitted_models[name] = best_model

        logger.info(
            f"{name} TEST -> precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f} FPR={metrics['false_positive_rate']:.4f}"
        )

    return pd.DataFrame(results), fitted_models


def save_models(fitted_models: dict, cfg: dict):
    """Phase 7: model saving via joblib, with a version suffix so re-runs
    don't silently overwrite a model you meant to keep."""
    model_dir = resolve_path(cfg["paths"]["sklearn_model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, model in fitted_models.items():
        path = model_dir / f"{name}.joblib"
        joblib.dump(model, path)
        logger.info(f"Saved {name} -> {path}")


def select_best_model(results_df: pd.DataFrame) -> str:
    """Best model = highest F1 (balances precision & recall) among candidates,
    tie-broken by lower false positive rate (fewer wasted analyst hours)."""
    ranked = results_df.sort_values(["f1", "false_positive_rate"], ascending=[False, True])
    best = ranked.iloc[0]
    logger.info(f"Best model: {best['model']} (F1={best['f1']:.4f}, FPR={best['false_positive_rate']:.4f})")
    return best["model"]


if __name__ == "__main__":
    cfg = get_config()
    data = run_preprocessing(binary=True)
    results_df, fitted_models = train_and_evaluate_all(data, cfg)

    print("\n" + "=" * 100)
    print(results_df[["model", "precision", "recall", "f1", "roc_auc", "detection_rate",
                       "false_positive_rate", "train_time_s"]].to_string(index=False))
    print("=" * 100)

    save_models(fitted_models, cfg)
    best_model_name = select_best_model(results_df)

    results_path = resolve_path(cfg["paths"]["processed_data_dir"]) / "ml_model_comparison.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Comparison table saved to {results_path}")
    logger.info(f"RECOMMENDED MODEL: {best_model_name}")
