"""
preprocessing.py
-----------------
Phase 3 of the IDS pipeline: turn raw CICFlowMeter CSV output into clean,
scaled, balanced train/val/test splits ready for model training.

Run directly: `python src/data/preprocessing.py`
Or import `run_preprocessing()` from train_ml.py / train_dl.py.

Steps, in order (order matters — cleaning must happen before scaling,
scaling before SMOTE, so scaled distances are used for synthetic sampling):
  1. Load raw CSV
  2. Drop identifier columns (Flow ID, IPs, Timestamp) — these would leak
     the data source into the model and don't generalize to new traffic
  3. Drop columns that are constant (zero variance) - they add no signal
  4. Replace inf -> NaN, then drop/impute NaN rows
  5. Drop exact duplicate rows (CICFlowMeter sometimes emits duplicates for
     retransmitted flows)
  6. Encode the Label column: BENIGN -> 0, everything else -> 1
     (binary IDS: normal vs malicious. Multiclass is easy to re-enable —
     see the `binary` flag below)
  7. Split into train/val/test BEFORE scaling/SMOTE, to prevent data leakage
     (fitting the scaler or SMOTE on data that includes test rows would let
     information leak from test into training)
  8. Fit scaler on train only, transform all three splits
  9. Apply SMOTE to the training split only (never oversample val/test —
     that would evaluate the model on synthetic, non-real data)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config_loader import get_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(cfg: dict) -> pd.DataFrame:
    path = resolve_path(f"{cfg['paths']['raw_data_dir']}/{cfg['data']['raw_filename']}")
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # Some CICFlowMeter exports have leading/trailing whitespace in headers
    df.columns = df.columns.str.strip()

    # 1. Drop identifier / leakage columns
    drop_cols = [c for c in cfg["data"]["drop_columns"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    logger.info(f"Dropped identifier columns: {drop_cols}")

    label_col = cfg["data"]["label_column"]
    feature_cols = [c for c in df.columns if c != label_col]

    # 2. Replace inf with NaN so we can handle both the same way
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # 3. Drop rows with any NaN in the feature columns
    #    (for a dataset this size, dropping is simpler and safer than imputing;
    #    imputing flow-statistics with mean/median can invent implausible flows)
    n_before = len(df)
    df = df.dropna(subset=feature_cols)
    logger.info(f"Dropped {n_before - len(df)} rows with NaN/inf values")

    # 4. Drop exact duplicate rows
    n_before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Dropped {n_before - len(df)} duplicate rows")

    # 5. Drop zero-variance columns (constant across all rows -> no signal)
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    zero_var_cols = [c for c in numeric_cols if df[c].nunique() <= 1]
    if zero_var_cols:
        df = df.drop(columns=zero_var_cols)
        logger.info(f"Dropped zero-variance columns: {zero_var_cols}")

    return df.reset_index(drop=True)


def encode_labels(df: pd.DataFrame, cfg: dict, binary: bool = True):
    """binary=True: BENIGN -> 0, any attack -> 1 (standard IDS framing).
    binary=False: full multiclass LabelEncoder over all attack types —
    useful for the EDA / "which attack is it" analysis in Phase 4-5."""
    label_col = cfg["data"]["label_column"]
    benign_label = cfg["data"]["benign_label"]

    if binary:
        y = (df[label_col] != benign_label).astype(int)
        class_names = {0: benign_label, 1: "ATTACK"}
        encoder = None
    else:
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(df[label_col]), index=df.index)
        class_names = dict(enumerate(encoder.classes_))

    X = df.drop(columns=[label_col])
    return X, y, class_names, encoder


def split_data(X: pd.DataFrame, y: pd.Series, cfg: dict):
    seed = cfg["project"]["random_seed"]
    test_size = cfg["data"]["test_size"]
    val_size = cfg["data"]["val_size"]

    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    # Then split val out of what remains (val_size is expressed as a fraction
    # of the ORIGINAL dataset, so we rescale it relative to X_temp)
    val_fraction_of_temp = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction_of_temp, random_state=seed, stratify=y_temp
    )

    logger.info(f"Split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test, cfg: dict):
    scaler_type = cfg["data"]["scaler"]
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()

    # Fit ONLY on training data -> prevents test/val statistics leaking into
    # the transform (a classic and easy-to-miss source of data leakage)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"Scaled features using {scaler_type} scaler")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def balance_training_data(X_train, y_train, cfg: dict):
    if not cfg["data"]["apply_smote"]:
        return X_train, y_train

    seed = cfg["project"]["random_seed"]
    logger.info(f"Class distribution before SMOTE: {pd.Series(y_train).value_counts().to_dict()}")

    smote = SMOTE(random_state=seed)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    logger.info(f"Class distribution after SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
    return X_resampled, y_resampled


def run_preprocessing(binary: bool = True):
    """Full Phase 3 pipeline. Returns everything downstream training code needs,
    and also persists processed arrays + the fitted scaler to disk so
    train_ml.py / train_dl.py don't need to redo this work every run."""
    cfg = get_config()

    df = load_raw_data(cfg)
    df = clean_data(df, cfg)
    X, y, class_names, label_encoder = encode_labels(df, cfg, binary=binary)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, cfg)
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test, cfg)
    X_train_bal, y_train_bal = balance_training_data(X_train_s, y_train, cfg)

    # Persist to disk
    processed_dir = resolve_path(cfg["paths"]["processed_data_dir"])
    np.savez(
        processed_dir / "processed_data.npz",
        X_train=X_train_bal, y_train=y_train_bal,
        X_val=X_val_s, y_val=y_val,
        X_test=X_test_s, y_test=y_test,
    )
    joblib.dump(scaler, processed_dir / "scaler.joblib")
    joblib.dump(list(X.columns), processed_dir / "feature_columns.joblib")
    joblib.dump(class_names, processed_dir / "class_names.joblib")
    logger.info(f"Processed data + scaler saved to {processed_dir}")

    return {
        "X_train": X_train_bal, "y_train": y_train_bal,
        "X_val": X_val_s, "y_val": y_val,
        "X_test": X_test_s, "y_test": y_test,
        "feature_columns": list(X.columns),
        "class_names": class_names,
        "scaler": scaler,
    }


if __name__ == "__main__":
    run_preprocessing()
