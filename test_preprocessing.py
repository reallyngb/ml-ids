"""
test_preprocessing.py
----------------------
Unit tests for src/data/preprocessing.py.

Run: `pytest tests/` from the project root (or `pytest tests/test_preprocessing.py -v`)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config_loader import get_config
from src.data.preprocessing import clean_data, encode_labels, split_data, scale_features


@pytest.fixture
def cfg():
    return get_config()


@pytest.fixture
def dirty_df(cfg):
    """A synthetic frame with the exact problems clean_data() must fix:
    an identifier column, an inf value, a NaN value, and a duplicate row.
    Enough clean rows remain afterward that Feature A/B still vary, so
    they survive the zero-variance drop and can be asserted on."""
    label_col = cfg["data"]["label_column"]
    df = pd.DataFrame({
        "Flow ID": ["a", "b", "c", "c", "d", "e"],
        "Feature A": [1.0, 2.0, np.inf, np.inf, 5.0, 6.0],
        "Feature B": [10.0, np.nan, 30.0, 30.0, 50.0, 60.0],
        label_col: ["BENIGN", "BENIGN", "DoS", "DoS", "BENIGN", "PortScan"],
    })
    return df


def test_clean_data_drops_identifier_columns(dirty_df, cfg):
    cleaned = clean_data(dirty_df.copy(), cfg)
    assert "Flow ID" not in cleaned.columns


def test_clean_data_removes_inf_and_nan_rows(dirty_df, cfg):
    cleaned = clean_data(dirty_df.copy(), cfg)
    # Row with inf and row with NaN must both be gone
    assert cleaned["Feature A"].isin([np.inf, -np.inf]).sum() == 0
    assert cleaned.isna().sum().sum() == 0


def test_clean_data_removes_duplicates(dirty_df, cfg):
    cleaned = clean_data(dirty_df.copy(), cfg)
    assert not cleaned.duplicated().any()


def test_encode_labels_binary(cfg):
    label_col = cfg["data"]["label_column"]
    df = pd.DataFrame({
        "Feature A": [1, 2, 3, 4],
        label_col: ["BENIGN", "BENIGN", "DoS", "PortScan"],
    })
    X, y, class_names, encoder = encode_labels(df, cfg, binary=True)
    assert label_col not in X.columns
    assert list(y) == [0, 0, 1, 1]
    assert encoder is None


def test_encode_labels_multiclass(cfg):
    label_col = cfg["data"]["label_column"]
    df = pd.DataFrame({
        "Feature A": [1, 2, 3, 4],
        label_col: ["BENIGN", "BENIGN", "DoS", "PortScan"],
    })
    X, y, class_names, encoder = encode_labels(df, cfg, binary=False)
    assert encoder is not None
    assert y.nunique() == 3


def test_split_data_preserves_row_count(cfg):
    n = 500
    X = pd.DataFrame(np.random.rand(n, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, n))
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, cfg)
    assert len(X_train) + len(X_val) + len(X_test) == n
    assert len(y_train) + len(y_val) + len(y_test) == n


def test_split_data_is_stratified(cfg):
    """With stratify=y, each split's class ratio should roughly match the
    original — this guards against a regression that drops stratify=y."""
    n = 1000
    X = pd.DataFrame(np.random.rand(n, 3), columns=["f0", "f1", "f2"])
    y = pd.Series([0] * 800 + [1] * 200)  # 80/20 imbalance, like real IDS data
    _, _, _, y_train, y_val, y_test = split_data(X, y, cfg)

    for split in (y_train, y_val, y_test):
        ratio = split.mean()  # fraction of class 1
        assert 0.15 <= ratio <= 0.25, f"split ratio {ratio} drifted too far from 0.20"


def test_scale_features_fits_only_on_train(cfg):
    """The scaler's learned mean/scale must come from X_train, not be
    influenced by X_val/X_test -- this is the data-leakage check."""
    X_train = pd.DataFrame({"f0": [0.0, 0.0, 0.0, 0.0]})
    X_val = pd.DataFrame({"f0": [100.0]})
    X_test = pd.DataFrame({"f0": [100.0]})

    _, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test, cfg)
    # A constant-zero train column should fit mean=0; val/test transform
    # should reflect that fitted mean, not their own values
    assert scaler.mean_[0] == 0.0
