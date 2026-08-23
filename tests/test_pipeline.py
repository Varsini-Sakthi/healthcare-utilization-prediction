"""
Basic correctness / leakage tests for the pipeline.
Run with: pytest -q  (from the project root, with src/ on PYTHONPATH)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import load_dataset, make_splits, FEATURE_COLS, TARGET, LEAKY_COLS  # noqa: E402
from generate_data import generate_members, HIGH_COST_PERCENTILE  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return generate_members(n=3000, seed=1)


def test_no_leaky_columns_in_features():
    """The outcome-period cost column must never appear in the feature set."""
    for col in LEAKY_COLS:
        assert col not in FEATURE_COLS


def test_target_is_binary(df):
    assert set(df[TARGET].unique()) <= {0, 1}


def test_prevalence_matches_configured_percentile(df):
    prevalence = df[TARGET].mean()
    expected = 1 - HIGH_COST_PERCENTILE
    assert abs(prevalence - expected) < 0.03, (
        f"Prevalence {prevalence:.3f} too far from expected ~{expected:.3f}"
    )


def test_splits_are_disjoint_and_stratified(df):
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(df, seed=1)

    train_idx, val_idx, test_idx = set(X_train.index), set(X_val.index), set(X_test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)

    # Stratification: positive rate shouldn't drift far from the full-data rate.
    full_rate = df[TARGET].mean()
    for y in (y_train, y_val, y_test):
        assert abs(y.mean() - full_rate) < 0.03


def test_no_missing_values_in_features(df):
    assert df[FEATURE_COLS].isna().sum().sum() == 0


def test_no_negative_costs(df):
    assert (df["prior_year_cost"] >= 0).all()
    assert (df["next_year_cost"] >= 0).all()
