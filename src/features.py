"""
features.py
------------
Feature engineering and the train/test split used by both the training
and evaluation scripts. Kept in one place so train and inference always
apply identical transformations.

Leakage guard: every feature here is measurable as of the END of the
index period. `next_year_cost` and `high_cost_next_year` are the
outcome and must never be used as inputs.
"""

from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "high_cost_next_year"
ID_COL = "member_id"
LEAKY_COLS = ["next_year_cost"]  # outcome-period info; must never be a feature

CATEGORICAL_FEATURES = ["sex"]
NUMERIC_FEATURES = [
    "age",
    "n_chronic_conditions",
    "charlson_index",
    "prior_ed_visits",
    "prior_inpatient_admits",
    "prior_outpatient_visits",
    "prior_rx_count",
    "dual_eligible",
    "lives_alone",
    "rural_residence",
    "missed_appts_prior_year",
    "prior_year_cost",
]

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(FEATURE_COLS + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing expected columns: {missing}")
    return df


def make_splits(df: pd.DataFrame, test_size=0.2, val_size=0.2, seed=42):
    """
    Three-way split: train / validation (for threshold + hyperparameter
    selection) / test (touched exactly once, for final reporting).
    Stratified on the target because the outcome is imbalanced (~20%
    positive).
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET].copy()

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_fraction_of_trainval = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction_of_trainval,
        stratify=y_trainval,
        random_state=seed,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
