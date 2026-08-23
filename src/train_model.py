"""
train_model.py
---------------
Trains a gradient-boosted classifier to predict `high_cost_next_year`.

Design choices (documented deliberately, since this is the part a PhD
committee / hiring manager would probe):

1. Metric choice: the outcome is imbalanced (~20% positive). Accuracy
   alone is a weak metric here, a trivial "always predict low-cost"
   classifier already scores ~80% accuracy. So model SELECTION is done
   on ROC-AUC / average precision via cross-validation, and the final
   report includes accuracy, balanced accuracy, precision, recall, F1,
   ROC-AUC, and PR-AUC together, not accuracy in isolation.

2. Train/val/test split: the test set is touched exactly once, at the
   very end, for the numbers reported in README.md. Hyperparameters and
   the classification threshold are chosen on train/validation only.

3. Threshold selection: the default 0.5 threshold is not assumed to be
   optimal for an imbalanced, cost-sensitive problem. The decision
   threshold is chosen on the validation set to maximize F1 (a
   documented, inspectable choice, swap in a payer's real cost
   asymmetry if this were deployed).
"""

from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import (
    load_dataset, make_splits, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "members.csv"
MODEL_PATH = ROOT / "models" / "high_cost_utilizer_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
PLOTS_DIR = ROOT / "reports"
SEED = 42


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"),
             CATEGORICAL_FEATURES),
        ]
    )
    clf = GradientBoostingClassifier(random_state=SEED)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def select_threshold(y_true, y_proba) -> float:
    """Pick the probability threshold on the validation set that
    maximizes F1. Returned threshold is later frozen and applied,
    unchanged, to the held-out test set."""
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    best_t = thresholds[int(np.argmax(f1s))]
    return float(best_t)


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


def main():
    MODEL_PATH.parent.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    df = load_dataset(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(df, seed=SEED)

    pipe = build_pipeline()

    param_grid = {
        "model__n_estimators": [150, 300],
        "model__max_depth": [2, 3],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True
    )
    print("Running 5-fold CV grid search (scoring = ROC-AUC)...")
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    # Threshold chosen on validation set only.
    val_proba = best_model.predict_proba(X_val)[:, 1]
    threshold = select_threshold(y_val.values, val_proba)
    val_pred = (val_proba >= threshold).astype(int)
    val_metrics = evaluate(y_val.values, val_pred, val_proba)
    print(f"Selected decision threshold (max F1 on val): {threshold:.3f}")
    print("Validation metrics:", {k: round(v, 4) if isinstance(v, float) else v
                                   for k, v in val_metrics.items()})

    # Refit best hyperparameters on train+val, then evaluate ONCE on test.
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    final_model = build_pipeline().set_params(**search.best_params_)
    final_model.fit(X_trainval, y_trainval)

    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    test_metrics = evaluate(y_test.values, test_pred, test_proba)
    print("TEST metrics (touched once):",
          {k: round(v, 4) if isinstance(v, float) else v for k, v in test_metrics.items()})

    # Baseline for comparison: majority-class classifier.
    majority_pred = np.zeros_like(y_test.values)
    baseline_metrics = evaluate(
        y_test.values, majority_pred,
        np.zeros_like(y_test.values, dtype=float) + 1e-6,
    )

    cm = confusion_matrix(y_test.values, test_pred).tolist()

    # Feature importances (Gini-based, from the fitted GBM).
    feature_names = (
        NUMERIC_FEATURES
        + list(final_model.named_steps["preprocess"]
               .named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    )
    importances = final_model.named_steps["model"].feature_importances_
    feat_importance = sorted(
        zip(feature_names, importances.tolist()), key=lambda x: -x[1]
    )

    fpr, tpr, _ = roc_curve(y_test.values, test_proba)
    prec, rec, _ = precision_recall_curve(y_test.values, test_proba)

    results = {
        "seed": SEED,
        "n_members_total": int(len(df)),
        "high_cost_prevalence_overall": float(df["high_cost_next_year"].mean()),
        "best_cv_roc_auc": float(search.best_score_),
        "best_params": search.best_params_,
        "decision_threshold": threshold,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "majority_class_baseline_test_metrics": baseline_metrics,
        "confusion_matrix_test": {
            "labels": ["pred_low_cost", "pred_high_cost"],
            "matrix": cm,
        },
        "feature_importance": feat_importance,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": prec.tolist(), "recall": rec.tolist()},
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    joblib.dump({"model": final_model, "threshold": threshold,
                 "feature_cols": NUMERIC_FEATURES + CATEGORICAL_FEATURES}, MODEL_PATH)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
