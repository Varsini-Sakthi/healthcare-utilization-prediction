"""
make_plots.py
-------------
Renders the evaluation figures (ROC curve, PR curve, confusion matrix,
feature importance) from reports/metrics.json into reports/*.png.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "reports" / "metrics.json"
REPORTS_DIR = ROOT / "reports"


def main():
    with open(METRICS_PATH) as f:
        r = json.load(f)

    # --- ROC curve ---
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(r["roc_curve"]["fpr"], r["roc_curve"]["tpr"],
            label=f"GBM (AUC = {r['test_metrics']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "-", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    # --- Precision-Recall curve ---
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(r["pr_curve"]["recall"], r["pr_curve"]["precision"],
            label=f"GBM (AP = {r['test_metrics']['pr_auc']:.3f})")
    ax.axhline(r["high_cost_prevalence_overall"], linestyle="-", color="gray",
               label=f"Chance ({r['high_cost_prevalence_overall']:.0%} prevalence)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Test Set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "pr_curve.png", dpi=150)
    plt.close(fig)

    # --- Confusion matrix ---
    cm = np.array(r["confusion_matrix_test"]["matrix"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="black", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Low-cost", "High-cost"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Low-cost", "High-cost"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — Test Set\n(threshold = {r['decision_threshold']:.2f})")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- Feature importance ---
    names, vals = zip(*r["feature_importance"])
    fig, ax = plt.subplots(figsize=(6, 4))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, color="#2C6E9E")
    ax.set_yticks(y_pos); ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Gini importance")
    ax.set_title("Feature Importance (GBM)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    print("Saved plots to", REPORTS_DIR)


if __name__ == "__main__":
    main()
