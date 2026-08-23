"""
cost_impact_simulation.py
--------------------------
IMPORTANT — read before quoting any number from this script's output.

A predictive model on its own does not reduce healthcare costs. Cost
reduction requires (a) flagging the right members, (b) actually
enrolling them in some intervention (care management, outreach,
medication reconciliation, etc.), and (c) that intervention working.
This project has no real intervention arm and no control group, so it
CANNOT produce a measured "costs went down by X%" number the way a
completed program evaluation (ideally an RCT or a matched pre/post
design with a comparison group) could.

What this script does instead: it runs an explicit, labeled SCENARIO
ANALYSIS. Given
  (1) the model's actual recall/precision on the held-out test set,
  (2) an assumed program ENGAGEMENT rate (fraction of flagged members
      who actually enroll), and
  (3) an assumed program EFFECTIVENESS (per-member cost reduction for
      members who are both correctly flagged AND engaged),
it computes the resulting reduction in total population cost. Rows (2)
and (3) are literature-informed ranges (see README "Cost impact"
section), not measured effects. The output is a small sensitivity
table, not a single headline number, precisely so the assumptions
stay visible.
"""

import json
import sys
print(sys.executable)
from pathlib import Path

import pandas as pd
import joblib

from features import load_dataset, make_splits, FEATURE_COLS

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "members.csv"
MODEL_PATH = ROOT / "models" / "high_cost_utilizer_model.joblib"
OUT_PATH = ROOT / "reports" / "cost_impact_scenarios.json"
SEED = 42

# Literature-informed ranges for care-management style programs targeting
# high-risk members (population health / care-management literature
# generally reports single-digit to high-teens % reductions in total cost
# for ENGAGED high-risk members, with wide variation by program design and
# population). These are illustrative ranges for the scenario grid, not a
# citation-backed point estimate for this synthetic population.
ENGAGEMENT_RATES = [0.4, 0.6, 0.8]
EFFECTIVENESS_RATES = [0.08, 0.12, 0.18]


def main():
    bundle = joblib.load(MODEL_PATH)
    model, threshold = bundle["model"], bundle["threshold"]

    df = load_dataset(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(df, seed=SEED)
    test_idx = X_test.index

    test_df = df.loc[test_idx].copy()
    proba = model.predict_proba(X_test)[:, 1]
    test_df["predicted_high_cost"] = (proba >= threshold).astype(int)

    total_pop_cost = test_df["next_year_cost"].sum()

    # Members correctly flagged: predicted high-cost AND actually high-cost.
    # Only THESE members' future cost is treated as "addressable" by the
    # program in this simulation - false positives get outreach that (by
    # assumption) doesn't change their cost since they weren't actually
    # heading toward high utilization, and false negatives are missed
    # entirely and get no intervention.
    true_positives = test_df[
        (test_df["predicted_high_cost"] == 1) & (test_df["high_cost_next_year"] == 1)
    ]
    addressable_cost = true_positives["next_year_cost"].sum()

    scenarios = []
    for engagement in ENGAGEMENT_RATES:
        for effectiveness in EFFECTIVENESS_RATES:
            cost_saved = addressable_cost * engagement * effectiveness
            pct_of_total = cost_saved / total_pop_cost
            scenarios.append({
                "engagement_rate": engagement,
                "per_member_effectiveness": effectiveness,
                "estimated_dollars_saved": round(float(cost_saved), 2),
                "estimated_pct_of_total_population_cost": round(float(pct_of_total), 4),
            })

    summary = {
        "note": (
            "SIMULATED scenario analysis, not a measured outcome. See module "
            "docstring and README 'Cost impact' section for assumptions and caveats."
        ),
        "test_set_n": int(len(test_df)),
        "test_set_total_next_year_cost": round(float(total_pop_cost), 2),
        "n_true_positive_members": int(len(true_positives)),
        "addressable_cost_true_positives": round(float(addressable_cost), 2),
        "model_recall_on_test": float(
            len(true_positives) / max(test_df["high_cost_next_year"].sum(), 1)
        ),
        "scenarios": scenarios,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    mid = [s for s in scenarios
           if s["engagement_rate"] == 0.6 and s["per_member_effectiveness"] == 0.12][0]
    print(f"Addressable cost (true positives, test set): ${addressable_cost:,.0f} "
          f"of ${total_pop_cost:,.0f} total ({addressable_cost/total_pop_cost:.1%})")
    print(f"Mid scenario (60% engagement, 12% per-member effectiveness): "
          f"{mid['estimated_pct_of_total_population_cost']:.1%} of total population cost "
          f"(${mid['estimated_dollars_saved']:,.0f})")
    print(f"Full sensitivity grid saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
