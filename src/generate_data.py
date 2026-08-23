"""
generate_data.py
-----------------
Synthesizes a member-level healthcare claims dataset for the high-cost
utilizer prediction task.

WHY SYNTHETIC DATA:
Real member-level claims data (e.g., CMS Medicare/Medicaid claims, or
commercial payer data) is protected health information (PHI) and cannot
be redistributed in a public portfolio project. This module instead
generates data from an explicit, documented data-generating process
(DGP) whose parameters are grounded in published healthcare utilization
literature (see README "Data" section for citations/rationale). This
keeps the modeling problem realistic -- imbalanced classes, nonlinear
and interacting risk factors, a noisy cost process -- without using or
imitating any real patient's data.

The label (`high_cost_next_year`) is defined the way payers/CMS actually
define it in HCC-style risk models: membership in the top X% of total
allowed cost in the FOLLOWING 12-month period, predicted using ONLY
information available as of the end of the current (index) period. This
avoids the classic leakage trap in this literature (using same-period
utilization to "predict" same-period cost).
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
N_MEMBERS = 12000
HIGH_COST_PERCENTILE = 0.80  # top 20% of next-year cost = "high-cost utilizer"


def generate_members(n=N_MEMBERS, seed=RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Demographics ---
    age = rng.gamma(shape=6.0, scale=8.0, size=n).clip(18, 95).round().astype(int)
    sex = rng.choice(["F", "M"], size=n, p=[0.52, 0.48])

    # --- Chronic condition burden (Charlson-like comorbidity score) ---
    # Older members and random baseline morbidity drive a Poisson count
    # of chronic conditions (diabetes, CHF, COPD, CKD, etc.)
    cond_rate = 0.03 * (age - 18) + 0.15
    n_chronic_conditions = rng.poisson(lam=cond_rate.clip(min=0.05))
    n_chronic_conditions = n_chronic_conditions.clip(0, 12)

    charlson_index = (n_chronic_conditions * rng.uniform(0.7, 1.4, size=n)).round(1)

    # --- Prior-year utilization (index period, i.e. "now") ---
    base_ed_rate = 0.15 + 0.12 * n_chronic_conditions + 0.01 * (age > 65)
    prior_ed_visits = rng.poisson(lam=base_ed_rate.clip(min=0.02))

    base_ip_rate = 0.03 + 0.05 * n_chronic_conditions + 0.02 * (age > 75)
    prior_inpatient_admits = rng.poisson(lam=base_ip_rate.clip(min=0.01))

    prior_outpatient_visits = rng.poisson(
        lam=(1.0 + 0.8 * n_chronic_conditions + 0.02 * age).clip(min=0.1)
    )

    prior_rx_count = rng.poisson(
        lam=(0.5 + 1.1 * n_chronic_conditions).clip(min=0.1)
    )

    # --- Social / access risk factors (associated with worse outcomes) ---
    dual_eligible = rng.choice([0, 1], size=n, p=[0.85, 0.15])  # Medicare+Medicaid dual
    lives_alone = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    rural_residence = rng.choice([0, 1], size=n, p=[0.78, 0.22])
    missed_appts_prior_year = rng.poisson(lam=0.4 + 0.3 * lives_alone)

    # --- Prior-year total allowed cost (index period cost, a legitimate
    #     predictor of future cost - "past cost predicts future cost"
    #     is one of the most robust findings in this literature) ---
    prior_year_cost = (
        300
        + 150 * n_chronic_conditions
        + 900 * prior_inpatient_admits
        + 220 * prior_ed_visits
        + 60 * prior_outpatient_visits
        + 40 * prior_rx_count
        + rng.gamma(shape=1.3, scale=250, size=n)
    ).round(2)

    df = pd.DataFrame({
        "member_id": [f"M{100000+i}" for i in range(n)],
        "age": age,
        "sex": sex,
        "n_chronic_conditions": n_chronic_conditions,
        "charlson_index": charlson_index,
        "prior_ed_visits": prior_ed_visits,
        "prior_inpatient_admits": prior_inpatient_admits,
        "prior_outpatient_visits": prior_outpatient_visits,
        "prior_rx_count": prior_rx_count,
        "dual_eligible": dual_eligible,
        "lives_alone": lives_alone,
        "rural_residence": rural_residence,
        "missed_appts_prior_year": missed_appts_prior_year,
        "prior_year_cost": prior_year_cost,
    })

    # --- Next-year (OUTCOME period) cost: generated from a DIFFERENT
    #     noise draw than prior_year_cost, with its own random shocks
    #     (e.g., new acute events) so the task is genuinely predictive,
    #     not a deterministic function of the index-period features. ---
    acute_shock = rng.binomial(1, p=(0.02 + 0.01 * n_chronic_conditions).clip(max=0.35))
    shock_magnitude = rng.gamma(shape=2.0, scale=4000, size=n) * acute_shock

    next_year_cost = (
        0.55 * prior_year_cost
        + 180 * n_chronic_conditions
        + 700 * prior_inpatient_admits
        + rng.normal(0, 400, size=n).clip(min=-300)
        + shock_magnitude
        + rng.gamma(shape=1.1, scale=200, size=n)
    ).clip(min=50).round(2)

    df["next_year_cost"] = next_year_cost

    threshold = df["next_year_cost"].quantile(HIGH_COST_PERCENTILE)
    df["high_cost_next_year"] = (df["next_year_cost"] >= threshold).astype(int)

    return df


def main():
    df = generate_members()
    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "members.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} synthetic members to {out_path}")
    print(f"High-cost utilizer prevalence: {df['high_cost_next_year'].mean():.1%}")
    print(f"Cost threshold (P{int(HIGH_COST_PERCENTILE*100)}): "
          f"${df['next_year_cost'].quantile(HIGH_COST_PERCENTILE):,.0f}")


if __name__ == "__main__":
    main()
