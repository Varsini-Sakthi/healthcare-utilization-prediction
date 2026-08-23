# Healthcare Utilization Predictive Model

Predicts which health-plan members will become **high-cost utilizers** in the following 12 months, using only information available today (prior utilization, chronic condition burden, demographics, and social/access risk factors). Built to demonstrate an end-to-end, leakage-aware ML pipeline for a classic population-health risk stratification task.

## TL;DR Results (held-out test set, n = 2,400)

| Metric | Value |
|---|---|
| Accuracy | **87.5%** |
| Balanced accuracy | 77.8% |
| Precision | 72.0% |
| Recall | 61.7% |
| ROC-AUC | 0.868 |
| PR-AUC (average precision) | 0.758 |
| Majority-class baseline accuracy | 80.0% |

## Data

Real member-level claims data is protected health information (PHI) and can't be shared in a public repo, so `src/generate_data.py` synthesizes a 12,000-member dataset from an explicit data-generating process: age, chronic condition count, prior-year ED/inpatient/outpatient utilization, prescription count, dual-eligibility, rural residence, living alone, missed appointments, and prior-year cost.

The *outcome* — next-year total cost — is generated with independent random shocks (including a rare "acute event" term), so it is genuinely predictable-but-noisy rather than a deterministic function of the inputs. This keeps the modeling problem realistic without using or approximating any real patient's data. See the module docstring in `src/generate_data.py` for the full rationale and the specific functional form.

`high_cost_next_year` = member is in the **top 20%** of next-year total cost — the standard payer/CMS-style definition for this task, analogous to HCC-based risk stratification.

## Methodology

1. **Leakage guard** — every feature is measurable as of the end of the index period; the outcome-period cost column is never used as a feature (`tests/test_pipeline.py::test_no_leaky_columns_in_features` enforces this).
2. **Split** — 60% train / 20% validation / 20% test, stratified on the outcome. The test set is touched exactly once, for the numbers above.
3. **Model** — `GradientBoostingClassifier` (scikit-learn) in a `Pipeline` with `StandardScaler` + `OneHotEncoder`. Hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, `subsample`) are selected by 5-fold stratified cross-validation, **scored on ROC-AUC, not accuracy** — with a 20%-prevalence outcome, optimizing for accuracy directly biases the search toward the majority class.
4. **Decision threshold** — chosen on the validation set only, by maximizing F1, then frozen and applied unchanged to the test set (threshold = 0.395, not the default 0.5).
5. **Baseline comparison** — a majority-class ("always predict low-cost") classifier already gets 80.0% accuracy, which is why accuracy alone is reported alongside ROC-AUC, PR-AUC, and recall rather than by itself.

## Why Not Just Accuracy?

With ~20% of members labeled high-cost, a model that never flags anyone "high-cost" scores 80% accuracy while being clinically useless. This model's 87.5% accuracy is only ~7.5 points above that trivial baseline, which is why the headline metrics table leads with ROC-AUC (0.868) and reports recall (61.7%) explicitly: the model correctly identifies about 6 in 10 members who will actually become high-cost, at 72% precision among those it flags.

## Feature Importance

Top drivers (Gini importance from the fitted model):

1. `prior_year_cost` — 71.4%
2. `prior_inpatient_admits` — 15.3%
3. `n_chronic_conditions` — 12.4%
4. `charlson_index` — 0.5%
5. `prior_rx_count` — 0.1%

Prior-year cost dominates, consistent with the literature finding that past utilization is the single strongest predictor of future utilization. This also means the model leans heavily on one feature — `reports/feature_importance.png` and `reports/pr_curve.png` are worth inspecting before treating this as a finished clinical tool.

## Cost Impact — and Why "Reduced Costs by 12%" Needs a Caveat

A prediction model does not, by itself, reduce anyone's costs — it only identifies who to intervene on. Actually measuring a cost reduction requires a real intervention arm and a comparison group (ideally a randomized controlled trial or a matched pre/post design), which this project doesn't have.

`src/cost_impact_simulation.py` instead runs an explicit **scenario analysis**: it takes the test-set members the model correctly flags (true positives — 24.6% of total test-set cost is concentrated in this group) and multiplies by an assumed program *engagement rate* and an assumed per-member *effectiveness*, both literature-informed ranges rather than measured values (see `reports/cost_impact_scenarios.json` for the full 3×3 grid).

At a middle-of-the-road assumption (60% engagement, 12% per-member cost reduction for engaged, correctly-flagged members), the estimated population-level impact is **~1.8% of total cost — not 12%**. A 12% *total population* cost reduction would require much higher engagement/effectiveness than typical published ranges, a much higher-recall model, or both.

**Practical takeaway**: "12% cost reduction" is a defensible number for *effectiveness among successfully engaged, correctly-flagged high-risk members* in some published care-management programs. It is not a defensible number for *total population* cost reduction from the model alone, and shouldn't be presented as one without a real program evaluation behind it.

## Project Structure

```
healthcare-utilization-prediction/
├── src/
│   ├── generate_data.py          # synthetic data generating process
│   ├── features.py               # feature list, leakage guard, splits
│   ├── train_model.py            # CV grid search, threshold selection, eval
│   ├── make_plots.py             # ROC/PR/confusion matrix/importance plots
│   └── cost_impact_simulation.py # labeled scenario analysis (not a measured result)
├── tests/
│   └── test_pipeline.py          # leakage, split, and sanity checks
├── data/members.csv              # generated when you run the pipeline
├── models/high_cost_utilizer_model.joblib
├── reports/                      # metrics.json, plots, cost_impact_scenarios.json
├── requirements.txt
└── run_pipeline.sh
```

## Running It Locally (macOS)

Requires Python 3.10+ (check with `python3 --version`; install via [python.org](https://www.python.org/downloads/) or `brew install python` if needed).

```bash
# 1. Clone and enter the project
git clone https://github.com/Varsini-Sakthi/healthcare-utilization-prediction.git
cd healthcare-utilization-prediction

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline (data -> train -> plots -> cost scenarios -> tests)
bash run_pipeline.sh
```

Or run steps individually:

```bash
python3 src/generate_data.py
cd src && python3 train_model.py && python3 make_plots.py && python3 cost_impact_simulation.py && cd ..
python3 -m pytest tests/ -q
```

Everything runs on CPU in well under a minute — no GPU or external data download required.

## Limitations

- **Synthetic data** — results describe how the *pipeline* performs on a documented synthetic data-generating process, not how it would perform on real claims data. Real payer data has messier missingness, coding artifacts, and distribution shift over time that this doesn't capture.
- **Recall is 62%** — over a third of true high-cost members are missed at the chosen threshold. A deployed version should tune the threshold to a real payer's cost asymmetry, since missing a high-cost member is usually costlier than an unnecessary outreach call.
- **Feature importance is dominated by one variable** (`prior_year_cost` at 71%) — worth checking model behavior with that feature removed before trusting the rest of the importance ranking.
- **Cost impact is a scenario analysis, not a measured effect** — see the "Cost Impact" section above.

## Author

**Varsini Sakthivadivel Ramasamy**
M.S. Bioinformatics, Johns Hopkins University
[LinkedIn](https://linkedin.com/in/varsini-sakthi) · [GitHub](https://github.com/Varsini-Sakthi)
