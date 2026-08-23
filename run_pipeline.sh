#!/usr/bin/env bash
# Runs the full pipeline end-to-end: generate data -> train -> plots -> cost scenarios -> tests.
# Usage: bash run_pipeline.sh
set -euo pipefail

cd "$(dirname "$0")"

echo "== 1/5  Generating synthetic data =="
python3 src/generate_data.py

echo "== 2/5  Training model (grid search + CV) =="
cd src && python3 train_model.py && cd ..

echo "== 3/5  Rendering evaluation plots =="
cd src && python3 make_plots.py && cd ..

echo "== 4/5  Running cost-impact scenario analysis =="
cd src && python3 cost_impact_simulation.py && cd ..

echo "== 5/5  Running test suite =="
python3 -m pytest tests/ -q

echo ""
echo "Done. See reports/ for metrics.json, plots, and cost_impact_scenarios.json."
