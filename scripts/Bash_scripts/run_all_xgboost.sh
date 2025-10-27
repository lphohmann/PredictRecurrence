#!/usr/bin/env bash
set -e

SCRIPT="./scripts/XGBoost/XGBoost_pipeline.py"
COHORTS=("TNBC" "ERpHER2n" "All")
METH_TYPES=("unadjusted") #"adjusted")
DATA_MODES=("clinical" "methylation" "combined")

for cohort in "${COHORTS[@]}"; do
  for mode in "${DATA_MODES[@]}"; do
    for meth in "${METH_TYPES[@]}"; do
      echo ">>> Running: python -u $SCRIPT --cohort_name $cohort --methylation_type $meth --data_mode $mode"
      python -u "$SCRIPT" --cohort_name "$cohort" --methylation_type "$meth" --data_mode "$mode"
    done
  done
done

echo "All XGBoost runs completed."
