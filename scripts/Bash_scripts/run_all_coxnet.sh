#!/usr/bin/env bash
set -e

SCRIPT="./scripts/CoxNet/CoxNet_pipeline.py"
COHORTS=("TNBC" "ERpHER2n" "All")
METH_TYPES=("unadjusted") #"adjusted")
DATA_MODES=("clinical" "methylation" "combined")

for cohort in "${COHORTS[@]}"; do
  for mode in "${DATA_MODES[@]}"; do
    for meth in "${METH_TYPES[@]}"; do
      echo ">>> Running: python -u $SCRIPT --cohort_name $cohort --data_mode $mode --methylation_type $meth"
      python -u "$SCRIPT" --cohort_name "$cohort" --methylation_type "$meth" --data_mode "$mode"
    done
  done
done

echo "All runs completed."