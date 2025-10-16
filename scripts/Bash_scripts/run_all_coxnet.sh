#!/usr/bin/env bash
set -e

SCRIPT="./scripts/CoxNet/CoxNet_pipeline.py"
COHORTS=("TNBC" "ERpHER2n" "All")
METH_TYPES=("unadjusted" "adjusted")
DATA_MODES=("clinical" "methylation" "combined")

for cohort in "${COHORTS[@]}"; do
  for mode in "${DATA_MODES[@]}"; do
    if [[ "$mode" == "clinical" ]]; then
      # For clinical mode, methylation type doesn't matter — run only once with unadjusted
      meth="unadjusted"
      echo ">>> Running: python -u $SCRIPT --cohort_name $cohort --methylation_type $meth --data_mode $mode"
      python -u "$SCRIPT" --cohort_name "$cohort" --methylation_type "$meth" --data_mode "$mode"
    else
      # For methylation and combined: run both methylation types
      for meth in "${METH_TYPES[@]}"; do
        echo ">>> Running: python -u $SCRIPT --cohort_name $cohort --methylation_type $meth --data_mode $mode"
        python -u "$SCRIPT" --cohort_name "$cohort" --methylation_type "$meth" --data_mode "$mode"
      done
    fi
  done
done

echo "All CoxNet runs completed."
