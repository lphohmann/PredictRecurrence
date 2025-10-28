#!/usr/bin/env bash
set -e

# example how to run
#./scripts/Bash_scripts/run_all_indiv_approach.sh ./scripts/CoxNet/CoxNet_pipeline.py 

# User can pass a script path as the first argument, otherwise default
SCRIPT="${1:-./scripts/CoxNet/CoxNet_pipeline.py}"

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
