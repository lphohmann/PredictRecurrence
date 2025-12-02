#!/usr/bin/env bash
set -e

# example how to run:
#   ./scripts/Bash_scripts/run_all_indiv_approach.sh ./scripts/CoxNet/CoxNet_pipeline.py 

# --- Require user to pass a script path ---
if [ -z "$1" ]; then
    echo "ERROR: No script provided."
    echo "Usage: $0 <path_to_python_script>"
    exit 1
fi

SCRIPT="$1"

COHORTS=("TNBC" "ERpHER2n" "All")
DATA_MODES=("clinical" "methylation" "combined")

for cohort in "${COHORTS[@]}"; do
  for mode in "${DATA_MODES[@]}"; do
      echo ">>> Running: python -u $SCRIPT --cohort_name $cohort --data_mode $mode"
      python -u "$SCRIPT" --cohort_name "$cohort" --data_mode "$mode"
  done
done

echo "All runs completed."
