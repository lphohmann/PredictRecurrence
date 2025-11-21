#!/usr/bin/env bash
set -e

# example how to run
#./scripts/Bash_scripts/run_all_indiv_approach.sh ./scripts/CoxNet/CoxNet_pipeline.py 

# User can pass a script path as the first argument, otherwise default
SCRIPT="${1:-./scripts/CoxNet/CoxNet_pipeline.py}"

COHORTS=("TNBC" "ERpHER2n" "All")
DATA_MODES=("clinical" "methylation" "combined")

for cohort in "${COHORTS[@]}"; do
  for mode in "${DATA_MODES[@]}"; do
      echo ">>> Running: python -u $SCRIPT --cohort_name $cohort --data_mode $mode --train_cpgs ./data/set_definitions/CpG_prefiltered_sets/cpg_ids_atac_overlap.txt"
      python -u "$SCRIPT" --cohort_name "$cohort" --data_mode "$mode" --train_cpgs ./data/set_definitions/CpG_prefiltered_sets/cpg_ids_atac_overlap.txt
  done
done

echo "All runs completed."
