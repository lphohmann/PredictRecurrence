#!/usr/bin/env bash
set -euo pipefail

SCRIPT="./scripts/Fine-Gray/FineGray_dual_pipeline.R"
COHORTS=("TNBC" "ERpHER2n" "All")

for COHORT in "${COHORTS[@]}"; do
  echo "========================================"
  echo "Running Fine-Gray pipeline for cohort: ${COHORT}"
  echo "========================================"

  Rscript "${SCRIPT}" "${COHORT}"

  echo "Finished cohort: ${COHORT}"
  echo
done

echo "All cohorts completed."

