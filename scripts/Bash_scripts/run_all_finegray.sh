#!/usr/bin/env bash

set -e  # stop on error

SCRIPT="./scripts/Fine-Gray/FineGray_pipeline_innerCVstability.R"

COHORTS=("TNBC" "ERpHER2n" "All")
MODES=("methylation" "combined")

for COHORT in "${COHORTS[@]}"; do
  for MODE in "${MODES[@]}"; do
    echo "===================================================="
    echo "Running Fine-Gray pipeline"
    echo "  Cohort:    ${COHORT}"
    echo "  Data mode: ${MODE}"
    echo "===================================================="

    Rscript "${SCRIPT}" "${COHORT}" "${MODE}"

    echo "✓ Finished ${COHORT} / ${MODE}"
    echo
  done
done

echo "🎉 All Fine-Gray runs completed"

