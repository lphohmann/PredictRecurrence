#!/usr/bin/env python

################################################################################
# Script: Filter CpGs for Survival Analysis
# Author: lennart hohmann
# Description: Performs two-stage CpG filtering:
#              1. Initial variance-based filtering to reduce dimensionality.
#              2. Univariate Cox regression filtering based on p-values.
#              Saves the list of selected CpG IDs for use in main pipelines.
################################################################################

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import warnings
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests
from sklearn.exceptions import ConvergenceWarning

# Add project src directory to path for imports (adjust as needed)
# This is needed if your `log` or `load_training_data` are in `src/utils.py`
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
from src.utils import log, load_training_data, beta2m, variance_filter, run_univariate_cox_for_cpgs, plot_hr_histogram, plot_pvalue_histograms, plot_histogram

# Set working directory
os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))

# ==============================================================================
# INPUT FILES
# ==============================================================================

infile_train_ids = "./data/train/train_subcohorts/TNBC_train_ids.csv" # ⚠️ ADAPT
infile_betavalues = "./data/train/train_methylation_adjusted.csv" # ⚠️ ADAPT
infile_clinical = "./data/train/train_clinical.csv"

# ==============================================================================
# PARAMS for Filtering
# ==============================================================================

# Output file for selected CpG IDs
output_dir = "./data/set_definitions/CpG_prefiltered_sets/TNBC/Adjusted/" # ⚠️ ADAPT
os.makedirs(output_dir, exist_ok=True)
outfile_filtered_cpgs = os.path.join(output_dir, "variance_plus_univarcox.txt") # ⚠️ ADAPT

logfile = open(os.path.join(output_dir, "run.log"), "w")
sys.stdout = logfile
sys.stderr = logfile

# Filtering parameters
initial_top_n_variance_cpgs = 100000 # <--- First stage: filter to this many CpGs by variance
final_top_n_univariate_cox_cpgs = 1000 # <--- Second stage: select this many CpGs by adjusted p-value
univariate_cox_penalizer_value = 0.01 # Penalizer for CoxPHFitter

# ==============================================================================
# MAIN FILTERING PIPELINE
# ==============================================================================

start_time = time.time()
log(f"CpG filtering pipeline started at: {time.ctime(start_time)}")

# 1. Load Data
train_ids = pd.read_csv(infile_train_ids, header=None).iloc[:, 0].tolist()
log("Loaded training IDs.")
beta_matrix, clinical_data = load_training_data(train_ids, infile_betavalues, infile_clinical)
log(f"Loaded methylation data (initial CpGs: {beta_matrix.shape[1]}) and clinical data.")

# 2. Convert Beta-values to M-values
mvals = beta2m(beta_matrix)
log("Converted beta values to M-values.")

# 3. Initial Variance Filtering
mvals_variance_filtered = variance_filter(mvals, top_n=initial_top_n_variance_cpgs)
log(f"Applied initial variance filter: {mvals_variance_filtered.shape[1]} CpGs remaining.")

# 4. Univariate Cox Regression Filtering
univariate_cox_results = run_univariate_cox_for_cpgs(
    mval_matrix=mvals_variance_filtered,
    clin_data=clinical_data,
    time_col="RFi_years",
    event_col="RFi_event",
    penalizer_value=univariate_cox_penalizer_value
)

# Plotting and Summary after Univariate Cox ---
log("--- Univariate Cox Regression Results Summary ---")
num_nan_padj = univariate_cox_results["padj"].isna().sum()
log(f"Number of CpGs with NaN adjusted p-values (failed univariate Cox fits): {num_nan_padj}")
log(f"Total CpGs processed by univariate Cox: {univariate_cox_results.shape[0]}")
log(f"Number of CpGs with padj < 0.05: {univariate_cox_results[univariate_cox_results['padj'] < 0.05].shape[0]}")
log(f"Number of CpGs with padj < 0.01: {univariate_cox_results[univariate_cox_results['padj'] < 0.01].shape[0]}")

# Generate and save histograms
plot_pvalue_histograms(univariate_cox_results, output_dir)
plot_hr_histogram(univariate_cox_results, output_dir)
log("-------------------------------------------------")

# 5. Select Final CpGs based on adjusted p-value
num_nan_padj = univariate_cox_results["padj"].isna().sum()
log(f"Number of CpGs with NaN adjusted p-values (failed univariate Cox fits): {num_nan_padj}")

# Ensure to drop NaNs from pval for sorting, as failed fits will have NaN pvals
filtered_results = univariate_cox_results.dropna(subset=["padj"]).sort_values(by="padj", ascending=True)
selected_cpg_ids = filtered_results["CpG_ID"].head(final_top_n_univariate_cox_cpgs).tolist()

log(f"Final selection: {len(selected_cpg_ids)} CpGs selected based on univariate Cox adjusted p-value (top {final_top_n_univariate_cox_cpgs}).")

# 6. Save the selected CpG IDs
# Save as a plain text file, one CpG ID per line
with open(outfile_filtered_cpgs, 'w') as f:
    for cpg_id in selected_cpg_ids:
        f.write(f"{cpg_id}\n")
log(f"Saved selected CpG IDs to: {outfile_filtered_cpgs}")

end_time = time.time()
log(f"CpG filtering pipeline ended at: {time.ctime(end_time)}")
log(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes.")
logfile.close()