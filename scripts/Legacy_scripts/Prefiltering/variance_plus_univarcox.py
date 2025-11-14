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
from statsmodels.stats.multitest import multipletests
import argparse

# Add project src directory to path for imports (adjust as needed)
# This is needed if your `log` or `load_training_data` are in `src/utils.py`
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
from src.utils import log, load_training_data, beta2m, variance_filter, run_univariate_cox_for_cpgs, plot_hr_histogram, plot_pvalue_histograms

# Set working directory
os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))

# ==============================================================================
# MAPPINGS OF INPUT FILES
# ==============================================================================

# Mapping of methylation types to their respective file paths
METHYLATION_DATA_PATHS = {
    "adjusted": "./data/train/train_methylation_adjusted.csv",
    "unadjusted": "./data/train/train_methylation_unadjusted.csv"
}

# Mapping of cohort names to their training IDs file paths
COHORT_TRAIN_IDS_PATHS = {
    "TNBC": "./data/train/train_subcohorts/TNBC_train_ids.csv",
    "ERpHER2n": "./data/train/train_subcohorts/ERpHER2n_train_ids.csv",
    "All": "./data/train/train_subcohorts/All_train_ids.csv"
}

# Common clinical data file path
INFILE_CLINICAL = "./data/train/train_clinical.csv"

# ==============================================================================
# COMMAND LINE ARGUMENTS
# ==============================================================================

parser = argparse.ArgumentParser(description="Filter CpGs using variance + Cox Lasso.")
parser.add_argument("--cohort_name", type=str, required=True,
                    choices=COHORT_TRAIN_IDS_PATHS.keys(), 
                    help="Name of the cohort to process")
parser.add_argument("--methylation_type", type=str, choices=METHYLATION_DATA_PATHS.keys(), required=True,
                    help="Type of methylation data")
# defaults, no need to change usually
parser.add_argument("--output_base_dir", type=str, default="./data/set_definitions/CpG_prefiltered_sets/",
                    help="Base output directory")
parser.add_argument("--output_cpg_filename", type=str, default="univarcox_selected_cpgs.txt",
                    help="Filename for the text file containing the selected CpG IDs")
args = parser.parse_args()

# ==============================================================================
# INPUT FILES
# ==============================================================================

infile_train_ids = COHORT_TRAIN_IDS_PATHS[args.cohort_name]
infile_betavalues = METHYLATION_DATA_PATHS[args.methylation_type] 
infile_clinical = INFILE_CLINICAL

# ==============================================================================
# PARAMS for Filtering
# ==============================================================================

# Cohort-specific output directories
current_output_dir = os.path.join(
    args.output_base_dir,
    args.cohort_name,
    args.methylation_type.capitalize() # Include methylation type in output dir
)
os.makedirs(current_output_dir, exist_ok=True)

# Logfile is now directly in the cohort's output directory
logfile_path = os.path.join(current_output_dir, "univarcox_run.log")
logfile = open(logfile_path, "w")
sys.stdout = logfile
sys.stderr = logfile

# Filtering parameters
INITIAL_VARIANCE_FILTER_TOP_N = 50000 # <--- First stage: filter to this many CpGs by variance
FINAL_UNIVARCOX_TARGET_N = 200 # <--- Second stage: select this many CpGs by adjusted p-value
UNIVAR_COX_PENALIZER_VALUE = 0.01 # Penalizer for CoxPHFitter

# ==============================================================================
# MAIN FILTERING PIPELINE
# ==============================================================================

start_time = time.time()
log(f"CpG filtering pipeline started at: {time.ctime(start_time)}")
log(f"Processing Cohort: {args.cohort_name}, Methylation Type: {args.methylation_type}")
log(f"Train IDs file: {COHORT_TRAIN_IDS_PATHS[args.cohort_name]}")
log(f"Methylation data file: {METHYLATION_DATA_PATHS[args.methylation_type]}")
log(f"Clinical data file: {INFILE_CLINICAL}")
log(f"Output directory: {current_output_dir}")
log(f"Output CpG filename: {args.output_cpg_filename}")

# ================================
# --- Load Data ---
# ================================
 
train_ids = pd.read_csv(COHORT_TRAIN_IDS_PATHS[args.cohort_name], header=None).iloc[:, 0].tolist()
log("Loaded training IDs.")

beta_matrix, clinical_data = load_training_data(train_ids, METHYLATION_DATA_PATHS[args.methylation_type], INFILE_CLINICAL)
log(f"Loaded methylation data (initial CpGs: {beta_matrix.shape[1]}) and clinical data.")

mvals = beta2m(beta_matrix)
log("Converted beta values to M-values.")

# ================================
# --- Initial Variance Filtering --- 
# ================================

mvals_variance_filtered = variance_filter(mvals, top_n=INITIAL_VARIANCE_FILTER_TOP_N)
log(f"Applied initial variance filter: {mvals_variance_filtered.shape[1]} CpGs remaining.")

# ================================
# --- Univariate Cox Regression Filtering --- 
# ================================

univariate_cox_results = run_univariate_cox_for_cpgs(
    mval_matrix=mvals_variance_filtered,
    clin_data=clinical_data,
    time_col="RFi_years",
    event_col="RFi_event",
    penalizer_value=UNIVAR_COX_PENALIZER_VALUE
)

# ================================
# --- Plotting and Summary after Univariate Cox --- 
# ================================

log("--- Univariate Cox Regression Results Summary ---")
num_nan_padj = univariate_cox_results["padj"].isna().sum()
log(f"Number of CpGs with NaN adjusted p-values (failed univariate Cox fits): {num_nan_padj}")
log(f"Total CpGs processed by univariate Cox: {univariate_cox_results.shape[0]}")
log(f"Number of CpGs with padj < 0.05: {univariate_cox_results[univariate_cox_results['padj'] < 0.05].shape[0]}")
log(f"Number of CpGs with padj < 0.01: {univariate_cox_results[univariate_cox_results['padj'] < 0.01].shape[0]}")

# Generate and save histograms
plot_pvalue_histograms(univariate_cox_results, current_output_dir)
plot_hr_histogram(univariate_cox_results, current_output_dir)
log("-------------------------------------------------")

# ================================
# --- Select Final CpGs based on adjusted p-value --- 
# ================================

#num_nan_padj = univariate_cox_results["padj"].isna().sum()
#log(f"Number of CpGs with NaN adjusted p-values (failed univariate Cox fits): {num_nan_padj}")

# Ensure to drop NaNs from pval for sorting, as failed fits will have NaN pvals
#filtered_results = univariate_cox_results.dropna(subset=["padj"]).sort_values(by="padj", ascending=True)
#selected_cpg_ids = filtered_results["CpG_ID"].head(FINAL_UNIVARCOX_TARGET_N).tolist()

#log(f"Final selection: {len(selected_cpg_ids)} CpGs selected based on univariate Cox adjusted p-value (top {FINAL_UNIVARCOX_TARGET_N}).")

# ================================
# --- Select Final CpGs based on Combined Score (prioritizing significance) --- 
# ================================

num_nan_values = univariate_cox_results[["padj", "HR"]].isna().any(axis=1).sum()
log(f"Number of CpGs with NaN padj or HR (failed fits): {num_nan_values}")

# Create combined score for valid CpGs
filtered_results = univariate_cox_results.dropna(subset=["padj", "HR"])

if len(filtered_results) == 0:
    log("ERROR: No valid CpGs after removing NaN values. Cannot proceed with selection.")
    selected_cpg_ids = []
else:
    # Calculate combined score: -log10(padj) * |log(HR)|
    filtered_results["combined_score"] = -np.log10(filtered_results["padj"]) * np.abs(np.log(filtered_results["HR"]))
    
    # Check for significant CpGs first
    significant_cpgs = filtered_results[filtered_results["padj"] < 0.05]
    
    if len(significant_cpgs) > 0:
        # Select only from significant CpGs
        n_to_select = min(FINAL_UNIVARCOX_TARGET_N, len(significant_cpgs))
        selected_cpg_ids = significant_cpgs.nlargest(n_to_select, "combined_score")["CpG_ID"].tolist()
        selection_method = "significant_only"
        log(f"Found {len(significant_cpgs)} significant CpGs (padj < 0.05). Selected {len(selected_cpg_ids)} based on combined score.")
        
        if len(significant_cpgs) < FINAL_UNIVARCOX_TARGET_N:
            log(f"NOTE: Only {len(significant_cpgs)} significant CpGs available (target was {FINAL_UNIVARCOX_TARGET_N}).")
    else:
        # Fallback: no significant CpGs, select from all based on combined score
        n_to_select = min(FINAL_UNIVARCOX_TARGET_N, len(filtered_results))
        selected_cpg_ids = filtered_results.nlargest(n_to_select, "combined_score")["CpG_ID"].tolist()
        selection_method = "fallback_all"
        log(f"WARNING: No CpGs with padj < 0.05 found. Selected top {len(selected_cpg_ids)} CpGs by combined score from all available CpGs.")
    
    # Get statistics for selected CpGs
    selected_data = filtered_results[filtered_results["CpG_ID"].isin(selected_cpg_ids)]
    n_significant_selected = (selected_data["padj"] < 0.05).sum()
    
    # Log selection summary
    log(f"Final selection: {len(selected_cpg_ids)} CpGs selected using {selection_method} method.")
    log(f"Selected CpGs with padj < 0.05: {n_significant_selected}/{len(selected_cpg_ids)}")
    
    # Log detailed statistics
    log(f"Selected CpGs - HR range: {selected_data['HR'].min():.3f} to {selected_data['HR'].max():.3f}")
    log(f"Selected CpGs - padj range: {selected_data['padj'].min():.2e} to {selected_data['padj'].max():.2e}")
    log(f"Selected CpGs - Combined score range: {selected_data['combined_score'].min():.3f} to {selected_data['combined_score'].max():.3f}")


# ================================
# --- Save Selected CpGs --- 
# ================================

# Save as a plain text file, one CpG ID per line
outfile_filtered_cpgs = os.path.join(current_output_dir, args.output_cpg_filename)
with open(outfile_filtered_cpgs, 'w') as f:
    for cpg_id in selected_cpg_ids:
        f.write(f"{cpg_id}\n")
log(f"Saved selected CpG IDs to: {outfile_filtered_cpgs}")

# ================================
# ================================

end_time = time.time()
log(f"CpG filtering pipeline ended at: {time.ctime(end_time)}")
log(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes.")
logfile.close()