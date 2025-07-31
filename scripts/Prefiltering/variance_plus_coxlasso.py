#!/usr/bin/env python

################################################################################
# Script: Cox Lasso CpG Filtering Pipeline
# Author: lennart hohmann (adapted by Gemini)
# Description: Performs two-stage CpG filtering:
#              1. Initial variance-based filtering to reduce dimensionality.
#              2. Cox Lasso (L1-regularized Cox regression) filtering
#                 to select outcome-relevant CpGs.
#              Saves the list of selected CpG IDs and coefficient plots.
#              Supports single cohort and adjusted/unadjusted methylation data.
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
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Sci-kit learn and Sci-kit survival imports for Cox Lasso
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import as_concordance_index_ipcw_scorer
from sksurv.util import Surv # For creating structured array for survival data

# Add project src directory to path for imports (adjust as needed)
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
from src.utils import log, load_training_data, beta2m, variance_filter, plot_coefficients_histogram
from src.coxnet_functions import estimate_alpha_grid
# Set working directory to the project root
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
parser.add_argument("--output_cpg_filename", type=str, default="cox_lasso_selected_cpgs.txt",
                    help="Filename for the text file containing the selected CpG IDs")
args = parser.parse_args()

# ==============================================================================
# PARAMS
# ==============================================================================

# Cohort-specific output directories
current_output_dir = os.path.join(
    args.output_base_dir,
    args.cohort_name,
    args.methylation_type.capitalize() # Include methylation type in output dir
)
os.makedirs(current_output_dir, exist_ok=True)

# Logfile is now directly in the cohort's output directory
logfile_path = os.path.join(current_output_dir, "coxlasso_run.log")
logfile = open(logfile_path, "w")
sys.stdout = logfile
sys.stderr = logfile

# Hardcoded Filtering parameters
MIN_SAMPLES_PER_CPG_THRESHOLD = 0.95 # Minimum proportion of non-missing values for a CpG to be kept
INITIAL_VARIANCE_FILTER_TOP_N = 1000 # First stage: filter to this many CpGs by variance
FINAL_COX_LASSO_TARGET_N = 1000 # Target number of CpGs to select after Cox Lasso.

# Hardcoded Cox Lasso tuning parameters (for the feature selection step)
#COX_LASSO_ALPHA_VALUES = np.array([0.0001, 0.001, 0.01, 0.1]) # This is a 1D array of alpha values
# use estimate_alpha_grid() instead
COX_LASSO_L1_RATIO_VALUES = np.array([0.9, 1.0]) # L1_ratio values for Lasso vs Ridge mix
COX_LASSO_CV_FOLDS = 5 # Number of CV folds for tuning Cox Lasso hyperparameters

# ==============================================================================
# MAIN FILTERING PIPELINE
# ==============================================================================

start_time = time.time()
log(f"CpG filtering pipeline started at: {time.ctime(start_time)}")
log(f"Processing Cohort: {args.cohort_name}, Methylation Type: {args.methylation_type}")
log(f"Train IDs file: {COHORT_TRAIN_IDS_PATHS[args.cohort_name]}")
log(f"Methylation data file: {COHORT_TRAIN_IDS_PATHS[args.cohort_name]}")
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

# Missing Data Filtering, not needed in SCAN-B cohort
initial_cpg_count = mvals.shape[1]
mvals_no_missing = mvals.dropna(axis=1, thresh=int(mvals.shape[0] * MIN_SAMPLES_PER_CPG_THRESHOLD))
log(f"Removed CpGs with >{100*(1-MIN_SAMPLES_PER_CPG_THRESHOLD):.1f}% missing values: {mvals_no_missing.shape[1]} CpGs remaining.")

# ================================
# --- Initial Variance Filtering --- 
# ================================

mvals_filtered_for_lasso = variance_filter(mvals_no_missing, top_n=INITIAL_VARIANCE_FILTER_TOP_N)
log(f"Applied initial variance filter: {mvals_filtered_for_lasso.shape[1]} CpGs remaining for Cox Lasso.")

# ================================
# Prepare survival labels for the full training data (for Cox Lasso tuning)
# ================================

# Use the original column names for accessing fields in y_full
EVENT_FIELD_NAME = "RFi_event"
TIME_FIELD_NAME = "RFi_years"

y_full = Surv.from_dataframe(EVENT_FIELD_NAME, TIME_FIELD_NAME, clinical_data)
log(f"Prepared survival labels for {len(y_full)} samples ({y_full[EVENT_FIELD_NAME].sum()} events).")

# ================================
# --- Cox Lasso Feature Selection ---
# ================================

log(f"Starting Cox Lasso feature selection on {mvals_filtered_for_lasso.shape[1]} CpGs...")

# Use CoxnetSurvivalAnalysis in the pipeline
cox_lasso_pipe = make_pipeline(CoxnetSurvivalAnalysis())
scorer_cox_lasso_pipe = as_concordance_index_ipcw_scorer(cox_lasso_pipe)

# Construct the param_grid for GridSearchCV 
alphas = estimate_alpha_grid(mvals_filtered_for_lasso, y_full, l1_ratio=0.9, alpha_min_ratio=0.1, n_alphas=10)

# CORRECTED: Each alpha value must be wrapped in its own list for CoxnetSurvivalAnalysis
param_grid_for_gs = {
    'estimator__coxnetsurvivalanalysis__alphas': [[alpha] for alpha in alphas], 
    'estimator__coxnetsurvivalanalysis__l1_ratio': COX_LASSO_L1_RATIO_VALUES
}

inner_cv = StratifiedKFold(n_splits=COX_LASSO_CV_FOLDS, shuffle=True, random_state=12)

grid_search_cox_lasso = GridSearchCV(
    scorer_cox_lasso_pipe,
    param_grid=param_grid_for_gs,
    cv=inner_cv, 
    #scoring=as_concordance_index_ipcw_scorer(cox_lasso_pipe, times=time_points_for_scorer),
    n_jobs=-1, # Use all available cores for grid search
    error_score=0, # Handle errors gracefully during grid search
    verbose=1 # Adjust verbosity for grid search output
)

grid_search_cox_lasso.fit(mvals_filtered_for_lasso, y_full)

best_cox_lasso_model = grid_search_cox_lasso.best_estimator_
best_cox_lasso_params = grid_search_cox_lasso.best_params_

log(f"Best Cox Lasso parameters: {best_cox_lasso_params}")
log(f"Best Cox Lasso score (IPCW C-index): {grid_search_cox_lasso.best_score_:.4f}")

# ================================
# Extract coefficients from the best Cox Lasso model
# ================================

# Access CoxnetSurvivalAnalysis estimator
cox_estimator = best_cox_lasso_model.named_steps['coxnetsurvivalanalysis']
# For CoxnetSurvivalAnalysis, coef_ is a 2D array (n_features, n_alphas).
# Since we passed single alphas, it's (n_features, 1). We need to flatten it.
coefficients = pd.Series(cox_estimator.coef_.flatten(), index=mvals_filtered_for_lasso.columns)

# ================================
# Select CpGs with non-zero coefficients
# ================================

selected_cpg_ids_from_lasso = coefficients[coefficients != 0].index.tolist()
log(f"Cox Lasso identified {len(selected_cpg_ids_from_lasso)} non-zero CpGs.")

# Apply final top_n selection if needed
if len(selected_cpg_ids_from_lasso) > FINAL_COX_LASSO_TARGET_N:
    selected_cpg_ids = coefficients.abs().nlargest(FINAL_COX_LASSO_TARGET_N).index.tolist()
    log(f"Truncating Cox Lasso selected CpGs to top {len(selected_cpg_ids)} by absolute coefficient value (target: {FINAL_COX_LASSO_TARGET_N}).")
elif len(selected_cpg_ids_from_lasso) < FINAL_COX_LASSO_TARGET_N:
    log(f"Warning: Cox Lasso selected only {len(selected_cpg_ids_from_lasso)} non-zero CpGs, which is less than the target {FINAL_COX_LASSO_TARGET_N}. Consider adjusting `COX_LASSO_ALPHA_VALUES` (e.g., smaller alpha) or `FINAL_COX_LASSO_TARGET_N`.")
    selected_cpg_ids = selected_cpg_ids_from_lasso
else:
    selected_cpg_ids = selected_cpg_ids_from_lasso

log(f"Final selection: {len(selected_cpg_ids)} CpGs selected for downstream analysis.")

# ================================
# --- Save Results ---
# ================================

outfile_filtered_cpgs = os.path.join(current_output_dir, args.output_cpg_filename)
with open(outfile_filtered_cpgs, 'w') as f:
    for cpg_id in selected_cpg_ids:
        f.write(f"{cpg_id}\n")
log(f"Saved selected CpG IDs to: {outfile_filtered_cpgs}")

# ================================
# --- Generate Plots ---
# ================================

log("Generating coefficient histograms...")
plot_coefficients_histogram(coefficients,
                            title=f"Cox Lasso Coefficients (All) - {args.cohort_name} ({args.methylation_type.capitalize()})",
                            xlabel="Coefficient Value",
                            outfile=os.path.join(current_output_dir, "cox_lasso_coefficients_all_histogram.png"))

plot_coefficients_histogram(coefficients[coefficients != 0],
                            title=f"Cox Lasso Coefficients (Non-Zero) - {args.cohort_name} ({args.methylation_type.capitalize()})",
                            xlabel="Coefficient Value",
                            outfile=os.path.join(current_output_dir, "cox_lasso_coefficients_nonzero_histogram.png"))
log("Coefficient histograms generated.")

# ================================
# ================================

end_time = time.time()
log(f"CpG filtering pipeline ended at: {time.ctime(end_time)}")
log(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes.")
logfile.close() # Close the logfile for the current run
