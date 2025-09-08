#!/usr/bin/env python

################################################################################
# Script: CoxLasso Pipeline
# Author: lennart hohmann
################################################################################

################################################################################
# IMPORTS
################################################################################

import os, sys, time
import numpy as np
import pandas as pd
import joblib
from sksurv.util import Surv
import argparse
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.preprocessing import RobustScaler

# Add project src directory to path for imports (adjust as needed)
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
from src.utils import log, load_training_data, beta2m, apply_admin_censoring, summarize_outer_models, summarize_performance,select_best_model, estimate_alpha_grid
from src.plotting_functions import plot_brier_scores, plot_auc_curves
from src.coxnet_functions import define_param_grid, evaluate_outer_models_coxnet, run_nested_cv_cox, print_selected_cpgs_counts

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

parser = argparse.ArgumentParser(description="Train CoxLasso.")
parser.add_argument("--cohort_name", type=str, required=True,
                    choices=COHORT_TRAIN_IDS_PATHS.keys(), 
                    help="Name of the cohort to process")
parser.add_argument("--methylation_type", type=str, choices=METHYLATION_DATA_PATHS.keys(), required=True,
                    help="Type of methylation data")
parser.add_argument("--train_cpgs", type=str, default=None,
                    help="Set of CpGs for training")
# defaults, no need to change usually
parser.add_argument("--output_base_dir", type=str, default="./output/CoxLasso",
                    help="Base output directory")
args = parser.parse_args()

# ==============================================================================
# PARAMS
# ==============================================================================

# Cohort-specific output directories
current_output_dir = os.path.join(
    args.output_base_dir,
    args.cohort_name,
    args.methylation_type.capitalize())
os.makedirs(current_output_dir, exist_ok=True)

# Logfile is now directly in the cohort's output directory
logfile_path = os.path.join(current_output_dir, "pipeline_run.log")
logfile = open(logfile_path, "w")
sys.stdout = logfile
sys.stderr = logfile

# Data preprocessing parameters
INNER_CV_FOLDS = 3
OUTER_CV_FOLDS = 5
EVAL_TIME_GRID = np.arange(1, 5.1, 0.5)  # time points for metrics

# type of cox regression; for Lasso set both to 1; for Ridge to 0; for ElasticNet to mixed
ALPHAS_ESTIMATION_L1RATIO = 1#0.9#[0.9]
PARAM_GRID_L1RATIOS  = [1.0]#[0.9]

# ==============================================================================
# INPUT AND OUTPUT FILES
# ==============================================================================

infile_train_ids = COHORT_TRAIN_IDS_PATHS[args.cohort_name]
infile_betavalues = METHYLATION_DATA_PATHS[args.methylation_type] 
infile_clinical = INFILE_CLINICAL
infile_cpg_ids = args.train_cpgs

# output files
outfile_outermodels = os.path.join(current_output_dir, "outer_cv_models.pkl")
outfile_brierplot = os.path.join(current_output_dir, "brier_scores.png")
outfile_aucplot = os.path.join(current_output_dir, "auc_curves.png")
outfile_bestfold = os.path.join(current_output_dir, "best_outer_fold.pkl")
outfile_performance = os.path.join(current_output_dir, "outer_cv_performance.pkl")

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

start_time = time.time()
log(f"CoxLasso pipeline started at: {time.ctime(start_time)}")
log(f"Processing Cohort: {args.cohort_name}, Methylation Type: {args.methylation_type}")
log(f"Train IDs file: {COHORT_TRAIN_IDS_PATHS[args.cohort_name]}")
log(f"Methylation data file: {METHYLATION_DATA_PATHS[args.methylation_type]}")
log(f"Clinical data file: {INFILE_CLINICAL}")
log(f"Filtered CpG set data file: {infile_cpg_ids}")
log(f"Output directory: {current_output_dir}")

# Load and preprocess data (same as CoxNet pipeline)
train_ids = pd.read_csv(infile_train_ids, header=None).iloc[:, 0].tolist()
log("Loaded training IDs.")
beta_matrix, clinical_data = load_training_data(train_ids, infile_betavalues, infile_clinical)
log("Loaded methylation and clinical data.")
mvals = beta2m(beta_matrix, beta_threshold=0.001)

# apply censoring at 5 years
# censoring cutoff > max evaluation time
clinical_data = apply_admin_censoring(clinical_data, "RFi_years", "RFi_event", time_cutoff=5.5, inplace=False)

# Subset to only include prefiltered CpGs if infile is provided
if infile_cpg_ids is not None:
    with open(infile_cpg_ids, 'r') as f:
        selected_cpg_ids = [line.strip() for line in f if line.strip()] # empy line would be skipped
    log(f"Successfully loaded {len(selected_cpg_ids)} pre-filtered CpG IDs.")

    # --- Subset mvals to only include prefiltered CpGs ---
    valid_selected_cpg_ids = [cpg for cpg in selected_cpg_ids if cpg in mvals.columns]
    missing_cpgs = [cpg for cpg in selected_cpg_ids if cpg not in mvals.columns]

    if missing_cpgs:
        log(f"Warning: {len(missing_cpgs)} CpGs from the input file are not in the training data: {missing_cpgs}")

    if len(valid_selected_cpg_ids) == 0:
        log("Error: No valid pre-filtered CpGs found in the current methylation data columns.")
        raise ValueError("No valid CpGs to proceed with.")
    
    X = mvals[valid_selected_cpg_ids]
    log(f"Successfully subsetted methylation data to {X.shape[1]} pre-filtered CpGs.")
else:
    # Keep all CpGs
    X = mvals.copy()
    log(f"No pre-filtered CpG file provided. Keeping all {X.shape[1]} CpGs.")

# Prepare survival labels (Surv object with event & time)
y = Surv.from_dataframe("RFi_event", "RFi_years", clinical_data)

# Define hyperparameter grid
alphas = estimate_alpha_grid(X, y, l1_ratio=ALPHAS_ESTIMATION_L1RATIO, n_alphas=20,
                             alpha_min_ratio=0.05)
param_grid = define_param_grid(grid_alphas=alphas, grid_l1ratio=PARAM_GRID_L1RATIOS)

# Run nested cross-validation
estimator = CoxnetSurvivalAnalysis()
scaler = RobustScaler()
outer_models = run_nested_cv_cox(X, y,
                             param_grid=param_grid, outer_cv_folds=OUTER_CV_FOLDS, inner_cv_folds=INNER_CV_FOLDS)

joblib.dump(outer_models, outfile_outermodels)
log(f"Saved outer CV models to: {outfile_outermodels}")

# Summarize and evaluate performance
summarize_outer_models(outer_models)
print_selected_cpgs_counts(outer_models)

model_performances = evaluate_outer_models_coxnet(outer_models, X, y, EVAL_TIME_GRID)
joblib.dump(model_performances, outfile_performance)
print(f"Saved model performances to: {outfile_performance}")

# Extract arrays for plotting
folds = [p["fold"] for p in model_performances]
brier_array = np.array([p["brier_t"] for p in model_performances])
ibs_array = np.array([p["ibs"] for p in model_performances])

# Plot performance metrics
log("Generating performance plots.")
plot_brier_scores(brier_array, ibs_array, folds, EVAL_TIME_GRID, outfile_brierplot)
plot_auc_curves(model_performances, EVAL_TIME_GRID, outfile_aucplot)
summarize_performance(model_performances)

# Select and save the best model (by chosen metric)
metric = "mean_auc"  # could be "ibs" or "auc_at_5y"
best_outer_fold = select_best_model(model_performances, outer_models, metric)
if best_outer_fold:
    joblib.dump(best_outer_fold, outfile_bestfold)
    log(f"Best model (fold {best_outer_fold['fold']}) saved to: {outfile_bestfold}")

end_time = time.time()
log(f"Pipeline ended at: {time.ctime(end_time)}")
log(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes.")
logfile.close()