#!/usr/bin/env python

################################################################################
# Script: RSF Pipeline
# Author: lennart hohmann
################################################################################

################################################################################
# SET UP
################################################################################

import os, sys, time
import numpy as np
import pandas as pd
import joblib
from sksurv.util import Surv

# Add project src directory to path for imports (adjust as needed)
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
from src.utils import log, load_training_data, preprocess_data
from src.rsf_functions import (
    define_param_grid,
    run_nested_cv,
    summarize_outer_models,
    evaluate_outer_models,
    plot_brier_scores,
    plot_auc_curves,
    summarize_performance,
    select_best_model#,compute_permutation_importance
)

# Set working directory
os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))

################################################################################
# INPUT FILES
################################################################################

infile_train_ids = "./data/train/train_subcohorts/TNBC_train_ids.csv"
infile_betavalues = "./data/train/train_methylation_adjusted.csv"  # ⚠️ ADAPT: adjusted/unadjusted
infile_clinical = "./data/train/train_clinical.csv"

################################################################################
# PARAMS
################################################################################

output_dir = "output/RSF_adjusted" # ⚠️ ADAPT: adjusted/unadjusted
os.makedirs(output_dir, exist_ok=True)
outfile_outermodels = os.path.join(output_dir, "outer_cv_models.pkl")
outfile_brierplot = os.path.join(output_dir, "brier_scores.png")
outfile_aucplot = os.path.join(output_dir, "auc_curves.png")
outfile_bestfold = os.path.join(output_dir, "best_outer_fold.pkl")
outfile_importancebyfold = os.path.join(output_dir, "importances_by_fold.pkl")

logfile = open(os.path.join(output_dir, "pipeline_run.log"), "w")
sys.stdout = logfile
sys.stderr = logfile

# Data preprocessing parameters
top_n_cpgs = 1000 #10000  # as in CoxNet or fewer?
inner_cv_folds = 3
outer_cv_folds = 5
eval_time_grid = np.arange(1, 10.1, 0.5)  # time points for metrics

################################################################################
# MAIN PIPELINE
################################################################################

start_time = time.time()
log(f"RSF pipeline started at: {time.ctime(start_time)}")

# Load and preprocess data (same as CoxNet pipeline)
train_ids = pd.read_csv(infile_train_ids, header=None).iloc[:, 0].tolist()
log("Loaded training IDs.")
beta_matrix, clinical_data = load_training_data(train_ids, infile_betavalues, infile_clinical)
log("Loaded methylation and clinical data.")
X = preprocess_data(beta_matrix, top_n_cpgs=top_n_cpgs)
log("Preprocessed feature matrix (top CpGs).")
# Prepare survival labels (Surv object with event & time)
y = Surv.from_dataframe("RFi_event", "RFi_years", clinical_data)

# Define hyperparameter grid
param_grid = define_param_grid(X, y)

# Run nested cross-validation
outer_models = run_nested_cv(X, y, param_grid, outer_cv_folds, inner_cv_folds,
                             inner_scorer="concordance_index_ipcw")
joblib.dump(outer_models, outfile_outermodels)
log(f"Saved outer CV models to: {outfile_outermodels}")

# Summarize and evaluate performance
summarize_outer_models(outer_models)
model_performances = evaluate_outer_models(outer_models, X, y, eval_time_grid)

# Extract arrays for plotting
folds = [p["fold"] for p in model_performances]
brier_array = np.array([p["brier_t"] for p in model_performances])
ibs_array = np.array([p["ibs"] for p in model_performances])

# Plot performance metrics
log("Generating performance plots.")
plot_brier_scores(brier_array, ibs_array, folds, eval_time_grid, outfile_brierplot)
plot_auc_curves(model_performances, eval_time_grid, outfile_aucplot)
summarize_performance(model_performances)

# Average importances_mean across folds
#outer_models = joblib.load(outfile_outermodels)
#importances_by_fold = compute_permutation_importance(outer_models, X, y)
#joblib.dump(importances_by_fold, outfile_importancebyfold)
#importances_all_folds = [m["feature_importances"]["importances_mean"] for m in outer_models]
#mean_importances = pd.concat(importances_all_folds, axis=1).mean(axis=1)
#mean_importances.sort_values(ascending=False).head(20)

# Select and save the best model (by chosen metric)
metric = "mean_auc"  # could be "ibs" or "auc_at_5y"
best_outer_fold = select_best_model(model_performances, outer_models, metric)
if best_outer_fold:
    joblib.dump(best_outer_fold, outfile_bestfold)
    log(f"Best model (fold {best_outer_fold['fold']}) saved to: {outfile_bestfold}")

end_time = time.time()
log(f"RSF pipeline ended at: {time.ctime(end_time)}")
log(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes.")
logfile.close()
