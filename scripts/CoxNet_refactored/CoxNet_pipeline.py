#!/usr/bin/env python

################################################################################
# Script: CoxNet pipeline
# Author: Lennart Hohmann
################################################################################

################################################################################
# SET UP
################################################################################

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from sksurv.util import Surv

# Import custom CoxNet functions
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
from src.utils import (
    load_training_data,
    preprocess_data,
    define_param_grid,
    run_nested_cv,
    summarize_outer_models,
    evaluate_outer_models,
    plot_brier_scores,
    plot_auc_curves,
    summarize_performance,
    select_best_model
)

# Set working directory
os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))

start_time = time.time()
print(f"Script started at: {time.ctime(start_time)}", flush=True)

################################################################################
# PARAMS
################################################################################

# Input files
infile_train_ids = "./data/train/train_subcohorts/TNBC_train_ids.csv"
infile_betavalues = "./data/train/train_methylation_adjusted.csv"
infile_clinical = "./data/train/train_clinical.csv"

# Output directory and files
output_dir = "output/CoxNet_rf"
os.makedirs(output_dir, exist_ok=True)
outfile_outermodels = os.path.join(output_dir, "outer_cv_models.pkl")
outfile_brierplot = os.path.join(output_dir, "brier_scores.png")
outfile_AUCplot = os.path.join(output_dir, "auc_curves.png")
outfile_bestCVfold = os.path.join(output_dir, "best_outer_fold.pkl")

# log file
out_logfile = os.path.join(output_dir, "run.log")
logfile = open(out_logfile, "w")
sys.stdout = logfile
sys.stderr = logfile

# Parameters
top_n_cpgs = 200000
inner_cv_folds = 3
outer_cv_folds = 5
eval_time_grid = np.arange(1, 10.1, 0.5)  # 1 to 10 in steps of 0.5

################################################################################
# MAIN PIPELINE
################################################################################

# Load and prepare data
train_ids = pd.read_csv(infile_train_ids, header=None).iloc[:, 0].tolist()
beta_matrix, clinical_data = load_training_data(train_ids, infile_betavalues, infile_clinical)
X = preprocess_data(beta_matrix, top_n_cpgs=top_n_cpgs)
y = Surv.from_dataframe("RFi_event", "RFi_years", clinical_data)

# Define hyperparameter grid
param_grid = define_param_grid(X, y, n_alphas=30)

# Run nested cross-validation
outer_models = run_nested_cv(X, y, param_grid, outer_cv_folds, inner_cv_folds)
joblib.dump(outer_models, outfile_outermodels)
print(f"Saved refitted best model per outer fold to: {outfile_outermodels}", flush=True)

# Summarize and evaluate models
summarize_outer_models(outer_models)
model_performances = evaluate_outer_models(outer_models, X, y, eval_time_grid)

# Extract fold and evaluation results
folds = [p["fold"] for p in model_performances]
brier_array = np.array([p["brier_t"] for p in model_performances])
ibs_array = np.array([p["ibs"] for p in model_performances])

# Visualize and summarize performance
plot_brier_scores(brier_array, ibs_array, folds, eval_time_grid, outfile_brierplot)
plot_auc_curves(model_performances, eval_time_grid, outfile_AUCplot)
summarize_performance(model_performances)

# Select best model based on specified metric
metric = "mean_auc"  # could also be "ibs" or "auc_at_5y"
best_outer_fold = select_best_model(model_performances, outer_models, metric)
joblib.dump(best_outer_fold, outfile_bestCVfold)
print(f"\nBest fold ({best_outer_fold['fold']}) saved to: {outfile_bestCVfold}")

################################################################################
# THE END
################################################################################

end_time = time.time()  # Record end time
print(f"Script ended at: {time.ctime(end_time)}")
print(f"Script executed in {end_time - start_time:.2f} seconds.")
logfile.close()