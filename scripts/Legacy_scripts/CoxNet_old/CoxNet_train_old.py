#!/usr/bin/env python

################################################################################
# Script: Training a penalized Cox model (Elastic Net)
# apporach from: https://scikit-survival.readthedocs.io/en/stable/user_guide/coxnet.html
# Author: Lennart Hohmann #/usr/bin/env python
# Date: 22.05.2025
################################################################################

################################################################################
# SET UP
################################################################################

import os
import sys
import time
import importlib
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.exceptions import FitFailedWarning
from sksurv.metrics import concordance_index_ipcw
from sklearn.model_selection import StratifiedKFold

sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
import src.utils
importlib.reload(src.utils)
from src.utils import beta2m, variance_filter, cindex_ipcw_scorer

# set wd
os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))
os.makedirs("output/CoxNet/", exist_ok=True)

start_time = time.time()  # Record start time

print(f"Script started at: {time.ctime(start_time)}",flush=True)

################################################################################
# PARAMS
################################################################################

top_n_cpgs = 200000
outer_cv_folds = 5
inner_cv_folds = 3

################################################################################
# SET FILE PATHS
################################################################################

# input paths
infile_0 = r"./data/train/train_subcohorts/TNBC_train_ids.csv" # subcohort ids
infile_1 = r"./data/train/train_methylation_adjusted.csv"
infile_2 = r"./data/train/train_clinical.csv"

# output paths
outfile_1 = r"output/CoxNet/outer_cv_models.pkl"

################################################################################
# LOAD AND SUBSET TRAINING DATA
################################################################################

# sample training set
train_ids = pd.read_csv(infile_0, header=None).iloc[:, 0].tolist()

# load clin data
clinical_data = pd.read_csv(infile_2)
clinical_data = clinical_data.set_index("Sample")
clinical_data = clinical_data.loc[train_ids]

# load beta values
beta_matrix = pd.read_csv(infile_1,index_col=0).T

# align dataframes
beta_matrix = beta_matrix.loc[train_ids]

################################################################################
# PREPROCESSING
################################################################################

# 1. Convert beta values to M-values with a threshold 
mval_matrix = beta2m(beta_matrix,beta_threshold=0.001)

# 2. Apply variance filtering to retain top N most variable CpGs
mval_matrix = variance_filter(mval_matrix, top_n=top_n_cpgs) 

################################################################################
# CREATE SURVIVAL OBJECT
################################################################################

y = Surv.from_dataframe("RFi_event", "RFi_years", clinical_data)
X = mval_matrix

################################################################################
# DEFINE GRID FOR HYPERPARAMETER TUNING
################################################################################

# tune alpha
# fit a Coxnet model to estimate reasonable alpha values for grid search
initial_pipe = make_pipeline(
    CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.1, n_alphas=30)
)

# Suppress convergence warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FitFailedWarning)

# Fit on full data just to estimate alphas
initial_pipe.fit(X, y)
estimated_alphas = initial_pipe.named_steps["coxnetsurvivalanalysis"].alphas_

# Set up full parameter grid for tuning
# Define l1_ratios and use alpha values from the initial fit
param_grid = {
    "coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]#,
    #"coxnetsurvivalanalysis__l1_ratio": [0.1, 0.5, 0.9]  # Elastic Net mixing
}

################################################################################
# SET UP NESTED CV
################################################################################

# Outer CV for performance estimation
#outer_cv = KFold(n_splits=outer_cv_folds, shuffle=True, random_state=21) #10

# Inner CV for hyperparameter tuning
inner_cv = KFold(n_splits=inner_cv_folds, shuffle=True, random_state=12) #5

# Define the model and wrap in GridSearchCV (for the inner loop)
inner_model = GridSearchCV(
    estimator=make_pipeline(CoxnetSurvivalAnalysis(l1_ratio=0.9)),
    param_grid=param_grid,
    cv=inner_cv,
    #scoring=cindex_ipcw_scorer,
    error_score=0.5,
    n_jobs=-1
)

################################################################################
# RUN NESTED CV AND SAVE BEST MODEL PER OUTER FOLD - EVENT STRATIFIED OUTER SPLIT
################################################################################

# Replace outer_cv with StratifiedKFold on the event labels
outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=21)
# Use the event column from clinical_data (0/1) to stratify
event_labels = clinical_data["RFi_event"].values  # or y["RFi_event"]

outer_models = []

# for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X)): # way with simple outer cv
for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X, event_labels)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    n_events_in_fold = sum(y_train['RFi_event'])
    print(f"Fold {fold_num} has {n_events_in_fold} events.", flush=True)

    try:
        inner_model.fit(X_train, y_train)
        best_model = inner_model.best_estimator_
    
        # Refit best model with fit_baseline_model=True for later eval
        best_alpha = best_model.named_steps["coxnetsurvivalanalysis"].alphas_[0]
        best_l1_ratio = best_model.named_steps["coxnetsurvivalanalysis"].l1_ratio
        refit_best_model = make_pipeline(
            CoxnetSurvivalAnalysis(
                alphas=[best_alpha],
                l1_ratio=best_l1_ratio,
                fit_baseline_model=True,
                max_iter=100000
            )
        )
        refit_best_model.fit(X_train, y_train)

        outer_models.append({
            "fold": fold_num,
            "model": refit_best_model,  # Save the refitted model
            "test_idx": test_idx,
            "train_idx": train_idx,
            "cv_results": inner_model.cv_results_,
            "error": None
        })

    except ArithmeticError as e:
        print(f"Skipping fold {fold_num} due to numerical error: {e}",flush=True)
        outer_models.append({
            "fold": fold_num,
            "model": None,
            "test_idx": test_idx,
            "train_idx": train_idx,
            "cv_results": None,
            "error": str(e)
        })

################################################################################
# SAVE OUTER CV MODELS
################################################################################

# Save outer_models list with model objects, train/test idx, etc.
joblib.dump(outer_models, outfile_1)
print(f"Saved refitted best model per outer fold  to: {outfile_1}",flush=True)

end_time = time.time()  # Record end time
print(f"Script ended at: {time.ctime(end_time)}")
print(f"Script executed in {end_time - start_time:.2f} seconds.")