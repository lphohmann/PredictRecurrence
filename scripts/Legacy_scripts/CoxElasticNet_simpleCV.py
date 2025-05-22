#!/usr/bin/env python3
################################################################################
# Script: Cox proportional-hazards ElasticNet model based on patient DNA methylation and outcome data
# Author: Lennart Hohmann
# Date: 11.04.2025
################################################################################

# import
import os
import sys
import pandas as pd
import numpy as np
import importlib
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
#sys.path.append("C:\Users\lhohmann\PredictRecurrence\src")
import src.utils
importlib.reload(src.utils)
from src.utils import beta2m, m2beta, create_surv, variance_filter, unicox_filter, preprocess, train_cox_lasso
import matplotlib.pyplot as plt
from sksurv.util import Surv
import warnings
from sklearn.exceptions import FitFailedWarning
from sklearn import set_config
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.datasets import load_breast_cancer
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
import joblib
import time

#set_config(display="text")  # displays text representation of estimators

# set wd
os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))
#os.chdir(os.path.expanduser("C:\Users\lhohmann\PredictRecurrence"))

start_time = time.time()  # Record start time

print(f"Script started at: {time.ctime(start_time)}")

################################################################################
# load data
################################################################################

# input paths
infile_0 = "./data/train/train_subcohorts/TNBC_train_ids.csv" 
infile_1 = "./data/train/train_methylation_adjusted.csv"
infile_2 = "./data/train/train_clinical.csv"

# output paths
outfile_model = "output/best_cox_model.pkl"
outfile_coefs = "output/non_zero_coefs.csv"
outfile_cv = "output/cv_results.csv"
outfile_plot_alpha = "output/concordance_vs_alpha.png"
outfile_plot_model = "output/best_model.png"

################################################################################
# format data
################################################################################

# sample training set
train_ids = pd.read_csv(infile_0, header=None).iloc[:, 0].tolist()

# load clin data
clinical_data = pd.read_csv(infile_2)
clinical_data = clinical_data.set_index("Sample")
clinical_data = clinical_data.loc[train_ids]
clinical_data.shape

# load beta values
beta_matrix = pd.read_csv(infile_1,index_col=0).T
# align dataframes
beta_matrix = beta_matrix.loc[train_ids]
beta_matrix.shape
beta_matrix.iloc[1:5,1:5]

################################################################################
# preprocessing steps
################################################################################

# 1. Convert beta values to M-values with a threshold 
beta_matrix_df = pd.DataFrame(beta_matrix)
mval_matrix = beta2m(beta_matrix,beta_threshold=0.001)

# 2. Apply variance filtering to retain top N most variable CpGs
mval_matrix = variance_filter(mval_matrix, top_n=200000)
mval_matrix.shape

################################################################################
# create Survival Object
################################################################################

y = Surv.from_dataframe("RFi_event", "RFi_years", clinical_data)

X = mval_matrix

################################################################################
# Fitting a penalized Cox model: Elastic Net
# apporach from: https://scikit-survival.readthedocs.io/en/stable/user_guide/coxnet.html
################################################################################

################################################################################
# finding the optimal alpha using cross-validation
################################################################################

# tune both alpha and l1_ratio
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.exceptions import FitFailedWarning

# ----------------------------------------------
# Step 1: Estimate alpha path from initial model
# ----------------------------------------------

# Fit a Coxnet model to extract reasonable alpha values for grid search
# Note: Only need one value of l1_ratio here to get the alpha path
initial_pipe = make_pipeline(
    StandardScaler(),
    CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01, n_alphas=30, max_iter=1000)
)

# Suppress convergence warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FitFailedWarning)

# Fit on full data just to estimate alphas (NOT for model evaluation!)
initial_pipe.fit(X, y)
estimated_alphas = initial_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
estimated_alphas.shape
# ----------------------------------------------
# Step 2: Set up full parameter grid for tuning
# ----------------------------------------------

# Define l1_ratios and use alpha values from the initial fit
param_grid = {
    "coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]#,
    #"coxnetsurvivalanalysis__l1_ratio": [0.1, 0.5, 0.9]  # Elastic Net mixing
}

# ----------------------------------------------
# Step 3: Set up nested cross-validation
# ----------------------------------------------

# Outer CV for performance estimation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Inner CV for hyperparameter tuning
inner_cv = KFold(n_splits=5, shuffle=True, random_state=1)

# Define the model and wrap in GridSearchCV (for the inner loop)
inner_model = GridSearchCV(
    estimator=make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9)),
    param_grid=param_grid,
    cv=inner_cv,
    error_score=0.5,
    n_jobs=-1
)

# ----------------------------------------------
# Step 4: Run nested CV to evaluate performance
# ----------------------------------------------

# Uses the inner model (which includes its own CV) for model selection in each fold
nested_scores = cross_val_score(inner_model, X, y, cv=outer_cv, scoring="concordance_index")

print(f"Nested CV Concordance Index: {np.mean(nested_scores):.3f} ± {np.std(nested_scores):.3f}")








# # Build a pipeline: standardize features and fit Coxnet model with elastic net penalty
# coxnet_pipe = make_pipeline(
#     StandardScaler(),
#     CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=100)
# )
# # Suppress warnings from failed model fits during CV
# warnings.simplefilter("ignore", UserWarning)
# warnings.simplefilter("ignore", FitFailedWarning)
# # Fit model once to compute a range of alpha values (regularization strengths)
# coxnet_pipe.fit(X, y)
# # Extract the alpha path computed during initial fit
# estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
# # Set up cross-validation strategy
# cv = KFold(n_splits=2, shuffle=True, random_state=0)
# # Perform grid search over the estimated alphas using cross-validation
# gcv = GridSearchCV(
#     make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9)),
#     param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in map(float, estimated_alphas)]},
#     cv=cv,
#     error_score=0.5,
#     n_jobs=-1,
# ).fit(X, y)
# # Collect cross-validation results
# cv_results = pd.DataFrame(gcv.cv_results_)



# visualize mean concordance index + SD for each alpha
alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
mean = cv_results.mean_test_score
std = cv_results.std_test_score

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(alphas, mean)
ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
ax.set_xscale("log")
ax.set_ylabel("concordance index")
ax.set_xlabel("alpha")
ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
ax.axhline(0.5, color="grey", linestyle="--")
ax.grid(True)

plt.savefig(outfile_plot_alpha, dpi=300, bbox_inches="tight")

################################################################################
# Best model
################################################################################

best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
best_coefs = pd.DataFrame(best_model.coef_, index=X.columns, columns=["coefficient"])

non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
print(f"Number of non-zero coefficients: {non_zero}")

non_zero_coefs = best_coefs.query("coefficient != 0")
coef_order = non_zero_coefs.abs().sort_values("coefficient").index

_, ax = plt.subplots(figsize=(6, 8))
non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
ax.set_xlabel("coefficient")
ax.grid(True)
plt.savefig(outfile_plot_model, dpi=300, bbox_inches="tight")

################################################################################
# Best model Save Outputs: Trained Model, Non-zero Coefficients, CV Results
################################################################################

# 1. Save the trained Cox model (best estimator)
joblib.dump(gcv.best_estimator_, outfile_model)
print(f"Saved best model to: {outfile_model}")

# 2. Save non-zero coefficients
non_zero_coefs.to_csv(outfile_coefs)
print(f"Saved non-zero coefficients to: {outfile_coefs}")

# 3. Save full cross-validation results
cv_results.to_csv(outfile_cv, index=False)
print(f"Saved CV results to: {outfile_cv}")

end_time = time.time()  # Record end time
print(f"Script ended at: {time.ctime(end_time)}")
print(f"Script executed in {end_time - start_time:.2f} seconds.")