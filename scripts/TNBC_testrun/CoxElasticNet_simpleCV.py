#!/usr/bin/env python3
################################################################################
# Script: Cox proportional-hazards ElasticNet model based on patient DNA methylation and outcome data
# Author: Lennart Hohmann
# Date: 11.04.2025
################################################################################

# Standard library imports
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.exceptions import FitFailedWarning

#sys.path.append(r"/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
sys.path.append("C:\\Users\\lhohmann\\PredictRecurrence")
sys.path.append("C:\\Users\\lhohmann\\PredictRecurrence\\src")
import src.utils
importlib.reload(src.utils)
from src.utils import beta2m, variance_filter, cindex_scorer

# set wd
#os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))
os.chdir(os.path.expanduser("C:\\Users\\lhohmann\\PredictRecurrence"))
os.makedirs("output", exist_ok=True)
start_time = time.time()  # Record start time

print(f"Script started at: {time.ctime(start_time)}")

################################################################################
# load data
################################################################################

# input paths
infile_0 = r"./data/train/train_subcohorts/TNBC_train_ids.csv" 
infile_1 = r"./data/train/train_methylation_adjusted.csv"
infile_2 = r"./data/train/train_clinical.csv"

# output paths
outfile_model = r"output/best_cox_model.pkl"
outfile_coefs = r"output/non_zero_coefs.csv"
outfile_cv = r"output/cv_results.csv"
outfile_plot_alpha = r"output/concordance_vs_alpha.png"
outfile_plot_model = r"output/best_model.png"

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

# Build a pipeline: standardize features and fit Coxnet model with elastic net penalty
coxnet_pipe = make_pipeline(
    StandardScaler(),
    CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, n_alphas=30)
)
# Suppress warnings from failed model fits during CV
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FitFailedWarning)

# Fit model once to compute a range of alpha values (regularization strengths)
coxnet_pipe.fit(X, y)

# Extract the alpha path computed during initial fit
estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_

# Set up cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=0)

# Perform grid search over the estimated alphas using cross-validation
gcv = GridSearchCV(
    make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9)),
    param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in map(float, estimated_alphas)]},
    cv=cv,
    error_score=0.5,
    n_jobs=-1,
).fit(X, y)

# Collect cross-validation results
cv_results = pd.DataFrame(gcv.cv_results_)

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