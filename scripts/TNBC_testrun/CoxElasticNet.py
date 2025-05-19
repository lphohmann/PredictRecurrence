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
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
from src.utils import beta2m, m2beta, create_surv, variance_filter, unicox_filter, preprocess, train_cox_lasso
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

start_time = time.time()  # Record start time

print(f"Script started at: {time.ctime(start_time)}")

################################################################################
# load data
################################################################################

# input paths
infile_1 = "./data/raw/tnbc235.csv" # replace with PC dat later
infile_2 = "./data/raw/tnbc_anno.csv" # replace with tnbc dat

# output paths
#outfile_1 = 

################################################################################
# format data
################################################################################

# rows are patient IDs and columns are features (CpGs)
clinical_data_df = pd.read_csv(infile_2)
clinical_data_df = clinical_data_df.set_index("PD_ID")
# Drop rows with NaN in 'RFIbin' or 'RFI' columns
clinical_data_df = clinical_data_df.dropna(subset=['RFIbin', 'RFI'])
clinical_data_df.shape
beta_matrix_df = pd.read_csv(infile_1,index_col=0).T

# align dataframes by shared samples
shared_samples = clinical_data_df.index.intersection(beta_matrix_df.index)
print(f"Shared samples: {len(shared_samples)}")
clinical_data_df = clinical_data_df.loc[shared_samples]
beta_matrix_df = beta_matrix_df.loc[shared_samples]

################################################################################
# preprocessing steps
################################################################################

beta_matrix_df = variance_filter(beta_matrix_df, 150000) # 200000
beta_matrix_df.shape

# imputing missing data
#from sklearn.impute import SimpleImputer
# missing data per sample/CpG
#missing_per_sample = beta_matrix_df.isna().sum(axis=0)
#print(missing_per_sample.describe())
#missing_per_cpg = beta_matrix_df.isna().sum(axis=0)
#print(missing_per_cpg.describe())

#imputer = SimpleImputer(strategy="mean")  # or median
#X_imputed = imputer.fit_transform(X)

################################################################################
# create Survival Object
################################################################################

y = Surv.from_dataframe("RFIbin", "RFI", clinical_data_df)

X = beta_matrix_df

################################################################################
# Fitting a penalized Cox model: Elastic Net
# apporach from: https://scikit-survival.readthedocs.io/en/stable/user_guide/coxnet.html
################################################################################

################################################################################
# finding the optimal alpha using cross-validation
################################################################################
# determine the set of alpha values to evaluate by fiting a penalized Cox model to the whole data and retrieving the estimated set of alphas
# scaling the data
coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=100))
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FitFailedWarning)
coxnet_pipe.fit(X, y)

# perform n=3 cross-validation to estimate the performance in terms of concordance index for each alpha
estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
cv = KFold(n_splits=3, shuffle=True, random_state=0)
gcv = GridSearchCV(
    make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9)),
    param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in map(float, estimated_alphas)]},
    cv=cv,
    error_score=0.5,
    n_jobs=-1,
).fit(X, y)

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

plt.savefig("output/concordance_vs_alpha.png", dpi=300, bbox_inches="tight")

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
plt.savefig("output/best_model.png", dpi=300, bbox_inches="tight")

################################################################################
# Best model Save Outputs: Trained Model, Non-zero Coefficients, CV Results
################################################################################

# 1. Save the trained Cox model (best estimator)
joblib.dump(gcv.best_estimator_, "output/best_cox_model.pkl")
print("Saved best model to: output/best_cox_model.pkl")

# 2. Save non-zero coefficients
non_zero_coefs.to_csv("output/non_zero_coefs.csv")
print("Saved non-zero coefficients to: output/non_zero_coefs.csv")

# 3. Save full cross-validation results
cv_results.to_csv("output/cv_results.csv", index=False)
print("Saved CV results to: output/cv_results.csv")

end_time = time.time()  # Record end time
print(f"Script ended at: {time.ctime(end_time)}")
print(f"Script executed in {end_time - start_time:.2f} seconds.")