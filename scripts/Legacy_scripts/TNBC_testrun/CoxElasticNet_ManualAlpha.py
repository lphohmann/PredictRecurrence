#!/usr/bin/env python3
################################################################################
# Script: Cox proportional-hazards ElasticNet model based on patient DNA methylation and outcome data, manual alpha
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

################################################################################
# create Survival Object
################################################################################

y = Surv.from_dataframe("RFIbin", "RFI", clinical_data_df)

X = beta_matrix_df

################################################################################
# Fitting a penalized Cox model: Elastic Net
# apporach from: https://scikit-survival.readthedocs.io/en/stable/user_guide/coxnet.html
################################################################################

# Manually set alpha (must be in a list)
alpha_value = 0.1
coxnet_model = make_pipeline(
    StandardScaler(),
    CoxnetSurvivalAnalysis(l1_ratio=0.9, alphas=[alpha_value], max_iter=100000)
)

# Fit model
coxnet_model.fit(X, y)

################################################################################
# Access the trained model
################################################################################

# Extract the CoxnetSurvivalAnalysis step
best_model = coxnet_model.named_steps["coxnetsurvivalanalysis"]

best_coefs = pd.DataFrame(best_model.coef_, index=X.columns, columns=["coefficient"])

non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
print(f"Number of non-zero coefficients: {non_zero}")

non_zero_coefs = best_coefs.query("coefficient != 0")
coef_order = non_zero_coefs.abs().sort_values("coefficient").index

_, ax = plt.subplots(figsize=(6, 8))
non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
ax.set_xlabel("coefficient")
ax.grid(True)
plt.savefig("output/manual_model.png", dpi=300, bbox_inches="tight")

################################################################################
# Best model Save Outputs: Trained Model, Non-zero Coefficients, CV Results
################################################################################

# 1. Save pipeline
joblib.dump(coxnet_model, "output/manual_cox_pipeline.pkl")
print("Saved full pipeline to: output/manual_cox_pipeline.pkl")

# 2. Save non-zero coefficients
non_zero_coefs.to_csv("output/manual_non_zero_coefs.csv")
print("Saved non-zero coefficients to: output/manual_non_zero_coefs.csv")

end_time = time.time()  # Record end time
print(f"Script ended at: {time.ctime(end_time)}")
print(f"Script executed in {end_time - start_time:.2f} seconds.")
