#!/usr/bin/env python

################################################################################
# Script: Exploring best model 
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score, brier_score

sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
import src.utils
importlib.reload(src.utils)
from src.utils import beta2m, variance_filter

# set wd
os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))
os.makedirs("output", exist_ok=True)

start_time = time.time()  # Record start time

print(f"Script started at: {time.ctime(start_time)}",flush=True)

################################################################################
# PARAMS
################################################################################

top_n_cpgs = 200000 # has to match n used in training

################################################################################
# SET FILE PATHS
################################################################################

# input paths
infile_0 = r"./data/train/train_subcohorts/TNBC_train_ids.csv" # subcohort ids
infile_1 = r"./data/train/train_methylation_adjusted.csv"
infile_2 = r"./data/train/train_clinical.csv"
infile_3 = r"./output/CoxNet/best_model.pkl"
infile_4 = r"./output/CoxNet/outer_cv_models.pkl"

# output paths

################################################################################
# LOAD AND SUBSET TRAINING DATA
################################################################################

# load best model
best_model = joblib.load(infile_3)

# load corresponsing outfer fold 2 data
outer_models = joblib.load(infile_4)
fold2_testidx = outer_models[2]["test_idx"].tolist()

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
mval_matrix = variance_filter(mval_matrix, top_n=top_n_cpgs) #200,000

################################################################################
# CREATE SURVIVAL OBJECT
################################################################################

#y = Surv.from_dataframe("RFi_event", "RFi_years", clinical_data)
X = mval_matrix
X_test = X.iloc[fold2_testidx,:].copy()
X_test.shape
clin_test = clinical_data.iloc[fold2_testidx,:].copy()
clin_test.shape
################################################################################
# Define consistent evaluation time grid based on test sets across outer folds
################################################################################

from sksurv.linear_model import CoxnetSurvivalAnalysis
from lifelines import KaplanMeierFitter, CoxPHFitter
import seaborn as sns

# ------------------------------------------------------------------------------
# 1. Inspect model hyperparameters and coefficients
# ------------------------------------------------------------------------------

# Unpack estimator from pipeline
coxnet = best_model.named_steps["coxnetsurvivalanalysis"]

# Now access alpha and coefficients
best_alpha_used = coxnet.alphas_[0]  # or simply: coxnet.alphas_[0]
print("Refit model was trained with alpha =", best_alpha_used)
print("L1 ratio:", coxnet.l1_ratio)

# Get non-zero coefficients
coefs = coxnet.coef_.flatten()
nonzero_mask = coefs != 0
nonzero_features = X.columns[nonzero_mask]
nonzero_features.shape
print(f"Number of non-zero coefficients: {np.sum(nonzero_mask)}")

# ------------------------------------------------------------------------------
# 2. Compute risk scores on training set
# ------------------------------------------------------------------------------

risk_scores = best_model.predict(X_test)
risk_scores = pd.Series(risk_scores, index=X_test.index)

# ------------------------------------------------------------------------------
# 3. Stratify into high/low risk using median
# ------------------------------------------------------------------------------

clin_test["risk_score"] = risk_scores

# median cutoff
clin_test["risk_group"] = pd.qcut(risk_scores, q=2, labels=["low", "high"])
summary = clin_test.groupby("risk_group")["RFi_event"].value_counts().unstack()
print(summary)

# # tertile cutoff
# low_cutoff = risk_scores.quantile(1/3)
# high_cutoff = risk_scores.quantile(2/3)
# clinical_data["risk_group"] = pd.cut(
#     risk_scores,
#     bins=[-np.inf, low_cutoff, high_cutoff, np.inf],
#     labels=["low", "medium", "high"]
# )
# summary = clinical_data.groupby("risk_group")["RFi_event"].value_counts().unstack()
# print(summary)

# Basic matplotlib histogram
plt.figure(figsize=(8, 6))
plt.hist(risk_scores, bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram of Risk Scores")
plt.xlabel("Risk Score")
plt.ylabel("Number of Patients")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 4. Kaplan-Meier Plot
# ------------------------------------------------------------------------------

kmf = KaplanMeierFitter()

plt.figure(figsize=(8,6))
for group in ["low", "high"]:
    mask = clinical_data["risk_group"] == group
    kmf.fit(clinical_data.loc[mask, "RFi_years"], clinical_data.loc[mask, "RFi_event"], label=group)
    kmf.plot(ci_show=True)

plt.title("Kaplan-Meier: High vs. Low Risk Groups")
plt.xlabel("Time (years)")
plt.ylabel("Survival Probability")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("./output/CoxNet/KM_risk_groups.png", dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# 5. Univariate Cox regression with risk score
# ------------------------------------------------------------------------------

# Prepare dataframe for lifelines
df_lifelines = clin_test[["RFi_event", "RFi_years", "risk_score"]].copy()
df_lifelines.columns = ["event", "time", "risk_score"]  # lifelines naming

cph = CoxPHFitter()
cph.fit(df_lifelines, duration_col="time", event_col="event")

print("\nUnivariate Cox regression (risk score):")
cph.print_summary()