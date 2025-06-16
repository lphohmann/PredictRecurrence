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
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
import src.utils
importlib.reload(src.utils)
from src.utils import beta2m, variance_filter

# set wd
os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))
os.makedirs("output", exist_ok=True)

start_time = time.time()  # Record start time

# Redirect stdout to a log file
script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
log_path = f"./output/CoxNet/{script_name}_log.txt"
sys.stdout = open(log_path, "w")

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

# output paths
# log path defined above

################################################################################
# LOAD AND SUBSET TRAINING DATA
################################################################################

# load best model
best_model_fold = joblib.load(infile_3)
print(best_model_fold.keys())
best_model = best_model_fold['model']
# load corresponsing outer cv model dat
bm_testidx = best_model_fold["test_idx"].tolist() # check which fold was best 
bm_trainidx = best_model_fold["train_idx"].tolist() # check which fold was best 

# sample training set
train_ids = pd.read_csv(infile_0, header=None).iloc[:, 0].tolist()

# load clin data
clinical_data = pd.read_csv(infile_2)
clinical_data = clinical_data.set_index("Sample")
clinical_data = clinical_data.loc[train_ids]

# calc median follow up time
clin_mf = clinical_data.copy()
#clinical_data['RFi_years'].median()
clin_mf['reverse_event'] = 1 - clin_mf['RFi_event']
kmf = KaplanMeierFitter()
kmf.fit(durations=clin_mf['RFi_years'], event_observed=clin_mf['reverse_event'])
median_followup = kmf.median_survival_time_
print(f"Median follow-up time (Reverse KM): {median_followup:.2f} years")

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
X_test = X.iloc[bm_testidx,:].copy()
clin_test = clinical_data.iloc[bm_testidx,:].copy()

X_train = X.iloc[bm_trainidx,:].copy()
clin_train = clinical_data.iloc[bm_trainidx,:].copy()

################################################################################
# 1. Inspect model hyperparameters and coefficients
################################################################################

# Unpack estimator from pipeline
coxnet = best_model.named_steps["coxnetsurvivalanalysis"]

# Now access alpha and coefficients
best_alpha_used = coxnet.alphas_[0]  # or simply: coxnet.alphas_[0]
print("alpha =", best_alpha_used)
print("L1 ratio:", coxnet.l1_ratio)

# Get non-zero coefficients
coefs = coxnet.coef_.flatten()
nonzero_mask = coefs != 0
nonzero_features = X.columns[nonzero_mask]
nonzero_features.shape
print(f"Number of non-zero coefficients: {np.sum(nonzero_mask)}")

################################################################################
# 2. Compute risk scores on training set to get cutoffs
################################################################################

risk_scores_train = best_model.predict(X_train)
risk_scores_train = pd.Series(risk_scores_train, index=X_train.index)
median_cutoff = risk_scores_train.median()  # define median cutoff from training risk scores

# Compute predictiveness-based cutoff
sorted_scores = np.sort(risk_scores_train.values)
event_rate = clin_train["RFi_event"].mean()  # prevalence of event
# Find percentile where predicted risk is closest to event rate
closest_idx = np.argmin(np.abs(sorted_scores - event_rate))
percentile = (closest_idx + 1) / len(sorted_scores) * 100
predictiveness_cutoff = np.percentile(sorted_scores, percentile)
print(f"Predictiveness-based cutoff: {predictiveness_cutoff:.4f}")

################################################################################
# 2. Compute risk scores on test set
################################################################################

risk_scores = best_model.predict(X_test)
risk_scores = pd.Series(risk_scores, index=X_test.index)

################################################################################
# 3. Stratify into high/low risk using median
################################################################################

clin_test["risk_score"] = risk_scores

# Basic matplotlib histogram
plt.figure(figsize=(8, 6))
plt.hist(risk_scores, bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram of Risk Scores")
plt.xlabel("Risk Score")
plt.ylabel("Number of Patients")
plt.grid(True)
plt.savefig(f"./output/CoxNet/Hist_riskscores.png", dpi=300)
plt.close()

################################################################################
# 4. Kaplan-Meier Plot
################################################################################

for cutoff in [median_cutoff, predictiveness_cutoff]:

    # Use the median cutoff from training to assign risk groups in test set
    clin_test["risk_group"] = pd.Series(["high" if x > cutoff else "low" for x in risk_scores], index=risk_scores.index)

    summary = clin_test.groupby("risk_group")["RFi_event"].value_counts().unstack()
    print(summary)

    # Assuming clin_test has columns "risk_group", "RFi_years", "RFi_event"
    kmf = KaplanMeierFitter()

    plt.figure(figsize=(8,6))

    # Fit and plot KM curves
    for group in ["low", "high"]:
        mask = clin_test["risk_group"] == group
        kmf.fit(clin_test.loc[mask, "RFi_years"], clin_test.loc[mask, "RFi_event"], label=group)
        kmf.plot(ci_show=True)

    # Compute log-rank test between low and high risk groups
    mask_low = clin_test["risk_group"] == "low"
    mask_high = clin_test["risk_group"] == "high"

    results = logrank_test(
        clin_test.loc[mask_low, "RFi_years"],
        clin_test.loc[mask_high, "RFi_years"],
        event_observed_A=clin_test.loc[mask_low, "RFi_event"],
        event_observed_B=clin_test.loc[mask_high, "RFi_event"]
    )

    p_value = results.p_value

    plt.title(f"Kaplan-Meier: High vs. Low Risk Groups\nLog-rank p-value = {p_value:.4f}")
    plt.xlabel("Time (years)")
    plt.ylabel("Survival Probability")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"./output/CoxNet/KM_risk_groups_{cutoff}.png", dpi=300)
    plt.close()

################################################################################
# 5. Univariate Cox regression with risk score
################################################################################

# Prepare dataframe for lifelines
df_lifelines = clin_test[["RFi_event", "RFi_years", "risk_score"]].copy()
df_lifelines.columns = ["event", "time", "risk_score"]  # lifelines naming

cph = CoxPHFitter()
cph.fit(df_lifelines, duration_col="time", event_col="event")

print("\nUnivariate Cox regression (risk score):")
cph.print_summary()

clin_test["RFi_event"].value_counts()