#!/usr/bin/env python3
################################################################################
# Script: use predictions for patient stratificaiton and survival analysis
# Author: Lennart Hohmann
# Date: 22.04.2025
################################################################################

# import
import os
import sys
import pandas as pd
import numpy as np
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
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

start_time = time.time()  # Record start time

print(f"Script started at: {time.ctime(start_time)}")

################################################################################
# load risk scores
################################################################################

# input paths
infile_1 = "./output/risk_scores_from_loaded_model.csv" 
infile_2 = "./data/raw/tnbc_anno.csv" # replace with tnbc dat

# stratify patients by risk score tertiles
risk_scores_df = pd.read_csv(infile_1)
risk_scores_df["risk_score_tert"] = pd.qcut(risk_scores_df["risk_score"], q=3, labels=False)
#risk_scores_df["risk_score_tert"].value_counts()
# Print the tertile boundaries
#tertile_boundaries = risk_scores_df['risk_score'].quantile([1/3, 2/3])
#print(tertile_boundaries) #  -0.411048 , 0.164139
# rows are patient IDs and columns are features (CpGs)
clinical_data_df = pd.read_csv(infile_2)
clinical_data_df = clinical_data_df.loc[:, ["PD_ID","RFIbin","RFI"]]
risk_scores_df.columns.values[0] = "PD_ID"

# merge 
merged_clinical_df = pd.merge(clinical_data_df, risk_scores_df, on="PD_ID", how="inner")
merged_clinical_df["RFIbin"] = merged_clinical_df["RFIbin"].astype(bool)
merged_clinical_df.head()

print(pd.crosstab(merged_clinical_df['risk_score_tert'], merged_clinical_df['RFIbin']))
merged_clinical_df['risk_score_tert'].value_counts()

# simple hist
# Plot histogram for the 'risk_score' column
plt.figure(figsize=(10, 6))
plt.hist(merged_clinical_df['risk_score'], bins=20, edgecolor='black')  # You can adjust the number of bins
plt.title('Histogram of Risk Score')
plt.xlabel('Risk Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('KM_RiskHist.png', dpi=300, bbox_inches='tight')
plt.close()
###############################################################################
# simple surv analysis
################################################################################

import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator

for group in (0,1,2):
    mask_treat = merged_clinical_df["risk_score_tert"] == group
    time_treatment, survival_prob_treatment, conf_int = kaplan_meier_estimator(
        merged_clinical_df["RFIbin"][mask_treat],
        merged_clinical_df["RFI"][mask_treat],
        conf_type="log-log",
    )

    plt.step(time_treatment, survival_prob_treatment, where="post", label=f"RiskTert = {group}")
    plt.fill_between(time_treatment, conf_int[0], conf_int[1], alpha=0.25, step="post")

plt.ylim(0, 1)
plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")
plt.savefig('KM_TertRiskStrat.png', dpi=300, bbox_inches='tight')