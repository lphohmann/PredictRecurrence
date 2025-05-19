#!/usr/bin/env python3
################################################################################
# Script: make prediction with model in TCGA and plot survival curves
# Author: Lennart Hohmann
# Date: 29.04.2025
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
# load and preprocess data
################################################################################

# load model to make predicitons with
# Load pipeline
loaded_pipeline = joblib.load("./output/CoxNet_manual/manual_cox_pipeline.pkl")

# input paths
infile_1 = "./data/raw/TCGA_TNBC_betaAdj.csv"
infile_2 = "./data/raw/TCGA_TNBC_MergedAnnotations.csv"

# rows are patient IDs and columns are features (CpGs)
clinical_data_df = pd.read_csv(infile_2)
clinical_data_df = clinical_data_df.set_index("PD_ID")
# Drop rows with NaN in 'RFIbin' or 'RFI' columns
clinical_data_df = clinical_data_df.dropna(subset=['RFIbin', 'RFI'])
beta_matrix_df = pd.read_csv(infile_1,index_col=0).T


# filter
beta_matrix_df = variance_filter(beta_matrix_df, 150000) # 200000
X = beta_matrix_df

################################################################################
# Predictions: Risk Score, Survival Function, Cumulative Hazard Function
################################################################################

# 1. risk score 
risk_scores = loaded_pipeline.predict(X)  # X must be same features, same format
#risk_scores = model.predict(X)
# Save risk scores to CSV
risk_scores_df = pd.DataFrame(risk_scores, columns=["risk_score"], index=X.index)
risk_scores_df.head()
risk_scores_df.to_csv("output/risk_scores_from_loaded_model.csv")
print("Saved patient risk scores to: output/risk_scores_from_loaded_model.csv")

# need re-fit the model with fit_baseline_model enabled for survival or cumulative hazard function
#coxnet_pred = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9, #fit_baseline_model=True))
#coxnet_pred.set_params(**gcv.best_params_)
#coxnet_pred.fit(X, y)

# 2. Survival Function

# survival probability curve for each patient
#surv_funcs = coxnet_pred.named_steps["coxnetsurvivalanalysis"].predict_survival_function(X)

# 3. Cumulative Hazard Function

# shows how the risk accumulates over time
#cumhaz_funcs = coxnet_pred.named_steps["coxnetsurvivalanalysis"].predict_cumulative_hazard_function(X)


