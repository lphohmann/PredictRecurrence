#!/usr/bin/env python3
################################################################################
# Script: make prediction with model
# Author: Lennart Hohmann
# Date: 21.04.2025
################################################################################

# import
import os
import sys
import pandas as pd
import numpy as np
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
from src.utils import beta2m, variance_filter
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

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
loaded_pipeline = joblib.load("./output/CoxNet_200k_simpleCV5/best_cox_model.pkl")


cox_model = loaded_pipeline.named_steps["coxnetsurvivalanalysis"]
print("Alpha used in best model:", cox_model.alphas_[0])
non_zero = (cox_model.coef_ != 0).sum()
print("Number of non-zero coefficients:", non_zero)
feature_names = loaded_pipeline.named_steps["standardscaler"].get_feature_names_out()
selected_features = feature_names[cox_model.coef_.flatten() != 0]
print(selected_features)


#loaded_pipeline = joblib.load("./output/manual_cox_pipeline.pkl")

# input paths
infile_0 = r"./data/train/train_subcohorts/TNBC_train_ids.csv" 

infile_1 = r"./data/train/train_methylation_adjusted.csv"
infile_2 = r"./data/train/train_clinical.csv"


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
#beta_matrix.iloc[1:5,1:5]
# 1. Convert beta values to M-values with a threshold 
mval_matrix = beta2m(beta_matrix,beta_threshold=0.001)

# 2. Apply variance filtering to retain top N most variable CpGs
mval_matrix = variance_filter(mval_matrix, top_n=200000)
mval_matrix.shape

X = mval_matrix

# save scaler for test set
# Select only the 108 features with non-zero coefficients
non_zero_coefs = pd.read_csv("./output/CoxNet_200k_simpleCV5/non_zero_coefs.csv", index_col=0)
features_108 = non_zero_coefs.index.tolist()
X_for_scaler = mval_matrix[features_108]
X_for_scaler.shape
# Fit a StandardScaler on these features
scaler = StandardScaler().fit(X_for_scaler)

# Save the scaler
joblib.dump(scaler, "output/CoxNet_200k_simpleCV5/scaler_108_features.pkl")

################################################################################
# Predictions: Risk Score, Survival Function, Cumulative Hazard Function
################################################################################

# 1. risk score 
risk_scores = loaded_pipeline.predict(X)  # X must be same features, same format
#risk_scores = model.predict(X)
# Save risk scores to CSV
risk_scores_df = pd.DataFrame(risk_scores, columns=["risk_score"], index=X.index)
risk_scores_df.head()
risk_scores_df.to_csv("output/CoxNet_200k_simpleCV5/risk_scores_from_loaded_model.csv")
risk_scores_df.to_csv("output/CoxNet_200k_simpleCV5/SCANB_risk_scores.csv")
print("Saved patient risk scores to: output/CoxNet_200k_simpleCV5/risk_scores_from_loaded_model.csv")

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


