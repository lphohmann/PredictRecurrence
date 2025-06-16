#!/usr/bin/env python3

################################################################################
# Script: calculate risk score with trained model in TCGA 
# Author: Lennart Hohmann
# Date: 29.04.2025
################################################################################

# import
import os
import sys
import pandas as pd
import numpy as np
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
from src.utils import beta2m
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

infile_0 = "./data/raw/TCGA_TNBC_MergedAnnotations.csv"
infile_1 = "./data/raw/TCGA_TNBC_betaAdj.csv"

# clinical dat
clinical = pd.read_csv(infile_0)
clinical.rename(columns={clinical.columns[0]: "Sample"}, inplace=True)
clinical = clinical.loc[:,["Sample","PFI","PFIbin","OS","OSbin"]]
clinical.head()

# full pipeline (which includes StandardScaler + CoxnetSurvivalAnalysis)
pipeline = joblib.load("./output/CoxNet_unadjusted/best_outer_fold.pkl")
pipeline = pipeline['model']


# methyl dat
methyl = pd.read_csv(infile_1, index_col=0).T
methyl = methyl.loc[clinical["Sample"]]
methyl.shape

methyl.head()

# checking if there are any missing values in methyl data
methyl.isnull()

# also check if all outomce dat is there for outcome smaples (OS)
clinical["OSbin"].value_counts()
clinical["PFIbin"].value_counts()

clinical.shape

################################################################################
# approcah 1 calc risk scores usign dot product
################################################################################

# # 3. Load the 108 non-zero coefficients
# non_zero_coefs = pd.read_csv("output/CoxNet_200k_simpleCV5/non_zero_coefs.csv", index_col=0)
# features_108 = non_zero_coefs.index.tolist()
# coefs_108 = non_zero_coefs["coefficient"].values

# # 4. Prepare your test data (e.g., TCGA samples)
# methyl_108 = methyl.loc[:, methyl.columns.intersection(features_108)]  # keep only 108 features
# methyl_108 = beta2m(methyl_108, beta_threshold=0.001)

# missing_features = [feat for feat in features_108 if feat not in methyl_108.columns]
# print(f"Number of missing features in test data: {len(missing_features)}")

# # 5. Reindex to match order and fill missing with 0 if needed
# X_test_108 = methyl_108.reindex(columns=features_108).fillna(0)

# # 6. Manually scale using the trained data parameters
# scaler = joblib.load("output/CoxNet_200k_simpleCV5/scaler_108_features.pkl")
# X_test_scaled = scaler.transform(X_test_108)

# # 7. Ensure the feature order matches between the model and the test data
# assert list(X_test_108.columns) == features_108, "Feature order mismatch!"
# assert len(coefs_108) == X_test_scaled.shape[1], "Coefficient and feature count mismatch!"

# # 8. Predict risk scores using the model directly
# risk_scores = np.dot(X_test_scaled, coefs_108)


################################################################################
# apporach 2 calc risk scores usign full pipeline and .predict()
################################################################################

print(pipeline.named_steps)  # see the steps

# Load train features (columns only)
train_methyl = pd.read_csv("./output/CoxNet_200k_simpleCV5/input_filtered_trainMethyl.csv", index_col=0)
all_features = train_methyl.columns.tolist()

# Load test methylation data
test_methyl = pd.read_csv(infile_1, index_col=0).T # methyl
test_methyl= test_methyl.loc[clinical["Sample"]]
test_methyl = beta2m(test_methyl, beta_threshold=0.001)
test_methyl.shape

# Keep only features in training data
test_methyl = test_methyl.loc[:, test_methyl.columns.intersection(all_features)]

# Find features missing in test but present in training
missing_feats = [f for f in all_features if f not in test_methyl.columns]

# Add missing features with zero values
missing_df = pd.DataFrame(0, index=test_methyl.index, columns=missing_feats)
test_methyl = pd.concat([test_methyl, missing_df], axis=1)

# Reorder columns to exact training feature order
test_methyl = test_methyl[all_features]

print(test_methyl.shape)  # Should be (number of test samples, 200000)

# Now pass test_methyl to pipeline.predict()
risk_scores = pipeline.predict(test_methyl)

####

# Assume risk_scores is a numpy array
# test_methyl.index contains the sample IDs in the same order as risk_scores
risk_df = pd.DataFrame({'Sample': test_methyl.index, 'risk_score': risk_scores})

# Merge on Sample ID
clinical_with_risk = clinical.merge(risk_df, on='Sample', how='left')

print(clinical_with_risk.head())

clinical_with_risk.to_csv("output/CoxNet_200k_simpleCV5/TCGA_risk_scores.csv")