#!/usr/bin/env python

################################################################################
# Script: CoxNet selected model - TCGA validation
# Author: Lennart Hohmann
################################################################################

################################################################################
# SET UP
################################################################################

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
#from sksurv.util import Surv
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

# Import custom CoxNet functions
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
from src.utils import (
    log,
    load_training_data,
    preprocess_data,
    beta2m
)

# Set working directory
os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))

################################################################################
# INPUT FILES
################################################################################

# Input files
infile_train_ids = "./data/train/train_subcohorts/TNBC_train_ids.csv" # sample ids of training cohort
infile_betavalues = "./data/train/train_methylation_adjusted.csv" # adjusted/unadjusted
infile_clinical = "./data/train/train_clinical.csv"
infile_outerfold = "./output/CoxNet_adjusted/best_outer_fold.pkl"#------------------------ADAPT

infile_tcga_clinical = "./data/raw/TCGA_TNBC_MergedAnnotations.csv"
infile_tcga_betavalues = "./data/raw/TCGA_TNBC_betaAdj.csv"#------------------------ADAPT TCGA_TNBC_betaAdj.csv

################################################################################
# PARAMS
################################################################################

# Output directory and files
output_dir = "output/CoxNet_adjusted/Selected_model/"#-------------------------------------ADAPT
os.makedirs(output_dir, exist_ok=True)
#outfile_brierplot = os.path.join(output_dir, "brier_scores.png")

# log file
path_logfile = os.path.join(output_dir, "tcga_run.log")
logfile = open(path_logfile, "w")
sys.stdout = logfile
sys.stderr = logfile

# Parameters
top_n_cpgs = 200000

# cutoffs to stratify by
median_cutoff = -0.0387
predictiveness_cutoff = 0.1988
cutoff_list = [median_cutoff, predictiveness_cutoff]

################################################################################
# load data
################################################################################

start_time = time.time()
log(f"Script started at: {time.ctime(start_time)}")

# Load and prepare data
selected_fold = joblib.load(infile_outerfold)
selected_model = selected_fold['model']
log("Loaded selected outer fold!")

train_ids = pd.read_csv(infile_train_ids, header=None).iloc[:, 0].tolist()
log("Loaded training ids!")

beta_matrix, clinical_data = load_training_data(train_ids, infile_betavalues, infile_clinical)
log("Loaded training data!")

X_train = preprocess_data(beta_matrix, top_n_cpgs=top_n_cpgs)
log("Finished preprocessing of training data!")

#tcga dat

tcga_clinical_data = pd.read_csv(infile_tcga_clinical)
tcga_clinical_data.rename(columns={tcga_clinical_data.columns[0]: "Sample"}, inplace=True)
tcga_clinical_data = tcga_clinical_data.loc[:,["Sample","PFI","PFIbin","OS","OSbin"]]
tcga_clinical_data = tcga_clinical_data.set_index("Sample")
clin_test = tcga_clinical_data
tcga_betavalues_data = pd.read_csv(infile_tcga_betavalues,index_col=0).T
tcga_betavalues_data = tcga_betavalues_data.loc[clin_test.index]
X_test = beta2m(tcga_betavalues_data, beta_threshold=0.001)
log("Finished loading TCGA data!")


################################################################################
# inspect model hyperparameters and coefficients
################################################################################

# Unpack estimator from pipeline
coxnet = selected_model.named_steps["coxnetsurvivalanalysis"]
# Get non-zero coefficients
coefs = coxnet.coef_.flatten()
nonzero_mask = coefs != 0
nonzero_features = X_train.columns[nonzero_mask]
print(f"Number of non-zero coefficients in the model: {np.sum(nonzero_mask)}")
coefs_df = pd.DataFrame(coefs, index=X_train.columns, columns=["coefficient"])
non_zero_coefs = coefs_df[coefs_df["coefficient"] != 0]
print(f"Number of model coefficients being in the TCGA dataset: {np.sum(nonzero_mask)}")

num_present = len(X_test.columns.intersection(nonzero_features.to_list()))
print(f"{num_present} out of {len(nonzero_features)} non-zero features are present in X_test.")

################################################################################
# prep TCGA data
################################################################################

# Load train features (columns only)
train_features = X_train.columns.tolist()

# Keep only features in training data
X_test = X_test.loc[:, X_test.columns.intersection(train_features)]
X_test.shape

# Find features missing in test but present in training
missing_feats = [f for f in train_features if f not in X_test.columns]
len(missing_feats)

# Add missing features with zero values
missing_df = pd.DataFrame(0, index=X_test.index, columns=missing_feats)
X_test = pd.concat([X_test, missing_df], axis=1)

# Reorder columns to exact training feature order
X_test = X_test[train_features]

print(X_test.shape)  # Should be (number of test samples, 200000)

################################################################################
# Compute risk scores in TCGA
################################################################################

# Now pass X_test to pipeline.predict()
risk_scores = selected_model.predict(X_test)
risk_scores = pd.Series(risk_scores, index=X_test.index)
clin_test["risk_score"] = risk_scores

log("Plotting Histogram of Risk Scores!")
# Basic matplotlib histogram
plt.figure(figsize=(8, 6))
plt.hist(risk_scores, bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram of Risk Scores")
plt.xlabel("Risk Score")
plt.ylabel("Number of Patients")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "TCGA_Hist_riskscores.png"), dpi=300)  
plt.close()

################################################################################
# Kaplan-Meier Plot
################################################################################

log("Plotting Kaplan-Meier curves for high/low risk groups defined by median and predictiveness cutoffs!")

for cutoff in cutoff_list:

    print(f"\nCutoff = {cutoff:.4f}\n",flush=True)

    # Use the median cutoff from training to assign risk groups in test set
    clin_test["risk_group"] = pd.Series(["high" if x > cutoff else "low" for x in risk_scores], index=risk_scores.index)

    summary = clin_test.groupby("risk_group")["OSbin"].value_counts().unstack()
    print(summary)

    # Initialize KM fitters
    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    # Masks
    mask_low = clin_test["risk_group"] == "low"
    mask_high = clin_test["risk_group"] == "high"

    # Fit
    kmf_low.fit(clin_test.loc[mask_low, "OS"], clin_test.loc[mask_low, "OSbin"], label="Low risk")
    kmf_high.fit(clin_test.loc[mask_high, "OS"], clin_test.loc[mask_high, "OSbin"], label="High risk")

    # plot
    # Allocate space for KM plot + risk table
    fig, ax = plt.subplots(figsize=(8, 6))

    kmf_low.plot(ax=ax, ci_show=True)
    kmf_high.plot(ax=ax, ci_show=True)

    # Add risk table
    add_at_risk_counts(kmf_low, kmf_high, ax=ax,rows_to_show= ["At risk"])
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Survival Probability")

    results = logrank_test(
        clin_test.loc[mask_low, "OS"],
        clin_test.loc[mask_high, "OS"],
        event_observed_A=clin_test.loc[mask_low, "OSbin"],
        event_observed_B=clin_test.loc[mask_high, "OSbin"]
    )

    p_value = results.p_value

    plt.title(f"Kaplan-Meier: High vs. Low Risk Groups\nLog-rank p-value = {p_value:.4f}")
    plt.xlabel("Time (years)")
    plt.ylabel("Survival Probability")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"TCGA_KM_risk_groups_{cutoff:.4f}.png"), dpi=300) 
    plt.close()

################################################################################
# 5. Univariate Cox regression with risk score
################################################################################

log("Calculating univariate Cox regression and plotting hazard ratio for risk score!")

# Prepare dataframe for lifelines
df_lifelines = clin_test[["OSbin", "OS", "risk_score"]].copy()
df_lifelines.columns = ["event", "time", "risk_score"]  # lifelines naming

# Fit univariate Cox model
cph = CoxPHFitter()
cph.fit(df_lifelines, duration_col="time", event_col="event")

# Print summary
print("\nUnivariate Cox regression (risk score):")
cph.print_summary()

# Optional: check event distribution
print("\nEvent value counts:")
print(clin_test["OSbin"].value_counts())

# Generate the forest plot
ax = cph.plot(hazard_ratios=True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "TCGA_cox_forest_plot.png"), dpi=300, bbox_inches="tight")  
plt.close()

################################################################################
# THE END
################################################################################

end_time = time.time()  # Record end time
log(f"Script ended at: {time.ctime(end_time)}")
log(f"Script executed in {end_time - start_time:.2f} seconds.")
logfile.close()