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
    preprocess_data
)

# Set working directory
os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))

################################################################################
# INPUT FILES
################################################################################

# Input files
infile_train_ids = "./data/train/train_subcohorts/TNBC_train_ids.csv" # sample ids of training cohort
infile_betavalues = "./data/train/train_methylation_unadjusted.csv" # adjusted/unadjusted
infile_clinical = "./data/train/train_clinical.csv"
infile_outerfold = "./output/CoxNet_unadjusted/best_outer_fold.pkl"

infile_0 = "./data/raw/TCGA_TNBC_MergedAnnotations.csv"
infile_1 = "./data/raw/TCGA_TNBC_betaAdj.csv"
"./data/raw/TC"
################################################################################
# PARAMS
################################################################################

# Output directory and files
output_dir = "output/CoxNet_unadjusted/Selected_model/"
os.makedirs(output_dir, exist_ok=True)
#outfile_brierplot = os.path.join(output_dir, "brier_scores.png")

# log file
path_logfile = os.path.join(output_dir, "selectedmodel_run.log")
logfile = open(path_logfile, "w")
sys.stdout = logfile
sys.stderr = logfile

# Parameters
top_n_cpgs = 200000

################################################################################
# MAIN CODE
################################################################################

start_time = time.time()
log(f"Script started at: {time.ctime(start_time)}")

# Load and prepare data
selected_fold = joblib.load(infile_outerfold)
#print(selected_fold.keys())
selected_model = selected_fold['model']
bm_testidx = selected_fold["test_idx"].tolist() 
bm_trainidx = selected_fold["train_idx"].tolist()
log("Loaded selected outer fold!")

train_ids = pd.read_csv(infile_train_ids, header=None).iloc[:, 0].tolist()
log("Loaded training ids!")

beta_matrix, clinical_data = load_training_data(train_ids, infile_betavalues, infile_clinical)
log("Loaded training data!")

X = preprocess_data(beta_matrix, top_n_cpgs=top_n_cpgs)
log("Finished preprocessing of training data!")

#y = Surv.from_dataframe("RFi_event", "RFi_years", clinical_data)

################################################################################
# calc median follow up time
################################################################################
log("Calculating median follow up time!")

clin_mf = clinical_data.copy()
#clinical_data['RFi_years'].median()
clin_mf['reverse_event'] = 1 - clin_mf['RFi_event']
kmf = KaplanMeierFitter()
kmf.fit(durations=clin_mf['RFi_years'], event_observed=clin_mf['reverse_event'])
median_followup = kmf.median_survival_time_
print(f"Median follow-up time (Reverse KM): {median_followup:.2f} years")

# survival analyses
#y = Surv.from_dataframe("RFi_event", "RFi_years", clinical_data)

################################################################################
# define clin test and train data of that outer fold
################################################################################
log("Defining clinical test and train data of selected outer fold!")

X_test = X.iloc[bm_testidx,:].copy()
clin_test = clinical_data.iloc[bm_testidx,:].copy()

X_train = X.iloc[bm_trainidx,:].copy()
clin_train = clinical_data.iloc[bm_trainidx,:].copy()

################################################################################
# inspect model hyperparameters and coefficients
################################################################################
log("Inspecting outer foldmodel hyperparameters and coefficients!")

# Unpack estimator from pipeline
coxnet = selected_model.named_steps["coxnetsurvivalanalysis"]

# Now access alpha and coefficients
alpha_used = coxnet.alphas_[0]  # or simply: coxnet.alphas_[0]
print("alpha =", alpha_used)
print("L1 ratio:", coxnet.l1_ratio)

# Get non-zero coefficients
coefs = coxnet.coef_.flatten()
nonzero_mask = coefs != 0
nonzero_features = X.columns[nonzero_mask]
nonzero_features.shape
print(f"Number of non-zero coefficients: {np.sum(nonzero_mask)}")

# plot
coefs_df = pd.DataFrame(coefs, index=X.columns, columns=["coefficient"])
non_zero_coefs = coefs_df[coefs_df["coefficient"] != 0]
# Sort by absolute coefficient size for plotting
coef_order = non_zero_coefs["coefficient"].abs().sort_values().index
non_zero_coefs = non_zero_coefs.loc[coef_order]

fig, ax = plt.subplots(figsize=(6, 8))
non_zero_coefs.plot.barh(ax=ax, legend=False)
ax.set_xlabel("Coefficient")
ax.set_title("Non-zero CoxNet Coefficients")
ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "coxnet_nonzero_coefficients.png"), dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved coefficient plot with {len(non_zero_coefs)} non-zero features.")

################################################################################
# Compute risk scores on training set to get cutoffs
################################################################################
log("Computing risk scores on training set to get cutoffs!")

risk_scores_train = selected_model.predict(X_train)
risk_scores_train = pd.Series(risk_scores_train, index=X_train.index)
median_cutoff = risk_scores_train.median()  # define median cutoff from training risk scores
print(f"Median-based cutoff: {median_cutoff:.4f}")

################################################################################
# Compute predictiveness-based cutoff
################################################################################

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
log("Computing risk scores on test set!")

risk_scores = selected_model.predict(X_test)
risk_scores = pd.Series(risk_scores, index=X_test.index)
clin_test["risk_score"] = risk_scores

################################################################################
# 3. Stratify into high/low risk using median
################################################################################

log("Plotting Histogram of Risk Scores!")

# Basic matplotlib histogram
plt.figure(figsize=(8, 6))
plt.hist(risk_scores, bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram of Risk Scores")
plt.xlabel("Risk Score")
plt.ylabel("Number of Patients")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Hist_riskscores.png"), dpi=300)  
plt.close()

################################################################################
# 4. Kaplan-Meier Plot
################################################################################

log("Plotting Kaplan-Meier curves for high/low risk groups defined by median and predictiveness cutoffs!")

for cutoff in [median_cutoff, predictiveness_cutoff]:

    print(f"\nCutoff = {cutoff:.4f}\n",flush=True)

    # Use the median cutoff from training to assign risk groups in test set
    clin_test["risk_group"] = pd.Series(["high" if x > cutoff else "low" for x in risk_scores], index=risk_scores.index)

    summary = clin_test.groupby("risk_group")["RFi_event"].value_counts().unstack()
    print(summary)

    # Initialize KM fitters
    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    # Masks
    mask_low = clin_test["risk_group"] == "low"
    mask_high = clin_test["risk_group"] == "high"

    # Fit
    kmf_low.fit(clin_test.loc[mask_low, "RFi_years"], clin_test.loc[mask_low, "RFi_event"], label="Low risk")
    kmf_high.fit(clin_test.loc[mask_high, "RFi_years"], clin_test.loc[mask_high, "RFi_event"], label="High risk")

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
    plt.savefig(os.path.join(output_dir, f"KM_risk_groups_{cutoff:.4f}.png"), dpi=300) 
    plt.close()

################################################################################
# 5. Univariate Cox regression with risk score
################################################################################

log("Calculating univariate Cox regression and plotting hazard ratio for risk score!")

# Prepare dataframe for lifelines
df_lifelines = clin_test[["RFi_event", "RFi_years", "risk_score"]].copy()
df_lifelines.columns = ["event", "time", "risk_score"]  # lifelines naming

# Fit univariate Cox model
cph = CoxPHFitter()
cph.fit(df_lifelines, duration_col="time", event_col="event")

# Print summary
print("\nUnivariate Cox regression (risk score):")
cph.print_summary()

# Optional: check event distribution
print("\nEvent value counts:")
print(clin_test["RFi_event"].value_counts())

# Generate the forest plot
ax = cph.plot(hazard_ratios=True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cox_forest_plot.png"), dpi=300, bbox_inches="tight")  
plt.close()