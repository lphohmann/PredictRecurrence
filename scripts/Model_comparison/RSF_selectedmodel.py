#!/usr/bin/env python

################################################################################
# Script: Validate model on external dataset
# Author: Lennart Hohmann
################################################################################

################################################################################
# IMPORTS
################################################################################

import os, sys, time
import numpy as np
import pandas as pd
import joblib
from sksurv.util import Surv
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

# Add project src directory to path for imports (adjust as needed)
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
from src.utils import log, beta2m

# Set working directory
os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))

################################################################################
# INPUT FILES & PARAMS
################################################################################

#INFILE_IDS_EXTERNAL = ""
INFILE_METHYLATION_EXTERNAL = "./data/raw/TCGA_n645_unadjustedBeta.csv"
INFILE_CLINICAL_EXTERNAL = "./data/raw/TCGA_TNBC_MergedAnnotations.csv"

INFILE_MODEL = "./output/RSF/TNBC/Unadjusted/Coxlasso/best_outer_fold.pkl" # GBM RSF
INFILE_CPG_SET = "./data/set_definitions/CpG_prefiltered_sets/TNBC/Unadjusted/cox_lasso_selected_cpgs.txt"

MODEL_TYPE = "randomsurvivalforest" #"gradientboostingsurvivalanalysis" #"randomsurvivalforest"
# has to match selected_model.named_steps.get(here)

RISKSCORE_CUTOFF = 7 #1 # calc medain in train set 
OUTCOME_VARS = ["PFS", "PFSbin"]#["OS", "OSbin"] # in clinical file columns

################################################################################
# OUTPUT FILES
################################################################################

# output directory
current_output_dir = os.path.join(
    "output/Model_validation/",
    MODEL_TYPE
)
os.makedirs(current_output_dir, exist_ok=True)

# log file
path_logfile = os.path.join(current_output_dir, "validation_run.log")
logfile = open(path_logfile, "w")
sys.stdout = logfile
sys.stderr = logfile

################################################################################
# MAIN CODE
################################################################################

start_time = time.time()
log(f"Validation pipeline started at: {time.ctime(start_time)}")
log(f"Model file: {INFILE_MODEL}")
log(f"Ext. methylation data file: {INFILE_METHYLATION_EXTERNAL}")
log(f"Ext. clinical data file: {INFILE_CLINICAL_EXTERNAL}")
log(f"Filtered CpG set data file: {INFILE_CPG_SET}")
print(f"Cutoff for patient sratification: {RISKSCORE_CUTOFF}")
log(f"Output directory: {current_output_dir}")

# Load 
#ids = pd.read_csv(INFILE_IDS_EXTERNAL, header=None).iloc[:, 0].tolist()

# load clin data
clinical_data = pd.read_csv(INFILE_CLINICAL_EXTERNAL)
clinical_data.rename(columns={clinical_data.columns[0]: "Sample"}, inplace=True)
clinical_data = clinical_data[clinical_data['TNBC'] == True]
clinical_data = clinical_data.loc[:,["Sample","PFI","PFIbin","OS","OSbin"]]
clinical_data = clinical_data.set_index("Sample")

#clinical_data = clinical_data.loc[ids]
# load beta values
beta_matrix = pd.read_csv(INFILE_METHYLATION_EXTERNAL,index_col=0).T

# align dataframes
beta_matrix = beta_matrix.loc[clinical_data.index]
mvals = beta2m(beta_matrix, beta_threshold=0.001)

log("Loaded methylation and clinical data.")

# subset to only include prefiltered cpgs that were used int he trianing of the model
with open(INFILE_CPG_SET, 'r') as f:
    model_cpg_ids = [line.strip() for line in f if line.strip()]
    log(f"Successfully loaded {len(model_cpg_ids)} CpG IDs from the model's feature set.")

    # Identify which CpGs from the model are in the new dataset
    available_cpgs = [cpg for cpg in model_cpg_ids if cpg in mvals.columns]
    
    # Identify the missing CpGs
    missing_cpgs = [cpg for cpg in model_cpg_ids if cpg not in mvals.columns]

    if not missing_cpgs:
        log("All CpGs from the model's feature set are present in the external dataset.")
        # Subset the external data to match the model's features
        X = mvals[model_cpg_ids]
    else:
        log(f"Warning: {len(missing_cpgs)} of the {len(model_cpg_ids)} CpGs from the model's feature set are not in the external dataset.")
        log(f"Missing CpGs: {', '.join(missing_cpgs)}")
        
        if not available_cpgs:
            log("Error: No CpGs from the model's feature set were found in the external dataset. Cannot proceed.")
            raise ValueError("No matching CpGs to proceed with.")

        # Create a new DataFrame with only the available CpGs
        X = mvals[available_cpgs].copy()
        log(f"Proceeding with the {len(available_cpgs)} available CpGs and imputing the missing ones.")
        
        # Add back missing CpGs and impute with the calculated M-value
        for cpg in missing_cpgs:
            X[cpg] = np.log2(0.5 / (1 - 0.5)) # is 0

        # Reorder the columns to match the original model's feature order
        X = X[model_cpg_ids]

    log(f"Final data matrix for validation (X) has shape: {X.shape}")

# Prepare survival labels (Surv object with event & time)
y = Surv.from_dataframe(OUTCOME_VARS[1], OUTCOME_VARS[0], clinical_data)

# load model
selected_fold = joblib.load(INFILE_MODEL)
selected_model = selected_fold['model']  # scorer object
pipeline = selected_model.estimator      # unwrap the pipeline

print(pipeline.named_steps)

# Access the RandomSurvivalForest estimator from the pipeline
model = pipeline.named_steps[MODEL_TYPE] # your RSF inside

#print(type(selected_model))
#print(dir(selected_model))

log("Predicting risk scores on the external dataset.")
risk_scores = model.predict(X)
clinical_data["risk_score"] = risk_scores

# Plot Histogram of Risk Scores
log("Plotting Histogram of Risk Scores!")

plt.figure(figsize=(8, 6))
plt.hist(risk_scores, bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram of Risk Scores")
plt.xlabel("Risk Score (Cumulative Hazard)")
plt.ylabel("Number of Patients")
plt.grid(True)
plt.savefig(os.path.join(current_output_dir, "Hist_riskscores.png"), dpi=300)  
plt.close()


################################################################################
# Kaplan-Meier Plot
################################################################################


log("Plotting Kaplan-Meier curves for high/low risk groups defined by cutoff")

cutoff_name = f"fixed_{RISKSCORE_CUTOFF}"
clinical_data["risk_group"] = ["high" if x > RISKSCORE_CUTOFF else "low" for x in risk_scores]

summary = clinical_data.groupby("risk_group")[OUTCOME_VARS[1]].value_counts().unstack(fill_value=0)
print(summary)

mask_low = clinical_data["risk_group"] == "low"
mask_high = clinical_data["risk_group"] == "high"

kmf_low = KaplanMeierFitter()
kmf_high = KaplanMeierFitter()

# Fit KM curves
kmf_low.fit(
    clinical_data.loc[mask_low, OUTCOME_VARS[0]],
    clinical_data.loc[mask_low, OUTCOME_VARS[1]],
    label="Low risk"
)
kmf_high.fit(
    clinical_data.loc[mask_high, OUTCOME_VARS[0]],
    clinical_data.loc[mask_high, OUTCOME_VARS[1]],
    label="High risk"
)

# Plot curves
fig, ax = plt.subplots(figsize=(8, 6))
kmf_low.plot(ax=ax, ci_show=True)
kmf_high.plot(ax=ax, ci_show=True)

add_at_risk_counts(kmf_low, kmf_high, ax=ax)
ax.set_xlabel("Time (years)")
ax.set_ylabel("Survival Probability")

# Log-rank test
results = logrank_test(
    clinical_data.loc[mask_low, OUTCOME_VARS[0]],
    clinical_data.loc[mask_high, OUTCOME_VARS[0]],
    event_observed_A=clinical_data.loc[mask_low, OUTCOME_VARS[1]],
    event_observed_B=clinical_data.loc[mask_high, OUTCOME_VARS[1]]
)
p_value = results.p_value

plt.title(f"Kaplan-Meier: High vs. Low Risk Groups\nLog-rank p-value = {p_value:.4f}\nCutoff method: {cutoff_name}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(current_output_dir, f"KM_risk_groups_cutoff_{cutoff_name}.png"), dpi=300)
plt.close()

################################################################################
# Univariate Cox regression with risk score
################################################################################
log("Calculating univariate Cox regression and plotting hazard ratio for risk score!")

df_lifelines = clinical_data[[OUTCOME_VARS[1], OUTCOME_VARS[0], "risk_score"]].copy()
df_lifelines.columns = ["event", "time", "risk_score"]

cph = CoxPHFitter()
cph.fit(df_lifelines, duration_col="time", event_col="event")

print("\nUnivariate Cox regression (risk score):")
cph.print_summary()

print("\nEvent value counts:")
print(clinical_data[OUTCOME_VARS[1]].value_counts())

ax = cph.plot(hazard_ratios=True)
plt.tight_layout()
plt.savefig(os.path.join(current_output_dir, "cox_forest_plot.png"), dpi=300, bbox_inches="tight")
plt.close()

log(f"Script finished at: {time.ctime(time.time())}")
log(f"Total runtime: {(time.time() - start_time):.2f} seconds.")
logfile.close()