#!/usr/bin/env python

################################################################################
# Script: RSF Survival Pipeline
# Author: lennart hohmann 
################################################################################

################################################################################
# SET UP
################################################################################

import os, sys, time
import numpy as np
import pandas as pd
import joblib
from sksurv.util import Surv
import argparse
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import randint

# Add project src directory to path for imports (adjust as needed)
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
from src.utils import log, load_training_data, beta2m, apply_admin_censoring, summarize_outer_models, summarize_performance,select_best_model, evaluate_outer_models, variance_filter, subset_methylation,univariate_cox_filter
from src.plotting_functions import plot_brier_scores, plot_auc_curves
from src.rsf_functions import run_nested_cv_rsf

# Set working directory
os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))

# ==============================================================================
# MAPPINGS OF INPUT FILES
# ==============================================================================

# Mapping of methylation types to their respective file paths
METHYLATION_DATA_PATHS = {
    "adjusted": "./data/train/train_methylation_adjusted.csv",
    "unadjusted": "./data/train/train_methylation_unadjusted.csv"
}

# Mapping of cohort names to their training IDs file paths
COHORT_TRAIN_IDS_PATHS = {
    "TNBC": "./data/train/train_subcohorts/TNBC_train_ids.csv",
    "ERpHER2n": "./data/train/train_subcohorts/ERpHER2n_train_ids.csv",
    "All": "./data/train/train_subcohorts/All_train_ids.csv"
}

# Common clinical data file path
INFILE_CLINICAL = "./data/train/train_clinical.csv"

# ==============================================================================
# COMMAND LINE ARGUMENTS
# ==============================================================================

parser = argparse.ArgumentParser(description="Train RSF.")
# cohort
parser.add_argument("--cohort_name", type=str, required=True,
                    choices=COHORT_TRAIN_IDS_PATHS.keys(), 
                    help="Name of the cohort to process")
# trainign features
parser.add_argument("--data_mode", type=str, 
                    choices=["clinical", "methylation", "combined"], required=True,
                    help="Which data to use: clinical only, methylation only, or both")
# methylation data type
parser.add_argument("--methylation_type", type=str, 
                    choices=METHYLATION_DATA_PATHS.keys(), 
                    default="unadjusted",
                    help="Type of methylation data")
# prefilter cpg input
parser.add_argument("--train_cpgs", type=str, 
                    default="./data/set_definitions/CpG_prefiltered_sets/cpg_ids_atac_overlap.txt",
                    help="Set of CpGs for training")
# output dir
parser.add_argument("--output_base_dir", type=str, default="./output/RSF",
                    help="Base output directory")
args = parser.parse_args()

if args.data_mode == "clinical":
    print("Note: Methylation type not considered when using clinical-only data mode.")

# ==============================================================================
# OUTPUT DIRECTORY
# ==============================================================================

# Cohort-specific output directories
current_output_dir = os.path.join(
    args.output_base_dir,
    args.cohort_name,
    args.data_mode.capitalize())

# Subfolder name
if args.data_mode == "clinical":
    subtype_folder = "None"  # keep folder depth consistent
else:
    subtype_folder = args.methylation_type.capitalize()  # could be Unadjusted/Adjusted etc.

current_output_dir = os.path.join(current_output_dir, subtype_folder)
os.makedirs(current_output_dir, exist_ok=True)

# ==============================================================================
# INPUT AND OUTPUT FILES
# ==============================================================================

# input files
infile_train_ids = COHORT_TRAIN_IDS_PATHS[args.cohort_name]
infile_betavalues = METHYLATION_DATA_PATHS[args.methylation_type] 
infile_clinical = INFILE_CLINICAL
infile_cpg_ids = args.train_cpgs

# output files
outfile_outermodels = os.path.join(current_output_dir, "outer_cv_models.pkl")
outfile_brierplot = os.path.join(current_output_dir, "brier_scores.png")
outfile_aucplot = os.path.join(current_output_dir, "auc_curves.png")
outfile_bestfold = os.path.join(current_output_dir, "best_outer_fold.pkl")
outfile_performance = os.path.join(current_output_dir, "outer_cv_performance.pkl")

# Logfile directly in the output directory
logfile_path = os.path.join(current_output_dir, "pipeline_run.log")
logfile = open(logfile_path, "w")
sys.stdout = logfile
sys.stderr = logfile

# ==============================================================================
# PARAMS
# ==============================================================================

# Data preprocessing parameters
INNER_CV_FOLDS = 5
OUTER_CV_FOLDS = 10

if args.cohort_name == "TNBC":
    # ensure censoring cutoff > max evaluation time!
    ADMIN_CENSORING_CUTOFF = 5.5
    EVAL_TIME_GRID = np.arange(2, 5.1, 1)  # time points for metrics
else:
    ADMIN_CENSORING_CUTOFF = None
    EVAL_TIME_GRID = np.arange(2, 9.1, 1)  # time points for metrics

if args.data_mode in ["clinical", "combined"]:
    CLINVARS_INCLUDED = ["Age", "Size.mm", "NHG", "LN"]
    CLIN_CATEGORICAL = ["NHG", "LN"]
else:
    CLINVARS_INCLUDED = None
    CLIN_CATEGORICAL = None

if args.data_mode in ["methylation", "combined"]:
    VARIANCE_PREFILTER = 20000
    FILTER_KEEP_N = 5000
else:
    VARIANCE_PREFILTER = 0
    FILTER_KEEP_N = 0 # no methlyation data included

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

start_time = time.time()
log(f"Pipeline started at: {time.ctime(start_time)}")
log(f"Processing Cohort: {args.cohort_name}, Methylation Type: {args.methylation_type}")
log(f"Train IDs file: {COHORT_TRAIN_IDS_PATHS[args.cohort_name]}")
log(f"Methylation data file: {METHYLATION_DATA_PATHS[args.methylation_type]}")
log(f"Clinical data file: {INFILE_CLINICAL}")
log(f"Filtered CpG set data file: {infile_cpg_ids}")
log(f"Output directory: {current_output_dir}")
log(f"Training features mode: {args.data_mode}")

# Load and preprocess data
train_ids = pd.read_csv(infile_train_ids, header=None).iloc[:, 0].tolist()
beta_matrix, clinical_data = load_training_data(train_ids, infile_betavalues, infile_clinical)

# convert to M-values
mvals = beta2m(beta_matrix, beta_threshold=0.001)

# apply censoring at 5 years for tnbc only
if ADMIN_CENSORING_CUTOFF is not None: 
    clinical_data = apply_admin_censoring(clinical_data, "RFi_years", "RFi_event", time_cutoff=ADMIN_CENSORING_CUTOFF, inplace=False)

# Subset to only include prefiltered CpGs if infile is provided
if infile_cpg_ids is not None:
    mvals = subset_methylation(mvals,infile_cpg_ids)

X = mvals.copy()

# add clincial vars to X to put all trainign featuer sin one object
if CLINVARS_INCLUDED is not None:
    # subset clinical data aligned to X
    clin = clinical_data[CLINVARS_INCLUDED].loc[X.index]
    # one-hot encode the categorical clinical variables
    encoder = OneHotEncoder(drop=None, dtype=float, sparse_output=False)
    encoded = encoder.fit_transform(clin[CLIN_CATEGORICAL])
    encoded_cols = encoder.get_feature_names_out(CLIN_CATEGORICAL).tolist()
    # make a DataFrame for the encoded columns
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=X.index)
    # build the encoded clinical DataFrame (drop original categorical cols)
    clin_encoded = pd.concat([clin.drop(columns=CLIN_CATEGORICAL), encoded_df], axis=1)
    # concatenate encoded clinical back into X
    X = pd.concat([X, clin_encoded], axis=1).copy()

    # build clinvars_included_encoded: replace original categorical names with encoded column names
    clinvars_included_encoded = [c for c in CLINVARS_INCLUDED if c not in CLIN_CATEGORICAL] + encoded_cols
    log(f"Added {clinvars_included_encoded} clinical variables. New X shape: {X.shape}")
else:
    clinvars_included_encoded = None
    encoded_cols = None
    log("No clinical variables added (CLINVARS_INCLUDED=None).")

# outcome-agnostic variance prefilter
#selected_cpgs = variance_filter(X, top_n=VARIANCE_PREFILTER,keep_vars=clinvars_included_encoded)
#X = X[selected_cpgs].copy()
#log(f"Applied variance prefilter. New X shape: {X.shape}")

# Prepare survival labels (Surv object with event & time)
y = Surv.from_dataframe("RFi_event", "RFi_years", clinical_data)
    
log(f"dont_filter_vars: {clinvars_included_encoded}")
log(f"dont_scale_vars: {encoded_cols}")

# set filter func
filter_func_1 = lambda X, y=None, **kwargs: variance_filter(X, y=y, top_n=VARIANCE_PREFILTER, **kwargs)
filter_func_2 = lambda X, y=None, **kwargs: univariate_cox_filter(X, y=y, top_n=FILTER_KEEP_N, **kwargs)

param_grid = {
    
    # Critical Expansion: Best value (800) was max, so sample up to 1499
    "estimator__randomsurvivalforest__n_estimators": randint(200, 1500), 
    
    # Expanded List: Includes proven 'sqrt' and tests wider fraction bounds
    "estimator__randomsurvivalforest__max_features": [
        0.01, 0.02, 0.05, 0.1, 0.15, 
        "sqrt"
    ],

    # Expanded List: Includes deeper constraint (20), as 12/None were frequently selected
    "estimator__randomsurvivalforest__max_depth": [None, 8, 12, 16, 20], 

    # Broad Continuous Sampling
    "estimator__randomsurvivalforest__min_samples_split": randint(5, 30), 
    
    # Broad Continuous Sampling
    "estimator__randomsurvivalforest__min_samples_leaf": randint(3, 20), 

    "estimator__randomsurvivalforest__bootstrap": [True],
}
print(f"\nDefined parameter grid:\n{param_grid}\n", flush=True)

# Run nested cross-validation
outer_models = run_nested_cv_rsf(X, y,
                             param_grid=param_grid, 
                             outer_cv_folds=OUTER_CV_FOLDS, 
                             inner_cv_folds=INNER_CV_FOLDS, 
                             #top_n_variance = FILTER_KEEP_N, 
                             filter_func_1=filter_func_1,
                             filter_func_2=filter_func_2,
                             dont_filter_vars=clinvars_included_encoded,
                             dont_scale_vars=encoded_cols,
                             output_fold_ids_file=os.path.join(current_output_dir, "cvfold_ids.pkl"))

joblib.dump(outer_models, outfile_outermodels)
log(f"Saved outer CV models to: {outfile_outermodels}")

# evaluate performance
model_performances = evaluate_outer_models(outer_models, X, y, EVAL_TIME_GRID)
joblib.dump(model_performances, outfile_performance)
print(f"Saved model performances to: {outfile_performance}")

# Plot performance metrics
folds = [p["fold"] for p in model_performances]
brier_array = np.array([p["brier_t"] for p in model_performances])
ibs_array = np.array([p["ibs"] for p in model_performances])
plot_brier_scores(brier_array, ibs_array, folds, EVAL_TIME_GRID, outfile_brierplot)
plot_auc_curves(model_performances, EVAL_TIME_GRID, outfile_aucplot)
summarize_outer_models(outer_models)
summarize_performance(model_performances)

# Select and save the best model (by chosen metric)
metric = "mean_auc"  # could be "ibs" or "auc_at_5y"
best_outer_fold = select_best_model(model_performances, outer_models, metric)
if best_outer_fold:
    joblib.dump(best_outer_fold, outfile_bestfold)
    log(f"Best model (fold {best_outer_fold['fold']}) saved to: {outfile_bestfold}")

end_time = time.time()
log(f"Pipeline ended at: {time.ctime(end_time)}")
log(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes.")
logfile.close()