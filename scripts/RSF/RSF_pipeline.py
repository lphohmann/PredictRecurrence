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
from src.utils import log, load_training_data, beta2m, apply_admin_censoring, summarize_outer_models, summarize_performance,select_best_model, evaluate_outer_models, variance_filter, subset_methylation,univariate_cox_filter, aggregate_performance
from src.plotting_functions import plot_brier_scores, plot_auc_curves, plot_auc_with_sem
from src.rsf_functions import run_nested_cv_rsf, train_final_rsf_model

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
outfile_auc_semplot = os.path.join(current_output_dir, "auc_sem_curves.png")
outfile_bestfold = os.path.join(current_output_dir, "best_outer_fold.pkl")
outfile_performance = os.path.join(current_output_dir, "outer_cv_performance.pkl")
outfile_finalmodel = os.path.join(current_output_dir, "final_model.pkl")

# Logfile directly in the output directory
logfile_path = os.path.join(current_output_dir, "pipeline_run.log")
logfile = open(logfile_path, "w")
sys.stdout = logfile
sys.stderr = logfile

# ==============================================================================
# PARAMS
# ==============================================================================

# Data preprocessing parameters
INNER_CV_FOLDS = 3
OUTER_CV_FOLDS = 5

# time grid
if args.cohort_name == "TNBC":
    # ensure censoring cutoff > max evaluation time!
    ADMIN_CENSORING_CUTOFF = 5.01
    EVAL_TIME_GRID = np.array([1.0, 3.0, 5.0])
else:
    ADMIN_CENSORING_CUTOFF = None
    EVAL_TIME_GRID = np.array([1.0, 3.0, 5.0, 10.0])

# clinvars
if args.data_mode in ["clinical", "combined"]:
    CLINVARS_INCLUDED = ["Age", "Size.mm", "NHG", "LN"]
    CLIN_CATEGORICAL = ["NHG", "LN"]
else:
    CLINVARS_INCLUDED = None
    CLIN_CATEGORICAL = None

# filter top n
if args.data_mode in ["methylation", "combined"]:
    FILTER_1_N = 500#10000
    FILTER_2_N = 250#1000
else:
    FILTER_1_N = 0
    FILTER_2_N = 0 # no methlyation data included

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

# ==============================================================================

# Load and preprocess data
train_ids = pd.read_csv(infile_train_ids, header=None).iloc[:, 0].tolist()
beta_matrix, clinical_data = load_training_data(train_ids, infile_betavalues, infile_clinical)

# convert to M-values
mvals = beta2m(beta_matrix, beta_threshold=0.001)

# ==============================================================================

# apply censoring at 5 years for tnbc only
if ADMIN_CENSORING_CUTOFF is not None: 
    clinical_data = apply_admin_censoring(clinical_data, "RFi_years", "RFi_event", time_cutoff=ADMIN_CENSORING_CUTOFF, inplace=False)

# ==============================================================================

# Subset to only include prefiltered CpGs if infile is provided
if infile_cpg_ids is not None:
    mvals = subset_methylation(mvals,infile_cpg_ids)

# ==============================================================================

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

# ==============================================================================

# Prepare survival labels (Surv object with event & time)
y = Surv.from_dataframe("RFi_event", "RFi_years", clinical_data)

# ==============================================================================

log(f"dont_filter_vars: {clinvars_included_encoded}")
log(f"dont_scale_vars: {encoded_cols}")
log(f"dont_penalize_vars: {clinvars_included_encoded}")

# set filter func
filter_func_1 = lambda X, y=None, **kwargs: variance_filter(X, y=y, top_n=FILTER_1_N, exclude_top_perc=0.5, **kwargs)
filter_func_2 = lambda X, y=None, **kwargs: univariate_cox_filter(X, y=y, top_n=FILTER_2_N, **kwargs)

# ==============================================================================

param_grid = {
    # 1. Feature Subsampling (Aggressive)
    'estimator__randomsurvivalforest__max_features': ['sqrt', 0.005, 0.01, 0.05, 0.1], 
    
    # 2. Tree Depth (Deep but constrained)
    'estimator__randomsurvivalforest__max_depth': [10, 25, 50],
    
    # 3. Split Size (High minimum for robustness)
    'estimator__randomsurvivalforest__min_samples_split': [10, 30, 60, 100],
    
    # 4. Leaf Size (Very high minimum to prevent overfitting to sparse events)
    'estimator__randomsurvivalforest__min_samples_leaf': [5, 15, 30, 50],
    
    # 5. Bootstrap
    'estimator__randomsurvivalforest__bootstrap': [True]
}
print(f"\nDefined parameter grid:\n{param_grid}\n", flush=True)

# ==============================================================================

# Run nested cross-validation
outer_models = run_nested_cv_rsf(X, y,
                             param_grid=param_grid, 
                             outer_cv_folds=OUTER_CV_FOLDS, 
                             inner_cv_folds=INNER_CV_FOLDS, 
                             #top_n_variance = FILTER_KEEP_N, 
                             filter_func_1=filter_func_1,
                             filter_func_2=filter_func_2,
                             dont_filter_vars=clinvars_included_encoded,
                             dont_scale_vars=encoded_cols)

joblib.dump(outer_models, outfile_outermodels)
log(f"Saved outer CV models to: {outfile_outermodels}")

# ==============================================================================

# evaluate performance
model_performances = evaluate_outer_models(outer_models, X, y, EVAL_TIME_GRID)

# aggregate results
aggregated_results = aggregate_performance(model_performances, EVAL_TIME_GRID)

# save results for later analysis
results_to_save = {
    'performance': model_performances,     # Per-fold performance
    'aggregated': aggregated_results,  # Mean ± SEM across folds
    'time_grid': EVAL_TIME_GRID
}

joblib.dump(results_to_save, outfile_performance)
print(f"Saved model performances to: {outfile_performance}")

# Plot performance metrics
plot_auc_curves(model_performances, EVAL_TIME_GRID, outfile_aucplot)
plot_brier_scores(model_performances, EVAL_TIME_GRID, outfile_brierplot)
plot_auc_with_sem(model_performances, EVAL_TIME_GRID, outfile_auc_semplot)

# ==============================================================================

# Select and save the best model (by chosen metric)
#metric = "mean_auc"  # could be "ibs" or "auc_at_5y"
#best_outer_fold = select_best_model(model_performances, outer_models, metric)
#if best_outer_fold:
#    joblib.dump(best_outer_fold, outfile_bestfold)
#    log(f"Best model (fold {best_outer_fold['fold']}) saved to: {outfile_bestfold}")

# ==============================================================================

# Train final aggregated model on full dataset
final_model_result = train_final_rsf_model(
    X, y,
    param_grid=param_grid,
    filter_func_1=filter_func_1,
    filter_func_2=filter_func_2,
    dont_filter_vars=clinvars_included_encoded,
    dont_scale_vars=encoded_cols
)

print(f"Best CV C-index of Final model: {final_model_result['cv_results']['mean_test_score'].max():.3f}")
# Save the final model
joblib.dump(final_model_result, outfile_finalmodel)
log(f"Saved final model to: {outfile_finalmodel}")

# ==============================================================================

end_time = time.time()
log(f"Pipeline ended at: {time.ctime(end_time)}")
log(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes.")
logfile.close()