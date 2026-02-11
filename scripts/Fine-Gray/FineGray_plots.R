#!/usr/bin/env Rscript

################################################################################
# Fine-Gray Competing Risks with Dual Final Models
# Author: Lennart Hohmann

################################################################################
# LIBRARIES
################################################################################

library(readr)        # Fast CSV reading
library(dplyr)        # Data manipulation
library(survival)     # Survival analysis (Surv objects, concordance)
library(glmnet)       # Elastic net Cox regression
library(cmprsk)
library(caret)        # Cross-validation fold creation
library(data.table)   # Fast data reading with fread
library(riskRegression) # Score() function for competing risks metrics
library(prodlim) #prodlim::Hist()
library(devtools)
library(fastcmprsk)

setwd("~/PhD_Workspace/PredictRecurrence/")
source("./src/finegray_functions.R")
source("./src/finegray_plotting_functions.R")

################################################################################
# COMMAND LINE ARGUMENTS
################################################################################

args <- commandArgs(trailingOnly = TRUE)

# Defaults
DEFAULT_COHORT <- "ERpHER2n"
DEFAULT_TRAIN_CPGS <- "./data/set_definitions/CpG_prefiltered_sets/cpg_ids_atac_overlap.txt"
DEFAULT_OUTPUT_DIR <- "./output/FineGray"
OUTPUT_BASE_DIR <- DEFAULT_OUTPUT_DIR
COHORT_NAME <- DEFAULT_COHORT
################################################################################
# INPUT OUTPUT SETTINGS
################################################################################

# Input files
INFILE_METHYLATION <- "./data/train/train_methylation_unadjusted.csv"
COHORT_TRAIN_IDS_PATHS <- list(
  TNBC = "./data/train/train_subcohorts/TNBC_train_ids.csv",
  ERpHER2n = "./data/train/train_subcohorts/ERpHER2n_train_ids.csv",
  All = "./data/train/train_subcohorts/All_train_ids.csv"
)
INFILE_CLINICAL <- "./data/train/train_clinical.csv"

# Output directory - simplified since we don't have DATA_MODE
current_output_dir <- file.path(
  OUTPUT_BASE_DIR, 
  COHORT_NAME, 
  "DualMode",  # New folder name to distinguish from old pipeline
  "Unadjusted"
)
dir.create(current_output_dir, recursive = TRUE, showWarnings = FALSE)

plot_dir <- file.path(
  current_output_dir,
  "Figures"
)
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

################################################################################
# LOAD DATA
################################################################################

load(file.path(current_output_dir, 
               "outer_fold_results.RData"))
#str(outer_fold_results)

# only include valid folds (where not skipped)
outer_fold_results <- Filter(function(f) {
  !is.null(f) && !isTRUE(f$metadata$skipped)
}, outer_fold_results)

perf_results_mrs <- aggregate_cv_performance_threefg(
  outer_fold_results,
  model_name = "FGR_MRS")

perf_results_clin <- aggregate_cv_performance_threefg(
  outer_fold_results,
  model_name = "FGR_CLIN")

perf_results_combined <- aggregate_cv_performance_threefg(
  outer_fold_results,
  model_name = "FGR_COMBINED")

aggregated_perf <- list(FGR_MRS = perf_results_mrs,
                        FGR_CLIN= perf_results_clin,
                        FGR_COMBINED= perf_results_combined)

stability_results <- assess_finegray_stability_threefg(
  outer_fold_results
)


# plots
plot_coxnet_stability_heatmap(plot_dir=plot_dir,
                              outer_fold_results,
                              min_freq = 0.4)

# echk where the coefs acutalyl omce from ehre? whic olf
# this plots mean coefs which is not what i want
plot_coxnet_coefficients(plot_dir=plot_dir,
                         outer_fold_results,
                         min_freq = 0.4)

stability_metrics <- stability_results$stability_metrics


#plot_penalized_fg_coefficients(plot_dir=plot_dir,stability_results)



load(file = file.path(current_output_dir, "final_fg_results.RData"))

final_results$variable_importance$fgr_MRS
final_results$variable_importance$fgr_CLIN
final_results$variable_importance$fgr_COMBINED


plot_penalized_fg_coefficients(
  plot_dir=plot_dir,
  cpg_coefs=final_results$mrs_construction$cpg_coefs,
  outer_fold_results = outer_fold_results)

plot_fg_forest_hr(
  plot_dir=plot_dir,
  vimp_fg = final_results$variable_importance$fgr_COMBINED)
