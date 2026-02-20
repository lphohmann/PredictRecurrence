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
#source("./src/utils.R")

################################################################################
# COMMAND LINE ARGUMENTS
################################################################################

args <- commandArgs(trailingOnly = TRUE)

# Defaults
DEFAULT_COHORT <- "All"
DEFAULT_TRAIN_CPGS <- "./data/set_definitions/CpG_prefiltered_sets/cpg_ids_atac_overlap.txt"
DEFAULT_OUTPUT_DIR <- "./output/FineGray"

# Parse arguments
if (length(args) == 0) {
  cat("No command line arguments provided. Using defaults.\n")
  COHORT_NAME <- DEFAULT_COHORT
  TRAIN_CPGS <- DEFAULT_TRAIN_CPGS
  OUTPUT_BASE_DIR <- DEFAULT_OUTPUT_DIR
  
} else if (length(args) == 1) {
  COHORT_NAME <- args[1]
  TRAIN_CPGS <- DEFAULT_TRAIN_CPGS
  OUTPUT_BASE_DIR <- DEFAULT_OUTPUT_DIR
  
} else {
  cat("\n=== USAGE ===\n")
  cat("Rscript FineGray_dual_pipeline.R <COHORT>\n\n")
  cat("Arguments:\n")
  cat("  COHORT      : 'TNBC', 'ERpHER2n', or 'All'\n\n")
  cat("Example:\n")
  cat("  Rscript FineGray_dual_pipeline.R ERpHER2n\n\n")
  cat("Or run without arguments to use defaults:\n")
  cat("  Rscript FineGray_dual_pipeline.R\n\n")
  stop("Incorrect number of arguments provided.")
}

# Validate inputs
if (!COHORT_NAME %in% c("TNBC", "ERpHER2n", "All")) {
  stop(sprintf("Invalid COHORT_NAME: '%s'. Must be 'TNBC', 'ERpHER2n', or 'All'", COHORT_NAME))
}

if (TRAIN_CPGS == "NULL") {
  TRAIN_CPGS <- NULL
}

# Print configuration
cat(sprintf("\n=== PIPELINE SETTINGS ===\n"))
cat(sprintf("Cohort:      %s\n", COHORT_NAME))
cat(sprintf("Mode:        Methylation CV → Dual Final Models (MRS-only & MRS+Clinical)\n"))
cat(sprintf("CpG file:    %s\n", ifelse(is.null(TRAIN_CPGS), "NULL (all CpGs)", TRAIN_CPGS)))
cat(sprintf("Output dir:  %s\n", OUTPUT_BASE_DIR))
cat(sprintf("=========================\n\n"))

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

################################################################################
# SETUP LOGGING
################################################################################

script_name <- "finegray_dual_pipeline_postCV"
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
log_filename <- sprintf("%s_%s.log", script_name, timestamp)
log_path <- file.path(current_output_dir, log_filename)

log_con <- file(log_path, open = "wt")
sink(log_con, type = "output", split = TRUE)
sink(log_con, type = "message")

cat(sprintf("%s\n", paste(rep("=", 80), collapse = "")))
cat(sprintf("FINE-GRAY DUAL MODEL PIPELINE\n"))
cat(sprintf("Started: %s\n", Sys.time()))
cat(sprintf("Cohort: %s\n", COHORT_NAME))
cat(sprintf("Log file: %s\n", log_path))
cat(sprintf("%s\n\n", paste(rep("=", 80), collapse = "")))

################################################################################
# PARAMETERS
################################################################################

# Administrative censoring
ADMIN_CENSORING_CUTOFF <- if (COHORT_NAME == "TNBC") 5.01 else NULL

# Clinical variables for final models only
# These are NOT used in CV feature selection, only in final Model 2
CLIN_CONT <- c("Age", "Size.mm")
CLIN_CATEGORICAL <- if (COHORT_NAME == "All") {
  c("NHG", "LN", "ER", "PR", "HER2")
} else {
  c("NHG", "LN")
}
CLINVARS_ALL <- c(CLIN_CONT, CLIN_CATEGORICAL)

cat(sprintf("Clinical variables (for unpen FG Model 2 only):\n"))
cat(sprintf("  Continuous: %s\n", paste(CLIN_CONT, collapse=", ")))
cat(sprintf("  Categorical: %s\n\n", paste(CLIN_CATEGORICAL, collapse=", ")))

# Stability threshold for final model
STABILITY_THRESHOLD_FG <- 0.4

# Performance evaluation timepoints (in years)
EVAL_TIMES <- seq(1, 10)
if (COHORT_NAME =="TNBC") {
  EVAL_TIMES <- EVAL_TIMES[1:5]
}

################################################################################
# DATA LOADING AND PREPROCESSING
################################################################################

start_time <- Sys.time()
cat(sprintf("\n========== LOADING DATA ==========\n"))

# Load sample IDs and data
train_ids <- read_csv(
  COHORT_TRAIN_IDS_PATHS[[COHORT_NAME]], 
  col_names = FALSE, 
  show_col_types = FALSE
)[[1]]

data_list <- load_training_data(train_ids, INFILE_METHYLATION, INFILE_CLINICAL)
beta_matrix <- data_list$beta_matrix
clinical_data <- data_list$clinical_data

################################################################################
# methylation matrix 
################################################################################

# Convert beta values to M-values
mvals <- beta_to_m(beta_matrix, beta_threshold = 0.001)

# Subset to predefined CpGs (if specified)
if (!is.null(TRAIN_CPGS)) {
  mvals <- subset_methylation(mvals, TRAIN_CPGS)
}

X <- mvals
cat(sprintf("CV feature matrix: %d samples × %d CpGs (methylation-only)\n", 
            nrow(X), ncol(X)))

################################################################################
# clinical matrix 
################################################################################

# Apply administrative censoring if needed
if (!is.null(ADMIN_CENSORING_CUTOFF)) {
  clinical_data <- apply_admin_censoring(
    clinical_data, "RFi_years", "RFi_event", ADMIN_CENSORING_CUTOFF
  )
  clinical_data <- apply_admin_censoring(
    clinical_data, "OS_years", "OS_event", ADMIN_CENSORING_CUTOFF
  )
}

clinical_data$LN <- gsub("N\\+", "Np", clinical_data$LN)
clin <- clinical_data[c(CLIN_CONT, CLIN_CATEGORICAL)]
clin <- clin[rownames(X), , drop = FALSE]
encoded_result <- onehot_encode_clinical(clin, CLIN_CATEGORICAL)

################################################################################
# COMPETING RISKS OUTCOME DEFINITION
################################################################################

cat(sprintf("\n========== DEFINING OUTCOMES ==========\n"))

# Create competing risks coding
# Event = 0: Censored
# Event = 1: Recurrence (outcome of interest)
# Event = 2: Death without recurrence (competing risk)

clinical_data$DeathNoR_event <- as.integer(
  clinical_data$OS_event == 1 & clinical_data$RFi_event == 0
)
clinical_data$DeathNoR_years <- clinical_data$OS_years

clinical_data$CompRisk_event_coded <- 0
clinical_data$CompRisk_event_coded[clinical_data$RFi_event == 1] <- 1
clinical_data$CompRisk_event_coded[clinical_data$DeathNoR_event == 1] <- 2

clinical_data$time_to_CompRisk_event <- pmin(
  clinical_data$RFi_years, 
  clinical_data$DeathNoR_years
)

# Print outcome summary
cat(sprintf("Competing risks summary:\n"))
cat(sprintf("  Censored: %d (%.1f%%)\n", 
            sum(clinical_data$CompRisk_event_coded == 0),
            100 * mean(clinical_data$CompRisk_event_coded == 0)))
cat(sprintf("  RFI events: %d (%.1f%%)\n", 
            sum(clinical_data$CompRisk_event_coded == 1),
            100 * mean(clinical_data$CompRisk_event_coded == 1)))
cat(sprintf("  Death without Recurrence: %d (%.1f%%)\n", 
            sum(clinical_data$CompRisk_event_coded == 2),
            100 * mean(clinical_data$CompRisk_event_coded == 2)))

# Create survival objects for CoxNet
y_rfi <- Surv(clinical_data$RFi_years, clinical_data$RFi_event)
y_death_no_r <- Surv(clinical_data$DeathNoR_years, clinical_data$DeathNoR_event)

################################################################################
# TRAIN FINAL MODEL ON ALL DATA
################################################################################

cat(sprintf("\n========== TRAINING FINAL MODEL ==========\n"))

X_all <- X
clinical_all <- clinical_data

cat(sprintf("Training samples: n=%d (RFi=%d, Death=%d)\n",
            nrow(X_all),
            sum(clinical_all$RFi_event),
            sum(clinical_all$DeathNoR_event)))

# ----------------------------------------------------------------------------
# Collect CV-Discovered Features
# ----------------------------------------------------------------------------
cat(sprintf("\n--- Collecting CV-Discovered Features ---\n"))

stability_metrics <- stability_results$stability_metrics

# Select stable non-clinical features
stable_cpgs <- stability_metrics$feature[
  stability_metrics$selection_freq >= STABILITY_THRESHOLD_FG &
    stability_metrics$direction_consistent == TRUE
]

cat(sprintf(
  "Including CpGs with selection_freq ≥ %.2f and consistent direction: n = %d\n",
  STABILITY_THRESHOLD_FG,
  length(stable_cpgs)
))

#--------------------------------------------------------------------------
# Fit Penalized Fine-Gray Model (MRS Construction)
# --------------------------------------------------------------------------

cat(sprintf("\n========== PENALIZED FINE-GRAY (MRS CONSTRUCTION) ==========\n"))

pFG_res <- fit_penalized_finegray_cv(
  X_input = X_all[, stable_cpgs, drop = FALSE],
  clinical_input = clinical_all,
  alpha_seq = 0,
  lambda_seq = NULL,
  cr_time = "time_to_CompRisk_event",
  cr_event = "CompRisk_event_coded",
  penalty = "RIDGE",
  n_inner_folds = 3
)

# Extract CpGs and their coefficients
pFG_selected_cpgs <- pFG_res$results_table$feature[pFG_res$results_table$selected]
pFG_cpg_coefs <- setNames(
  pFG_res$results_table$pFG_coefficient[pFG_res$results_table$selected],
  pFG_selected_cpgs
)

# Extract scaling parameters (CpGs were scaled inside pFG model)
pFG_cpg_scaling <- list(
  center = setNames(
    pFG_res$results_table$scale_center[pFG_res$results_table$selected],
    pFG_selected_cpgs
  ),
  scale = setNames(
    pFG_res$results_table$scale_scale[pFG_res$results_table$selected],
    pFG_selected_cpgs
  )
)

cat(sprintf("BEST PENALIZED FG: Alpha=%.2f, Lambda=%.6f, CpGs=%d\n",
            pFG_res$best_alpha,
            pFG_res$best_lambda,
            length(pFG_selected_cpgs)))
cat("Selected CpGs:", paste(pFG_selected_cpgs, collapse=", "), "\n")

# --------------------------------------------------------------------------
# Calculate Methylation Risk Score (MRS)
# --------------------------------------------------------------------------

# Training MRS
mrs_all <- calculate_methylation_risk_score(
  X_data = X_all,#[, stable_cpgs, drop = FALSE],
  cpg_coefficients = pFG_cpg_coefs,
  scaling_params = pFG_cpg_scaling,
  verbose = TRUE
)
MRS_all <- mrs_all$mrs

# Scale the MRS itself (optional)
mrs_all_scaled <- scale(MRS_all)
mrs_scaling <- list(
  center = attr(mrs_all_scaled, "scaled:center"),
  scale = attr(mrs_all_scaled, "scaled:scale")
)
MRS_all <- as.numeric(mrs_all_scaled)
cat("  MRS scaled to mean=0, sd=1\n")


# --------------------------------------------------------------------------
# Prepare Data for Final Unpenalized Fine-Gray Model
# --------------------------------------------------------------------------

cat(sprintf("\n--- Preparing Final Model Data ---\n"))

# Verify row alignment
if (!identical(rownames(clinical_all), rownames(X_all))) {
  stop("Row names don't match between clinical_all and X_all")
}

# Build training data
fgr_all_data <- data.frame(
  time_to_CompRisk_event = clinical_all$time_to_CompRisk_event,
  CompRisk_event_coded = clinical_all$CompRisk_event_coded,
  encoded_result$encoded_df[rownames(X_all), , drop = FALSE],
  methylation_risk_score = MRS_all,
  row.names = rownames(clinical_all)
)

# Scale clinical continuous variables
clin_all_scaled <- scale(fgr_all_data[, CLIN_CONT, drop = FALSE])
clin_scaling <- list(
  variables = CLIN_CONT,
  center = attr(clin_all_scaled, "scaled:center"),
  scale = attr(clin_all_scaled, "scaled:scale")
)

fgr_all_data[, CLIN_CONT] <- clin_all_scaled

cat(sprintf("  Final model predictors: %s\n", 
            paste(setdiff(colnames(fgr_all_data), 
                          c("time_to_CompRisk_event", "CompRisk_event_coded")), 
                  collapse=", ")))

# --------------------------------------------------------------------------
# Fit Unpenalized Fine-Gray Model (FINAL MODEL)
# --------------------------------------------------------------------------
# 
# # Fit Unpenalized Fine-Gray Model 1: only MRS
# 
# cat(sprintf("\n========== UNPENALIZED FINE-GRAY MODELS ==========\n"))
# 
# fgr_MRS <- fit_fine_gray_model(
#   fgr_data = fgr_all_data[,!colnames(fgr_all_data) %in% names(encoded_result$encoded_df)],
#   cr_time = "time_to_CompRisk_event",
#   cr_event = "CompRisk_event_coded",
#   cause = 1
# )
# 
# fgr_MRS <- fgr_MRS$model
# 
# # Fit Unpenalized Fine-Gray Model 2: only clin
# fgr_CLIN <- fit_fine_gray_model(
#   fgr_data = fgr_all_data[,colnames(fgr_all_data) != "methylation_risk_score"],
#   cr_time = "time_to_CompRisk_event",
#   cr_event = "CompRisk_event_coded",
#   cause = 1
# )
# 
# fgr_CLIN <- fgr_CLIN$model

# Fit Unpenalized Fine-Gray Model 3: both
fgr_COMBINED <- fit_fine_gray_model(
  fgr_data = fgr_all_data,
  cr_time = "time_to_CompRisk_event",
  cr_event = "CompRisk_event_coded",
  cause = 1
)

fgr_COMBINED <- fgr_COMBINED$model

# ----------------------------------------------------------------------------
# Calculate Variable Importance
# ----------------------------------------------------------------------------

cat("\n--- Fine-Gray Variable Importance ---\n")
# 
# vimp_fgr_MRS <- calculate_fgr_importance(
#   fgr_model = fgr_MRS,
#   encoded_cols = encoded_result$encoded_cols,
#   verbose = FALSE
# )
# 
# vimp_fgr_CLIN <- calculate_fgr_importance(
#   fgr_model = fgr_CLIN,
#   encoded_cols = encoded_result$encoded_cols,
#   verbose = FALSE
# )

vimp_fgr_COMBINED <- calculate_fgr_importance(
  fgr_model = fgr_COMBINED,
  encoded_cols = encoded_result$encoded_cols,
  verbose = FALSE
)

###############################################################################
# CALCULATE RISK SCORE CUTOFF FOR DICHOTOMIZATION
###############################################################################
library(cmprsk)

fgr_all_data$RFi_event <- clinical_all$RFi_event[match(rownames(fgr_all_data),clinical_all$Sample)]
fgr_all_data$RFi_years <- clinical_all$RFi_years[match(rownames(fgr_all_data),clinical_all$Sample)]

pdf(file.path(current_output_dir, "cutoff_selection_plots.pdf"), width = 7, height = 5)

selected_cutoffs <- list()
for(risk_time_cutoff in c(5,10)) {
  # apply model to whole data chohrt (same data it was trained on)
  risk <- predictRisk(fgr_COMBINED,
                            newdata = fgr_all_data,
                            times = risk_time_cutoff,
                            cause = 1)
  risk <- as.numeric(risk[, 1])
  hist(log(risk),breaks = 100,
       main = paste0("Predicted risk at year = ",risk_time_cutoff))
  
  threshold <- plotpredictiveness(
    risk = risk,        # predicted risk at 5 years
    marker = risk,      # same marker?
    status = fgr_all_data$RFi_event
  )
  print(threshold)
  selected_cutoffs <- append(selected_cutoffs,list(threshold))
  fgr_all_data$risk_group <- ifelse(risk >= threshold, 
                                    "High risk", "Low risk")
  ci <- cuminc(
    ftime = fgr_all_data$time_to_CompRisk_event, 
    fstatus = fgr_all_data$CompRisk_event_coded,
    group = fgr_all_data$risk_group
  )
  # Rename curves with counts
  counts <- table(fgr_all_data$risk_group)
  names(ci)[1:4] <- c(
    paste0("High risk – Recurrence (n=", counts["High risk"], ")"),
    paste0("Low risk – Recurrence (n=", counts["Low risk"], ")"),
    paste0("High risk – Death without recurrence (n=", counts["High risk"], ")"),
    paste0("Low risk – Death without recurrence (n=", counts["Low risk"], ")")
  )
  
  # plot 
  plot(ci[1:2], 
       lty = 1:2, 
       col = c("red", "pink"),  # 4 colors for the 4 curves
       xlab = "Time (years)",
       ylab = "Cumulative incidence",
       main = "Cumulative Incidence of Recurrence")
  mtext(paste0("Year ",risk_time_cutoff," Cutoff = ", threshold), side = 3, line = 0.5, cex = 0.9)
  
  # Plot with 4 distinct colors
  plot(ci, 
       lty = 1:2, 
       col = c("red", "pink", "blue", "lightblue"),  # 4 colors for the 4 curves
       xlab = "Time (years)",
       ylab = "Cumulative incidence",
       main = "Cumulative Incidence of Recurrence and Death w/o Recurrence")
  mtext(paste0("Cutoff = ", threshold), side = 3, line = 0.5, cex = 0.9)
}
names(selected_cutoffs) <- c("5_year","10_year")
dev.off()

###############################################################################
# SAVE FINAL MODEL RESULTS
###############################################################################

final_results <- list(
  
  # The three models
  final_model = list(
    model = fgr_COMBINED,
    rs_cutoffs = selected_cutoffs,
    var_effects = vimp_fgr_COMBINED
  ),
  
  # MRS construction
  mrs_construction = list(
    selected_cpgs = pFG_selected_cpgs,
    cpg_coefs = pFG_cpg_coefs,
    cpg_scaling = pFG_cpg_scaling,
    mrs_scaling = mrs_scaling
  ),
  
  # Clinical preprocessing  
  clinical_preprocessing = list(
    continuous_vars = CLIN_CONT,
    categorical_vars = CLIN_CATEGORICAL,
    encoded_cols = encoded_result$encoded_cols,
    clinical_scaling = clin_scaling
  )
)

save(final_results, file = file.path(current_output_dir, "final_fg_results.RData"))


###############################################################################
# SAVE DEPLOYMENT OBJECT (for applying to independent data)
###############################################################################

deployment_object <- list(
  
  # For MRS calculation
  mrs_params = list(
    required_cpgs = pFG_selected_cpgs,
    cpg_coefs = pFG_cpg_coefs,
    cpg_scaling = pFG_cpg_scaling,
    mrs_scaling = mrs_scaling
  ),
  
  # For clinical variable preparation
  clinical_params = list(
    continuous_vars = CLIN_CONT,
    categorical_vars = CLIN_CATEGORICAL,
    encoded_cols = encoded_result$encoded_cols,
    clinical_scaling = clin_scaling
  ),
  
  # The three models
  final_model = list(
    model = fgr_COMBINED,
    rs_cutoffs = selected_cutoffs
    )
)

save(deployment_object, file = file.path(current_output_dir, "deployment_object.RData"))

cat("\nSaved:\n")
cat("  - final_fg_results.RData (complete results)\n")
cat("  - deployment_object.RData (for predictions on new data)\n")

###############################################################################
# CLEANUP AND FINISH
###############################################################################

total_runtime <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
# 
cat(sprintf("\n%s\n", paste(rep("=", 80), collapse = "")))
cat(sprintf("PIPELINE COMPLETED\n"))
cat(sprintf("Total runtime: %.1f minutes\n", total_runtime))
cat(sprintf("Finished: %s\n", Sys.time()))
cat(sprintf("%s\n", paste(rep("=", 80), collapse = "")))

# Close log file
sink(type = "message")
sink(type = "output")
close(log_con)
