#!/usr/bin/env Rscript
################################################################################
# Final Fine-Gray Model training
# Author: Lennart Hohmann
################################################################################
# LIBRARIES
################################################################################
setwd("~/PhD_Workspace/PredictRecurrence/")

library(readr)          # fast CSV reading
library(dplyr)          # data manipulation
library(survival)       # Surv objects
library(glmnet)         # elastic net Cox regression
library(cmprsk)         # competing risks
library(caret)          # CV fold creation
library(data.table)     # fast data reading
library(riskRegression) # Score() for competing risks metrics
library(prodlim)        # prodlim::Hist()
library(devtools)
library(fastcmprsk)

source("./src/finegray_functions.R")

################################################################################
# COMMAND LINE ARGUMENTS
################################################################################
args <- commandArgs(trailingOnly = TRUE)

COHORT_NAME <- "TNBC" # 'TNBC', 'ERpHER2n', 'All'
if (length(args) == 1) COHORT_NAME <- args[1]

################################################################################
# INPUT / OUTPUT
################################################################################
SCRIPT_NAME <- "finegray_FinalModel_CV"

INFILE_METHYLATION <- "./data/train/train_methylation_unadjusted.csv"
INFILE_CLINICAL    <- "./data/train/train_clinical.csv"
TRAIN_CPGS         <- "./data/set_definitions/CpG_prefiltered_sets/cpg_ids_atac_overlap.txt"

COHORT_TRAIN_IDS_PATHS <- list(
  TNBC     = "./data/train/train_subcohorts/TNBC_train_ids.csv",
  ERpHER2n = "./data/train/train_subcohorts/ERpHER2n_train_ids.csv",
  All      = "./data/train/train_subcohorts/All_train_ids.csv"
)

current_output_dir <- file.path("./output/FineGray", COHORT_NAME)
dir.create(current_output_dir, recursive = TRUE, showWarnings = FALSE)

################################################################################
# LOGGING
################################################################################
log_path <- file.path(current_output_dir,
                      sprintf("%s_%s.log", SCRIPT_NAME,
                              format(Sys.time(), "%Y%m%d_%H%M%S")))
# comment to switch off logging
log_con <- file(log_path, open = "wt")
sink(log_con, type = "output", split = TRUE)
sink(log_con, type = "message")

################################################################################
# PARAMETERS
################################################################################
# Cross-validation
N_INNER_FOLDS     <- 5
VARIANCE_QUANTILE <- 0.75
ALPHA_GRID        <- c(0.5)#c(0.5, 0.7, 0.9)

# Cohort-specific
ADMIN_CENSORING_CUTOFF <- if (COHORT_NAME == "TNBC") 5.01 else NULL
EVAL_TIMES             <- if (COHORT_NAME == "TNBC") 1:5 else 1:10

# Clinical variables (final combined model only)
CLIN_CONT        <- c("Age", "Size.mm")
CLIN_CATEGORICAL <- if (COHORT_NAME == "All") c("NHG", "LN", "ER", "PR", "HER2") else c("NHG", "LN")
CLINVARS_ALL     <- c(CLIN_CONT, CLIN_CATEGORICAL)

################################################################################
# RUN INFO
################################################################################
start_time <- Sys.time()
cat(paste(rep("=", 80), collapse = ""), "\n")
cat(sprintf("%s\n", toupper(SCRIPT_NAME)))
cat(sprintf("Started:             %s\n", start_time))
cat(sprintf("Cohort:              %s\n", COHORT_NAME))
cat(sprintf("Clinical variables:  %s\n", paste(CLINVARS_ALL, collapse = ", ")))
cat(sprintf("Evaluation times:    %s\n", paste(EVAL_TIMES,   collapse = ", ")))
cat(sprintf("Log file:            %s\n", log_path))
cat(paste(rep("=", 80), collapse = ""), "\n\n")

################################################################################
# DATA LOADING AND PREPROCESSING
################################################################################

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

cat(sprintf("Training data: n=%d (RFi=%d, Death=%d)\n",
            nrow(X), 
            sum(clinical_data$RFi_event), 
            sum(clinical_data$DeathNoR_event)))

# --------------------------------------------------------------------------
X_filtered <- prepare_filtered_features(
  X = X,
  vars_preserve = NULL,
  variance_quantile = VARIANCE_QUANTILE,
  apply_cox_filter = FALSE)

# --------------------------------------------------------------------------
# CoxNet for RFI
cat(sprintf("\n========== COXNET FOR RFI ==========\n"))
rfi_model <- tune_and_fit_coxnet(
  X_train = X_filtered,
  y_train = y_rfi,
  clinical_train = clinical_data,
  event_col = "RFi_event",
  alpha_grid = ALPHA_GRID,
  penalty_factor = rep(1, ncol(X_filtered)),
  n_inner_folds = N_INNER_FOLDS,
  outcome_name = "RFI"
)
# Extract coefficients and features
rfi_coef_res <- extract_nonzero_coefs(rfi_model$final_fit)
coef_rfi_df <- rfi_coef_res$coef_df
features_rfi <- rfi_coef_res$features
cat(sprintf(
  "BEST RFI MODEL: Alpha=%.2f, Lambda=%.6f, Features=%d\n",
  rfi_model$best_alpha, 
  rfi_model$best_lambda, 
  length(features_rfi)
))
cat("Selected:", paste(features_rfi, collapse=", "), "\n")

# --------------------------------------------------------------------------
# CoxNet for Death without Recurrence
cat(sprintf("\n========== COXNET FOR DEATH w/o RECURRENCE ==========\n"))
death_model <- tune_and_fit_coxnet(
  X_train = X_filtered,
  y_train = y_death_no_r,
  clinical_train = clinical_data,
  event_col = "DeathNoR_event",
  alpha_grid = ALPHA_GRID,
  penalty_factor = rep(1, ncol(X_filtered)),
  n_inner_folds = N_INNER_FOLDS,
  outcome_name = "DeathNoR"
)
# Extract coefficients and features
death_coef_res <- extract_nonzero_coefs(death_model$final_fit)
coef_death_df <- death_coef_res$coef_df
features_death <- death_coef_res$features
cat(sprintf(
  "BEST DeathNoR MODEL: Alpha=%.2f, Lambda=%.6f, Features=%d\n",
  death_model$best_alpha, 
  death_model$best_lambda, 
  length(features_death)
))
cat("Selected:", paste(features_death, collapse=", "), "\n")

# --------------------------------------------------------------------------
# Pool Features from Both Models
features_pooled <- union(features_rfi, features_death)
cat(sprintf("\n--- Feature Pooling ---\n"))
cat(sprintf("RFI features: %d\n", length(features_rfi)))
cat(sprintf("Death features: %d\n", length(features_death)))
cat(sprintf("Overlap: %d\n", length(intersect(features_rfi, features_death))))
cat(sprintf("Pooled total: %d\n", length(features_pooled)))
# Prepare Fine-Gray input
X_pooled <- X[, features_pooled, drop = FALSE]

###############################################################################
# Fit Penalized Fine-Gray Model (MRS Construction)
###############################################################################
cat(sprintf("\n========== PENALIZED FINE-GRAY (MRS CONSTRUCTION) ==========\n"))
pFG_res <- fit_penalized_finegray_cv(
  X_input = X_pooled,
  clinical_input = clinical_data,
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

###############################################################################
# Calculate Methylation Risk Score (MRS)
###############################################################################
mrs_res <- calculate_methylation_risk_score(
  X_data = X_pooled,
  cpg_coefficients = pFG_cpg_coefs,
  scaling_params = pFG_cpg_scaling,
  verbose = TRUE
)
MRS <- mrs_res$mrs
# scale
mrs_scaled <- scale(MRS)
mrs_scaling <- list(
  center = attr(mrs_scaled, "scaled:center"),
  scale = attr(mrs_scaled, "scaled:scale")
)
MRS <- as.numeric(mrs_scaled)
cat("  MRS scaled to mean=0, sd=1\n")

###############################################################################
# Prepare Data for Final Unpenalized Fine-Gray Model
###############################################################################
cat(sprintf("\n--- Preparing Final Model Data ---\n"))
# Verify row alignment
if (!identical(rownames(clinical_data), rownames(X_pooled))) {
  stop("Row names don't match between clinical_train and X_pooled_train")
}
# Build training data
fgr_data <- data.frame(
  time_to_CompRisk_event = clinical_data$time_to_CompRisk_event,
  CompRisk_event_coded = clinical_data$CompRisk_event_coded,
  encoded_result$encoded_df[rownames(X_pooled), , drop = FALSE],
  MeRS = MRS,
  row.names = rownames(clinical_data)
)
# Scale clinical continuous variables
clin_scaled <- scale(fgr_data[, CLIN_CONT, drop = FALSE])
clin_scaling <- list(
  variables = CLIN_CONT,
  center = attr(clin_scaled, "scaled:center"),
  scale = attr(clin_scaled, "scaled:scale")
)
fgr_data[, CLIN_CONT] <- clin_scaled
cat(sprintf("Final model predictors: %s\n", 
            paste(setdiff(colnames(fgr_data), 
                          c("time_to_CompRisk_event", "CompRisk_event_coded")), 
                  collapse=", ")))

###############################################################################
# Fit Unpenalized Fine-Gray Model
###############################################################################
fgr_COMBINED <- fit_fine_gray_model(
  fgr_data = fgr_data,
  cr_time = "time_to_CompRisk_event",
  cr_event = "CompRisk_event_coded",
  cause = 1
)

fgr_COMBINED <- fgr_COMBINED$model

# ----------------------------------------------------------------------------
# Calculate Variable Importance
cat("\n--- Fine-Gray Variable Importance ---\n")
vimp_fgr_COMBINED <- calculate_fgr_importance(
  fgr_model = fgr_COMBINED,
  encoded_cols = encoded_result$encoded_cols,
  verbose = FALSE
)
print(vimp_fgr_COMBINED)
###############################################################################
# CALCULATE RISK SCORE CUTOFF FOR DICHOTOMIZATION
###############################################################################
cat(sprintf("\n--- Timepoint Risk Score Cutoffs ---\n"))

fgr_data$RFi_event <- clinical_data$RFi_event[match(rownames(fgr_data),clinical_data$Sample)]
fgr_data$RFi_years <- clinical_data$RFi_years[match(rownames(fgr_data),clinical_data$Sample)]

pdf(file.path(current_output_dir, "cutoff_selection_plots.pdf"), width = 6, height = 11)
par(mfrow = c(3,1))

selected_cutoffs <- list()
cutoff_time_points <- EVAL_TIMES

for(risk_time_cutoff in cutoff_time_points) {
  # apply model to whole data cohort (same data it was trained on)
  risk <- predictRisk(fgr_COMBINED,
                            newdata = fgr_data,
                            times = risk_time_cutoff,
                            cause = 1)
  risk <- as.numeric(risk[, 1])
  hist(log(risk),breaks = 100,
       main = paste0("Predicted risk at year = ",risk_time_cutoff))
  
  threshold <- plotpredictiveness(
    risk = risk,        # predicted risk at 5 years
    marker = risk,      # same marker?
    status = fgr_data$RFi_event
  )

  # print info
  threshold_value <- as.numeric(threshold)
  threshold_percentile <- as.numeric(gsub("%", "", names(threshold)))
  counts <- table(fgr_data$risk_group)
  cat(sprintf(
    "Year %d | Cutoff at %.2fth percentile | Risk threshold = %.4f | High risk: %d | Low risk: %d\n",
    risk_time_cutoff,
    threshold_percentile,
    threshold_value,
    counts["High risk"],
    counts["Low risk"]
  ))
  
  # cumulative incidence plots 
  selected_cutoffs <- append(selected_cutoffs,list(threshold))
  fgr_data$risk_group <- ifelse(risk >= threshold, 
                                    "High risk", "Low risk")
  ci <- cuminc(
    ftime = fgr_data$time_to_CompRisk_event, 
    fstatus = fgr_data$CompRisk_event_coded,
    group = fgr_data$risk_group
  )
  # Rename curves with counts
  names(ci)[1:4] <- c(
    paste0("High risk - Recurrence (n=", counts["High risk"], ")"),
    paste0("Low risk - Recurrence (n=", counts["Low risk"], ")"),
    paste0("High risk - Death without recurrence (n=", counts["High risk"], ")"),
    paste0("Low risk - Death without recurrence (n=", counts["Low risk"], ")")
  )
  
  # plot 
  plot(ci[1:2], 
       lty = 1:2, 
       col = c("red", "pink"),  # 4 colors for the 4 curves
       xlab = "Time (years)",
       ylab = "Cumulative incidence",
       main = "Cumulative Incidence of Recurrence")
  mtext(
    sprintf("Year %d | Cutoff = %.4f (%.2fth percentile)",
            risk_time_cutoff,
            threshold_value,
            threshold_percentile),
    side = 3, line = 0.5, cex = 0.9
  )
  
  # Plot with 4 distinct colors
  # plot(ci, 
  #      lty = 1:2, 
  #      col = c("red", "pink", "blue", "lightblue"),  # 4 colors for the 4 curves
  #      xlab = "Time (years)",
  #      ylab = "Cumulative incidence",
  #      main = "Cumulative Incidence of Recurrence and Death w/o Recurrence")
  # mtext(paste0("Cutoff = ", threshold), side = 3, line = 0.5, cex = 0.9)
}
names(selected_cutoffs) <- paste0(cutoff_time_points, "_year")
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
cat(paste(rep("=", 80), collapse = ""), "\n")
cat(sprintf("PIPELINE COMPLETED\n"))
cat(sprintf("Finished:      %s\n", Sys.time()))
cat(sprintf("Total runtime: %.1f minutes\n", total_runtime))
cat(paste(rep("=", 80), collapse = ""), "\n")

# uncomment if logging enabled
sink(type = "message")
sink(type = "output")
close(log_con)

