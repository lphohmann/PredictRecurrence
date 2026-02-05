#!/usr/bin/env Rscript

################################################################################
# Fine-Gray Competing Risks with Nested Cross-Validation
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
source("./src/finegray_functions.R") # match real name

################################################################################
# COMMAND LINE ARGUMENTS
################################################################################

args <- commandArgs(trailingOnly = TRUE)

# Defaults
DEFAULT_COHORT <- "ERpHER2n"
DEFAULT_DATA_MODE <- "combined"
DEFAULT_TRAIN_CPGS <- "./data/set_definitions/CpG_prefiltered_sets/cpg_ids_atac_overlap.txt"
DEFAULT_OUTPUT_DIR <- "./output/FineGray"

# Parse arguments
if (length(args) == 0) {
  cat("No command line arguments provided. Using defaults.\n")
  COHORT_NAME <- DEFAULT_COHORT
  DATA_MODE <- DEFAULT_DATA_MODE
  TRAIN_CPGS <- DEFAULT_TRAIN_CPGS
  OUTPUT_BASE_DIR <- DEFAULT_OUTPUT_DIR
  
} else if (length(args) == 2) {
  COHORT_NAME <- args[1]
  DATA_MODE <- args[2]
  TRAIN_CPGS <- DEFAULT_TRAIN_CPGS
  OUTPUT_BASE_DIR <- DEFAULT_OUTPUT_DIR
  
} else {
  cat("\n=== USAGE ===\n")
  cat("Rscript finegray_pipeline.R <COHORT> <DATA_MODE>\n\n")
  cat("Arguments:\n")
  cat("  COHORT      : 'TNBC', 'ERpHER2n', or 'All'\n")
  cat("  DATA_MODE   : 'methylation' or 'combined'\n\n")
  cat("Example:\n")
  cat("  Rscript finegray_pipeline.R ERpHER2n combined\n\n")
  cat("Or run without arguments to use defaults:\n")
  cat("  Rscript finegray_pipeline.R\n\n")
  stop("Incorrect number of arguments provided.")
}

# Validate inputs
if (!COHORT_NAME %in% c("TNBC", "ERpHER2n", "All")) {
  stop(sprintf("Invalid COHORT_NAME: '%s'. Must be 'TNBC', 'ERpHER2n', or 'All'", COHORT_NAME))
}

if (!DATA_MODE %in% c("methylation", "combined")) {
  stop(sprintf("Invalid DATA_MODE: '%s'. Must be 'methylation' or 'combined'", DATA_MODE))
}

if (TRAIN_CPGS == "NULL") {
  TRAIN_CPGS <- NULL
}

# Print configuration
cat(sprintf("\n=== PIPELINE SETTINGS ===\n"))
cat(sprintf("Cohort:      %s\n", COHORT_NAME))
cat(sprintf("Data mode:   %s\n", DATA_MODE))
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

# Output directory
current_output_dir <- file.path(
  OUTPUT_BASE_DIR, 
  COHORT_NAME, 
  tools::toTitleCase(DATA_MODE), 
  "Unadjusted"
)
dir.create(current_output_dir, recursive = TRUE, showWarnings = FALSE)

################################################################################
# SETUP LOGGING
################################################################################

# Create log filename with script name and timestamp
script_name <- "finegray_pipeline_var_coxnet"
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
log_filename <- sprintf("%s_%s.log", script_name, timestamp)
log_path <- file.path(current_output_dir, log_filename)

# Start logging (captures all output and messages)
log_con <- file(log_path, open = "wt")
sink(log_con, type = "output", split = TRUE)  # split = TRUE shows output in console too
sink(log_con, type = "message")

cat(sprintf("%s\n", paste(rep("=", 80), collapse = "")))
cat(sprintf("FINE-GRAY COMPETING RISKS PIPELINE\n"))
cat(sprintf("Started: %s\n", Sys.time()))
cat(sprintf("Cohort: %s\n", COHORT_NAME))
cat(sprintf("Data mode: %s\n", DATA_MODE))
cat(sprintf("Log file: %s\n", log_path))
cat(sprintf("%s\n\n", paste(rep("=", 80), collapse = "")))

################################################################################
# PARAMETERS
################################################################################

# Clinical variables - always preserved in filtering
ADMIN_CENSORING_CUTOFF <- if (COHORT_NAME == "TNBC") 5.01 else NULL

# Categorical variables depend on cohort
if (COHORT_NAME == "All") {
  CLIN_CATEGORICAL <- c("NHG", "LN", "ER", "PR", "HER2")
} else {
  CLIN_CATEGORICAL <- c("NHG", "LN")
}

CLIN_CONT <- c("Age", "Size.mm")

cat(sprintf("Clinical categorical variables: %s\n", paste(CLIN_CATEGORICAL, collapse=", ")))
cat(sprintf("Clinical continuous variables: %s\n", paste(CLIN_CONT, collapse=", ")))

# Feature selection - variance filter only
VARIANCE_QUANTILE <- 0.75  # Keep top 25% most variable features

# Variables to not penalize in elastic net (after one-hot encoding)
# These get coefficient shrinkage but not eliminated (penalty_factor = 0)

# Cross-validation
N_OUTER_FOLDS <- 5
N_INNER_FOLDS <- 3
ALPHA_GRID <- c(0.5,0.7,0.9)  # Elastic net mixing: 0=ridge, 1=lasso, 0.9=mostly lasso

# Performance evaluation timepoints (in years)
EVAL_TIMES <- seq(1, 10)

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

# Convert beta values to M-values
mvals <- beta_to_m(beta_matrix, beta_threshold = 0.001)

# Apply administrative censoring if needed
if (!is.null(ADMIN_CENSORING_CUTOFF)) {
  clinical_data <- apply_admin_censoring(
    clinical_data, "RFi_years", "RFi_event", ADMIN_CENSORING_CUTOFF
  )
}

# Subset to predefined CpGs (if specified)
if (!is.null(TRAIN_CPGS)) {
  mvals <- subset_methylation(mvals, TRAIN_CPGS)
}

################################################################################
# FEATURE MATRIX CONSTRUCTION
################################################################################

if (DATA_MODE == "methylation") {
  # Methylation-only mode
  X <- mvals
  clinvars <- NULL
  encoded_result <- list(encoded_cols = NULL)
  cat(sprintf("Using methylation features only: %d CpGs\n", ncol(X)))
  
} else if (DATA_MODE == "combined") {
  # Combined mode: methylation + clinical
  X <- mvals
  clinical_data$LN <- gsub("N\\+", "Np", clinical_data$LN)
  clin <- clinical_data[c(CLIN_CONT, CLIN_CATEGORICAL)]
  clin <- clin[rownames(X), , drop = FALSE]
  encoded_result <- onehot_encode_clinical(clin, CLIN_CATEGORICAL)
  X <- cbind(X, encoded_result$encoded_df)
  clinvars <- c(CLIN_CONT, encoded_result$encoded_cols)
  
  cat(sprintf("Combined features: %d CpGs + %d clinical = %d total\n", 
              ncol(mvals), length(clinvars), ncol(X)))
  cat(sprintf("Clinical variables: %s\n", paste(clinvars, collapse = ", ")))
}

# Set feature handling parameters
included_clinvars <- if (DATA_MODE == "methylation") NULL else clinvars
VARS_PRESERVE <- included_clinvars
VARS_NO_PENALIZATION <- included_clinvars

# Validate variables
if (!is.null(VARS_PRESERVE)) {
  missing <- setdiff(VARS_PRESERVE, colnames(X))
  if (length(missing) > 0) {
    warning(sprintf("VARS_PRESERVE contains non-existent columns: %s", 
                    paste(missing, collapse = ", ")))
    VARS_PRESERVE <- intersect(VARS_PRESERVE, colnames(X))
  }
  cat(sprintf("Preserving %d variables from pre-filtering.\n", length(VARS_PRESERVE)))
}

if (!is.null(VARS_NO_PENALIZATION)) {
  missing <- setdiff(VARS_NO_PENALIZATION, colnames(X))
  if (length(missing) > 0) {
    warning(sprintf("VARS_NO_PENALIZATION contains non-existent columns: %s",
                    paste(missing, collapse = ", ")))
    VARS_NO_PENALIZATION <- intersect(VARS_NO_PENALIZATION, colnames(X))
  }
  cat(sprintf("Not penalizing %d variables in elastic net.\n", 
              length(VARS_NO_PENALIZATION)))
}

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
# NESTED CROSS-VALIDATION SETUP
################################################################################

cat(sprintf("\n========== NESTED CROSS-VALIDATION ==========\n"))
cat(sprintf("Outer folds: %d\n", N_OUTER_FOLDS))
cat(sprintf("Inner folds: %d\n", N_INNER_FOLDS))
cat(sprintf("Alpha grid: %s\n", paste(ALPHA_GRID, collapse = ", ")))

# Create stratified outer folds based on RFI events
# Stratification ensures similar event rates across folds
set.seed(123)
outer_folds <- createFolds(
  y = clinical_data$RFi_event,
  k = N_OUTER_FOLDS,
  list = TRUE,
  returnTrain = FALSE  # Returns test indices
)

# Storage for results
outer_fold_results <- list()

################################################################################
# OUTER CV LOOP - Performance Estimation
################################################################################

for (fold_idx in 1:N_OUTER_FOLDS) {
  cat(sprintf("\n========== OUTER FOLD %d/%d ==========\n", 
              fold_idx, N_OUTER_FOLDS))
  
  fold_start_time <- Sys.time()
  
  # --------------------------------------------------------------------------
  # Split Data into Outer Train and Test
  # --------------------------------------------------------------------------
  
  test_idx <- outer_folds[[fold_idx]]
  train_idx <- setdiff(1:nrow(X), test_idx)
  
  outer_data <- cv_data_prep(X,clinical_data,train_idx,test_idx)
  
  X_train <- outer_data$X_train
  X_test <- outer_data$X_test
  clinical_train <- outer_data$clinical_train
  clinical_test <- outer_data$clinical_test
  
  y_rfi_train <- y_rfi[train_idx]
  y_rfi_test <- y_rfi[test_idx]
  
  y_death_train <- y_death_no_r[train_idx]
  y_death_test <- y_death_no_r[test_idx]
  
  cat(sprintf("Train: n=%d (RFi=%d, Death=%d)\n",
              nrow(X_train), 
              sum(clinical_train$RFi_event), 
              sum(clinical_train$DeathNoR_event)))
  cat(sprintf("Test: n=%d (RFi=%d, Death=%d)\n",
              nrow(X_test), 
              sum(clinical_test$RFi_event), 
              sum(clinical_test$DeathNoR_event)))

  # --------------------------------------------------------------------------
  # INNER CV LOOP #1: CoxNet for RFI
  # --------------------------------------------------------------------------
  
  # Reduce dimensionality before elastic net
  # Feature Filtering on Outer Training Data
  cat(sprintf("\n========== COXNET FOR RFI ==========\n"))
  
  X_train_filtered_rfi <- prepare_filtered_features(
    X = X_train,
    vars_preserve = VARS_PRESERVE,
    variance_quantile = VARIANCE_QUANTILE,
    apply_cox_filter = FALSE)#,
    #y_train = y_rfi_train,
    #cox_selection_method = "top_n",
    #cox_top_n = 5000
  #)
  
  # Create penalty factor (0 = not penalized, 1 = penalized)
  penalty_factor_rfi <- rep(1, ncol(X_train_filtered_rfi))
  if (!is.null(VARS_NO_PENALIZATION)) {
    penalty_factor_rfi[colnames(X_train_filtered_rfi) %in% VARS_NO_PENALIZATION] <- 0
  }
  
  # PURPOSE: Select features predictive of recurrence
  # Cox proportional hazards with elastic net regularization
  rfi_model <- tune_and_fit_coxnet(
    X_train = X_train_filtered_rfi,
    y_train = y_rfi_train,
    clinical_train = clinical_train,
    event_col = "RFi_event",
    alpha_grid = ALPHA_GRID,
    penalty_factor = penalty_factor_rfi,
    n_inner_folds = N_INNER_FOLDS,
    outcome_name = "RFI",
    compute_stability = FALSE
  )
  
  # Extract coefficients and features
  rfi_coef_res <- extract_nonzero_coefs(rfi_model$final_fit)
  coef_rfi_df <- rfi_coef_res$coef_df
  features_rfi <- rfi_coef_res$features
  
  # in case of stablity filter
  # stability_info_rfi <- rfi_model$best_result$stability_info
  # stable_nonclinical_rfi <- stability_info_rfi$feature[
  #   !(stability_info_rfi$feature %in% VARS_NO_PENALIZATION) &
  #     stability_info_rfi$selection_freq >= STABILITY_THRESHOLD &
  #     stability_info_rfi$sign_consistent == TRUE
  # ]
  # features_rfi <- c(VARS_NO_PENALIZATION, stable_nonclinical_rfi)
  
  cat(sprintf(
    "  BEST RFI MODEL: Alpha=%.2f, Lambda=%.6f, Features=%d\n",
    rfi_model$best_alpha, 
    rfi_model$best_lambda, 
    length(features_rfi)
  ))
  
  cat("    Selected:", paste(features_rfi, collapse=", "), "\n")
  
  # --------------------------------------------------------------------------
  # INNER CV LOOP #2: CoxNet for Death without Recurrence
  # --------------------------------------------------------------------------
  
  # Feature Filtering on Outer Training Data
  cat(sprintf("\n========== COXNET FOR DEATH w/o RECURRENCE ==========\n"))
  
  X_train_filtered_death <- prepare_filtered_features(
    X = X_train,
    vars_preserve = VARS_PRESERVE,
    variance_quantile = VARIANCE_QUANTILE,
    apply_cox_filter = FALSE)#,
    #y_train = y_death_train,
    #cox_selection_method = "top_n",
    #cox_top_n = 5000
  #)
  
  # Create penalty factor (0 = not penalized, 1 = penalized)
  # Unpenalized features can still be shrunk but won't be eliminated
  penalty_factor_death <- rep(1, ncol(X_train_filtered_death))
  if (!is.null(VARS_NO_PENALIZATION)) {
    penalty_factor_death[colnames(X_train_filtered_death) %in% VARS_NO_PENALIZATION] <- 0
  }
  
  # PURPOSE: Select features predictive of death (competing risk)
  # This ensures we account for features that predict competing events
  
  death_model <- tune_and_fit_coxnet(
    X_train = X_train_filtered_death,
    y_train = y_death_train,
    clinical_train = clinical_train,
    event_col = "DeathNoR_event",
    alpha_grid = ALPHA_GRID,
    penalty_factor = penalty_factor_death,
    n_inner_folds = N_INNER_FOLDS,
    outcome_name = "DeathNoR",
    compute_stability = FALSE
  )
  
  # Extract coefficients and features
  death_coef_res <- extract_nonzero_coefs(death_model$final_fit)
  coef_death_df <- death_coef_res$coef_df
  features_death <- death_coef_res$features
  
  cat(sprintf(
    "  BEST DeathNoR MODEL: Alpha=%.2f, Lambda=%.6f, Features=%d\n",
    death_model$best_alpha, 
    death_model$best_lambda, 
    length(features_death)
  ))
  
  cat("    Selected:", paste(features_death, collapse=", "), "\n")
  
  # --------------------------------------------------------------------------
  # Pool Features from Both Models
  # --------------------------------------------------------------------------

  features_pooled <- union(features_rfi, features_death)
  
  cat(sprintf("\n--- Feature Pooling ---\n"))
  cat(sprintf("RFI features: %d\n", length(features_rfi)))
  cat(sprintf("Death features: %d\n", length(features_death)))
  cat(sprintf("Overlap: %d\n", length(intersect(features_rfi, features_death))))
  cat(sprintf("Pooled total: %d\n", length(features_pooled)))
  
  # Check if any features were selected
  if (length(features_pooled) == 0) {
    warning(sprintf("Fold %d: No features selected by either model. Skipping fold.", fold_idx))
    next
  }
  
  # Prepare Fine-Gray input
  X_pooled_train <- X_train[, features_pooled, drop = FALSE]
  X_pooled_test <- X_test[, features_pooled, drop = FALSE]
  
  # --------------------------------------------------------------------------
  # Fit Penalized Fine-Gray Model on Pooled Features to calc. Risk score, 
  # dont include clinical vars in this calculation
  # --------------------------------------------------------------------------
  # Fine-Gray model: Models subdistribution hazard for event of interest
  
  selected_cpgs <- setdiff(colnames(X_pooled_train), included_clinvars)
  
  cat(sprintf("\n========== PENALIZED FINE-GRAY (MRS CONSTRUCTION) ==========\n"))
  
  pFG_res <- fit_penalized_finegray_cv(
    X_input = X_pooled_train[, selected_cpgs, drop = FALSE],
    clinical_input = clinical_train,
    alpha_seq = c(0.1, 0.2, 0.3, 0.4, 0.5),
    lambda_seq = NULL,
    cr_time = "time_to_CompRisk_event",
    cr_event = "CompRisk_event_coded",
    penalty = "ENET",
    n_inner_folds = 3
  )
  
  # Extract selected features and coefficients
  selected_cpgs_for_mrs <- pFG_res$results_table$feature[pFG_res$results_table$selected]
  selected_cpgs_coefs <- setNames(
    pFG_res$results_table$pFG_coefficient[pFG_res$results_table$selected],
    selected_cpgs_for_mrs
  )
  # scaling parsm for meRS calculation
  selected_cpgs_scaleparams <- list ("centers"=pFG_res$results_table$scale_center,
                                    "scale"=pFG_res$results_table$scale_scale)
  
  cat(sprintf("  BEST PENALIZED FG: Alpha=%.2f, Lambda=%.6f, CpGs=%d\n",
              pFG_res$best_alpha,
              pFG_res$best_lambda,
              length(selected_cpgs_for_mrs)))
  cat("    Selected CpGs:", paste(selected_cpgs_for_mrs, collapse=", "), "\n")
  
  # --------------------------------------------------------------------------
  # Calculate Methylation Risk Score (MRS)
  # --------------------------------------------------------------------------
  # MRS = weighted sum of selected CpGs (scaled)
  
  cat(sprintf("\n--- Calculating Methylation Risk Score ---\n"))
  
  # Calculate MRS for Training Data
  mrs_train_result <- calculate_methylation_risk_score(
    X_data = X_pooled_train,
    cpg_coefficients = selected_cpgs_coefs,
    scaling_params = selected_cpgs_scaleparams,
    verbose = TRUE
  )
  MRS_train <- mrs_train_result$mrs

  # Calculate MRS for Test Data (using training scaling)
  mrs_test_result <- calculate_methylation_risk_score(
    X_data = X_pooled_test,
    cpg_coefficients = selected_cpgs_coefs,
    scaling_params = selected_cpgs_scaleparams,
    verbose = TRUE
  )
  MRS_test <- mrs_test_result$mrs
  
  # Scale the MRS itself
  scale_mrs <- TRUE
  if (scale_mrs) {
    # Fit scaling on training MRS
    mrs_train_scaled <- scale(MRS_train)
    # Extract parameters
    mrs_center <- attr(mrs_train_scaled, "scaled:center")
    mrs_scale  <- attr(mrs_train_scaled, "scaled:scale")
    # Apply the same scaling to test MRS
    MRS_train <- as.numeric(mrs_train_scaled)
    MRS_test  <- as.numeric((MRS_test - mrs_center) / mrs_scale)
    cat("  MRS scaled to mean=0, sd=1\n")
  }
  
  # --------------------------------------------------------------------------
  # Prepare Data for Final Unpenalized Fine-Gray Model
  # --------------------------------------------------------------------------
  # Combine: clinical variables + MRS
  
  cat(sprintf("\n--- Preparing Final Model Data ---\n"))
  
  identical(clinical_train$Sample,row.names(X_pooled_train)) #true
  
  fgr_train_data <- cbind(
    clinical_train[, c("time_to_CompRisk_event", "CompRisk_event_coded")],
    X_pooled_train[, included_clinvars, drop = FALSE],
    methylation_risk_score = MRS_train
  )
  
  fgr_test_data <- cbind(
    clinical_test[, c("time_to_CompRisk_event", "CompRisk_event_coded")],
    X_pooled_test[, included_clinvars, drop = FALSE],
    methylation_risk_score = MRS_test
  )
  
  # Scale clinical continuous variables
  clin_scale <- scale_continuous_features(
    X_train = fgr_train_data[, CLIN_CONT, drop = FALSE],
    X_test  = fgr_test_data[, CLIN_CONT, drop = FALSE]
  )
  
  fgr_train_data[, CLIN_CONT] <- clin_scale$X_train_scaled
  fgr_test_data[, CLIN_CONT] <- clin_scale$X_test_scaled
  
  cat(sprintf("  Final model predictors: %s\n", 
              paste(setdiff(colnames(fgr_train_data), 
                            c("time_to_CompRisk_event", "CompRisk_event_coded")), 
                    collapse=", ")))
  
  # preprocessing params
  fg_scaling_params <- list(
    
    # need scaling aprams from final pFG model input
    #pFG_res$results_table
    # CpGs scaling pre MRS calculation
    scaleparams_meRS_cpgs = list(cpg_coefs = mrs_train_result$input_cpg_coefs,
                                  center = mrs_train_result$scaling_params$center,
                                 scale = mrs_train_result$scaling_params$scale),
    
    # MRS scaling
    scaleparams_fg_meRS = list(center = mrs_center,scale = mrs_scale),
    
    # Clinical continuous variables scaling
    scaleparams_fg_clincont = list(
      variables = CLIN_CONT,
      center = clin_scale$center,
      scale  = clin_scale$scale
    )
  )
  
  # --------------------------------------------------------------------------
  # Fit Unpenalized Fine-Gray Model (FINAL MODEL)
  # --------------------------------------------------------------------------
  # Purpose: Estimate effects of clinical variables + MRS on competing risk outcome
  # No regularization - all variables enter the model
  
  cat(sprintf("\n========== UNPENALIZED FINE-GRAY (FINAL MODEL) ==========\n"))
  
  results <- fit_fine_gray_model(
    fgr_data = fgr_train_data,
    cr_time = "time_to_CompRisk_event",
    cr_event = "CompRisk_event_coded",
    cause = 1
  )
  
  fgr_final <- results$model
  
  cat(sprintf("  Final model fitted with %d predictors\n", length(fgr_final$crrFit$coef)))
  
  # --------------------------------------------------------------------------
  # Evaluate Final Model Performance on Test Set
  # --------------------------------------------------------------------------
  
  cat(sprintf("\n--- Test Set Performance ---\n"))
  
  score_final <- Score(
    list("FGR_Final" = fgr_final),
    formula = Hist(time_to_CompRisk_event, CompRisk_event_coded) ~ 1,
    data = fgr_test_data,
    cause = 1,
    times = EVAL_TIMES,
    metrics = c("auc", "brier"),
    summary = c("ibs"),
    se.fit = TRUE,
    conf.int = 0.95,
    null.model = TRUE,
    cens.model = "cox"
  )
  
  # Extract performance metrics
  auc_final <- score_final$AUC$score[model == "FGR_Final"]
  brier_final <- score_final$Brier$score[model == "FGR_Final"]
  
  eval_times <- auc_final$times
  
  # Create performance dataframe
  auc_cols <- setNames(auc_final$AUC, paste0("auc_", eval_times, "yr"))
  brier_cols <- setNames(brier_final$Brier, paste0("brier_", eval_times, "yr"))
  
  fgr_performance <- data.frame(
    model = "FGR_Final",
    mean_auc = mean(auc_final$AUC, na.rm = TRUE),
    final_ibs = brier_final[times == max(times), IBS],
    as.list(auc_cols),
    as.list(brier_cols)
  )
  
  fgr_performance[, -1] <- round(fgr_performance[, -1], 2)
  
  print(fgr_performance)
  
  # --------------------------------------------------------------------------
  # Extract Predictions on Test Set
  # --------------------------------------------------------------------------
  
  cat(sprintf("\n--- Extracting Test Set Predictions ---\n"))
  
  pred_risks <- predictRisk(
    fgr_final,
    newdata = fgr_test_data,
    times = EVAL_TIMES,
    cause = 1
  )
  
  # Combine with outcomes and MRS
  fold_predictions <- data.frame(
    fold = fold_idx,
    sample = rownames(fgr_test_data),
    time = fgr_test_data$time_to_CompRisk_event,
    event_coded = fgr_test_data$CompRisk_event_coded,
    rfi_event = clinical_test$RFi_event,
    methylation_risk_score = MRS_test,
    pred_risks
  )
  
  colnames(fold_predictions)[7:ncol(fold_predictions)] <- paste0("risk_", EVAL_TIMES, "yr")
  
  cat(sprintf("  Predictions extracted for %d test samples at %d time points\n",
              nrow(fold_predictions), length(EVAL_TIMES)))
  
  # --------------------------------------------------------------------------
  # Extract and Store Coefficients (kept separate by model type)
  # --------------------------------------------------------------------------
  
  cat(sprintf("\n--- Extracting Model Coefficients ---\n"))
  
  # ========== 1. PENALIZED FG: CpG coefficients (used to build MRS) ==========
  coef_pen_fg_df <- data.frame(
    feature = selected_cpgs_for_mrs,
    pen_fg_coef = selected_cpg_coefs,
    pen_fg_HR = exp(selected_cpg_coefs)
  )
  rownames(coef_pen_fg_df) <- NULL
  
  cat(sprintf("\n  Penalized FG (CpG coefficients for MRS):\n"))
  cat(sprintf("    %d CpGs in Model\n", nrow(coef_pen_fg_df)))
  print(coef_pen_fg_df)
  
  # ========== 2. FINAL FG: Clinical + MRS coefficients ==========
  fg_final_coef <- fgr_final$crrFit$coef
  coef_final_fg_df <- data.frame(
    feature = names(fg_final_coef),
    final_fg_coef = as.vector(fg_final_coef),
    final_fg_HR = exp(as.vector(fg_final_coef))
  )
  rownames(coef_final_fg_df) <- NULL
  
  cat(sprintf("\n  Final Unpenalized FG (Clinical + MRS):\n"))
  print(coef_final_fg_df)
  
  # ========== 3. COXNET: Clinical + CpG coefficients (for comparison) ==========
  
  names(coef_rfi_df) <- c("feature", "cox_rfi_coef")
  names(coef_death_df) <- c("feature", "cox_death_coef")
  # Full outer join to see all features from both Cox models
  coef_coxnet_comparison <- merge(coef_rfi_df, coef_death_df, 
                                  by = "feature", all = TRUE)
  # Keep NAs as they indicate non-selection (don't replace with 0)
  # Add HRs (only for non-NA coefficients)
  coef_coxnet_comparison$cox_rfi_HR <- ifelse(
    is.na(coef_coxnet_comparison$cox_rfi_coef),
    NA,
    exp(coef_coxnet_comparison$cox_rfi_coef)
  )
  coef_coxnet_comparison$cox_death_HR <- ifelse(
    is.na(coef_coxnet_comparison$cox_death_coef),
    NA,
    exp(coef_coxnet_comparison$cox_death_coef)
  )
  coef_coxnet_comparison <- coef_coxnet_comparison[c(
    "feature",
    "cox_rfi_coef",
    "cox_rfi_HR",
    "cox_death_coef",
    "cox_death_HR"
  )]
  rownames(coef_coxnet_comparison) <- NULL
  
  # --------------------------------------------------------------------------
  # Store Fold Results (UPDATED)
  # --------------------------------------------------------------------------
  
  fold_runtime <- as.numeric(difftime(Sys.time(), fold_start_time, units = "mins"))
  
  outer_fold_results[[fold_idx]] <- list(
    
    # ========== FOLD METADATA ==========
    fold_idx = fold_idx,
    train_samples = rownames(X_train),
    test_samples = rownames(X_test),
    n_train = nrow(X_train),
    n_test = nrow(X_test),
    fold_runtime = fold_runtime,
    
    # ========== COXNET FEATURE SELECTION ==========
    features_rfi = features_rfi,
    features_death = features_death,
    features_pooled = features_pooled,
    
    # CoxNet models and their coefficients
    coxnet_rfi_model = rfi_model$final_fit,
    coxnet_rfi_cv_result = rfi_model$best_result,
    coxnet_death_model = death_model$final_fit,
    coxnet_death_cv_result = death_model$best_result,
    coxnet_coefficients = coef_coxnet_comparison,  # comparison of both Cox models
    
    # ========== PENALIZED FINE-GRAY (MRS CONSTRUCTION) ==========
    mrs_selected_cpgs = selected_cpgs_for_mrs,
    mrs_cpg_coefficients = selected_cpg_coefs,
    
    penalized_fg_best_alpha = pFG_res$best_alpha,
    penalized_fg_best_lambda = pFG_res$best_lambda,
    penalized_fg_cv_result = pFG_res,
    penalized_fg_coefficients = coef_pen_fg_df,  # CpG-level coefficients
    
    mrs_train = MRS_train,
    mrs_test = MRS_test,
    
    # ========== FINAL UNPENALIZED FINE-GRAY MODEL ==========
    finegray_final_model = fgr_final,
    final_fg_coefficients = coef_final_fg_df,  # Clinical + MRS coefficients
    
    # Performance and predictions
    performance_df = fgr_performance,
    fold_predictions = fold_predictions,
    
    # ========== SCALING PARAMETERS ==========
    scale_params = list(
      meth_scale = meth_scale,
      clin_scale = clin_scale,
      mrs_scale = if(scale_mrs) mrs_scale else NULL
    )
  )
  cat(sprintf("\n✓ Fold %d completed in %.1f minutes\n", fold_idx, fold_runtime))
  
}

################################################################################
# SAVE RAW CV RESULTS (SAFETY BACKUP)
################################################################################

save(outer_fold_results, file = file.path(current_output_dir, 
                                          "outer_fold_results.RData"))
cat("Outer fold results saved (backup)\n")

################################################################################
# AGGREGATE CV RESULTS
################################################################################

perf_results <- aggregate_cv_performance(outer_fold_results)

stability_results <- assess_finegray_stability(
  outer_fold_results
)

################################################################################
# SAVE COMPLETE CV RESULTS (OVERWRITES BACKUP)
################################################################################

cv_results <- list(
  outer_fold_results = outer_fold_results,
  perf_results = perf_results,
  stability_results = stability_results,
  metadata = list(
    N_OUTER_FOLDS = N_OUTER_FOLDS,
    EVAL_TIMES = EVAL_TIMES,
    clinvars = clinvars,
    COHORT_NAME = COHORT_NAME,
    DATA_MODE = DATA_MODE
  )
)

save(cv_results, file = file.path(current_output_dir, 
                                  "outer_fold_results.RData"))
cat("Outer fold results saved (complete with aggregations)\n")


#
#load("~/PhD_Workspace/PredictRecurrence/output/FineGray/ERpHER2n/Methylation/Unadjusted/outer_fold_results.RData")
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
STABILITY_THRESHOLD_FG = 0.6
cat(sprintf("\n--- Collecting CV-Discovered Features ---\n"))

stability_metrics <- stability_results$stability_metrics

# Select stable non-clinical features
stable_features <- stability_metrics$feature[
    stability_metrics$selection_freq >= STABILITY_THRESHOLD_FG &
    stability_metrics$direction_consistent == TRUE
]

# Always include clinical variables
stable_features <- c(
  stability_metrics$feature[stability_metrics$is_clinical == TRUE], 
  stable_features
)

X_pooled_all <- X_all[, stable_features, drop = FALSE]

# ----------------------------------------------------------------------------
# Scale Features
# ----------------------------------------------------------------------------

cat(sprintf("\n--- Scaling Continuous Features ---\n"))

scale_res_all <- scale_continuous_features(
  X_train = X_pooled_all, 
  X_test = NULL, 
  dont_scale = if (DATA_MODE == "methylation") NULL else encoded_result$encoded_cols
)

X_all_scaled <- scale_res_all$X_train_scaled

# ----------------------------------------------------------------------------
# Fit Final Fine-Gray Model
# ----------------------------------------------------------------------------

cat(sprintf("\n--- Fitting Fine-Gray Model ---\n"))

fgr_all_data <- cbind(
  clinical_all[c("time_to_CompRisk_event","CompRisk_event_coded")], 
  X_all_scaled
)

feature_cols <- setdiff(
  colnames(fgr_all_data), 
  c("time_to_CompRisk_event", "CompRisk_event_coded")
)

formula_str <- paste(
  "Hist(time_to_CompRisk_event, CompRisk_event_coded) ~", 
  paste(feature_cols, collapse = " + ")
)
formula_fg <- as.formula(formula_str)

fgr_final <- FGR(
  formula = formula_fg,
  data = fgr_all_data,
  cause = 1
)

cat(sprintf("Fine-Gray model fitted with %d features\n", length(feature_cols)))
print(fgr_final)

# ----------------------------------------------------------------------------
# Calculate Variable Importance
# ----------------------------------------------------------------------------

cat("\n--- Fine-Gray Variable Importance ---\n")

vimp_fg_final <- calculate_fgr_importance(
  fgr_model = fgr_final,
  encoded_cols = if (DATA_MODE == "methylation") NULL else encoded_result$encoded_cols,
  verbose = FALSE
)

print(vimp_fg_final)

################################################################################
# SAVE FINAL MODEL RESULTS
################################################################################

final_results <- list(
  fgr_final = fgr_final,
  vimp_fg_final = vimp_fg_final,
  metadata = list(
    N_OUTER_FOLDS = N_OUTER_FOLDS,
    EVAL_TIMES = EVAL_TIMES,
    clinvars = clinvars,
    COHORT_NAME = COHORT_NAME,
    DATA_MODE = DATA_MODE
  )
)

save(final_results, file = file.path(current_output_dir, 
                                     "final_fg_results.RData"))
cat("Final model results saved\n")



################################################################################
# CLEANUP AND FINISH
################################################################################

total_runtime <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))

cat(sprintf("\n%s\n", paste(rep("=", 80), collapse = "")))
cat(sprintf("PIPELINE COMPLETED\n"))
cat(sprintf("Total runtime: %.1f minutes\n", total_runtime))
cat(sprintf("Finished: %s\n", Sys.time()))
cat(sprintf("%s\n", paste(rep("=", 80), collapse = "")))

# Close log file
sink(type = "message")
sink(type = "output")
close(log_con)
