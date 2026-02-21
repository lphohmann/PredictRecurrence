#!/usr/bin/env Rscript

################################################################################
# Fine-Gray Competing Risks Model with Nested Cross-Validation
# Author: Lennart Hohmann
#
# PIPELINE OVERVIEW:
# ==================
# This pipeline implements a nested cross-validation framework for competing
# risks analysis using Fine-Gray subdistribution hazard models.
#
# Key Features:
# - Nested CV: Outer loop for performance estimation, inner loop for feature selection
# - Two-stage feature selection: CoxNet for dimension reduction → Fine-Gray for final model
# - Competing risks: Models both recurrence (event of interest) and death (competing event)
# - Stability-based feature filtering: Only retains features with consistent selection
# - Supports both methylation-only and combined (methylation + clinical) modes
#
# Workflow:
# 1. For each outer CV fold:
#    a. Run CoxNet on RFI endpoint with inner CV → select stable features
#    b. Run CoxNet on Death endpoint with inner CV → select stable features
#    c. Pool features from both endpoints
#    d. Train Fine-Gray model on pooled features
#    e. Evaluate on held-out test set
# 2. Aggregate performance across outer folds
# 3. Train final model on all data using CV-discovered stable features
################################################################################

################################################################################
# LIBRARIES
################################################################################
setwd("~/PhD_Workspace/PredictRecurrence/")

library(readr)          # Fast CSV reading
library(dplyr)          # Data manipulation
library(survival)       # Survival analysis (Surv objects, concordance)
library(glmnet)         # Elastic net Cox regression
library(cmprsk)         # Competing risks regression
library(caret)          # Cross-validation fold creation
library(data.table)     # Fast data reading with fread
library(riskRegression) # Score() function for competing risks metrics
library(prodlim)        # Hist() for competing risks

#setwd("~/PhD_Workspace/PredictRecurrence/")
source("./src/finegray_functions.R")

################################################################################
# COMMAND LINE ARGUMENTS
################################################################################

args <- commandArgs(trailingOnly = TRUE)

# Defaults
DEFAULT_COHORT <- "ERpHER2n"
DEFAULT_DATA_MODE <- "methylation"
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
# FILE PATHS
################################################################################

# Input files
INFILE_METHYLATION <- "./data/train/train_methylation_unadjusted.csv"
INFILE_CLINICAL <- "./data/train/train_clinical.csv"

COHORT_TRAIN_IDS_PATHS <- list(
  TNBC = "./data/train/train_subcohorts/TNBC_train_ids.csv",
  ERpHER2n = "./data/train/train_subcohorts/ERpHER2n_train_ids.csv",
  All = "./data/train/train_subcohorts/All_train_ids.csv"
)

# Output directory
current_output_dir <- file.path(
  OUTPUT_BASE_DIR, 
  COHORT_NAME, 
  tools::toTitleCase(DATA_MODE), 
  "Unadjusted"
)
dir.create(current_output_dir, recursive = TRUE, showWarnings = FALSE)

################################################################################
# LOGGING SETUP
################################################################################

run_timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")

script_name <- "finegray_pipeline_stability"
log_filename <- sprintf("%s_%s.log", script_name, run_timestamp)
log_path <- file.path(current_output_dir, log_filename)

# Start logging
log_con <- file(log_path, open = "wt")
sink(log_con, type = "output", split = TRUE)
sink(log_con, type = "message")

cat(sprintf("%s\n", paste(rep("=", 80), collapse = "")))
cat(sprintf("FINE-GRAY COMPETING RISKS PIPELINE\n"))
cat(sprintf("Started: %s\n", Sys.time()))
cat(sprintf("Cohort: %s\n", COHORT_NAME))
cat(sprintf("Data mode: %s\n", DATA_MODE))
cat(sprintf("Log file: %s\n", log_path))
cat(sprintf("%s\n\n", paste(rep("=", 80), collapse = "")))

################################################################################
# ANALYSIS PARAMETERS
################################################################################

# Administrative censoring
ADMIN_CENSORING_CUTOFF <- if (COHORT_NAME == "TNBC") 5.01 else NULL

# Clinical variables (cohort-dependent)
if (COHORT_NAME == "All") {
  CLIN_CATEGORICAL <- c("NHG", "LN", "ER", "PR", "HER2")
} else {
  CLIN_CATEGORICAL <- c("NHG", "LN")
}
CLIN_CONT <- c("Age", "Size.mm")

cat(sprintf("Clinical categorical variables: %s\n", paste(CLIN_CATEGORICAL, collapse=", ")))

# Feature filtering
VARIANCE_QUANTILE <- 0.75  # Keep top 25% most variable features


# inner cv stablity filter
STABILITY_THRESHOLD <- 0.8

# for final fg model
STABILITY_THRESHOLD_FG <- 0.4

# Cross-validation settings
N_OUTER_FOLDS <- 5
N_INNER_FOLDS <- 5
ALPHA_GRID <- c(0.9)  # Elastic net mixing parameter

# Performance evaluation
EVAL_TIMES <- seq(1, 10)  # Years for time-dependent metrics

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
VARS_PRESERVE <- if (DATA_MODE == "methylation") NULL else clinvars
VARS_NO_PENALIZATION <- if (DATA_MODE == "methylation") NULL else clinvars

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

# Create stratified outer folds
set.seed(123)
outer_folds <- createFolds(
  y = clinical_data$RFi_event,
  k = N_OUTER_FOLDS,
  list = TRUE,
  returnTrain = FALSE
)

outer_fold_results <- list()

################################################################################
# OUTER CROSS-VALIDATION LOOP
################################################################################

for (fold_idx in 1:N_OUTER_FOLDS) {
  cat(sprintf("\n========== OUTER FOLD %d/%d ==========\n", 
              fold_idx, N_OUTER_FOLDS))
  
  fold_start_time <- Sys.time()
  
  # ============================================================================
  # OUTER FOLD: DATA SPLITTING
  # ============================================================================
  
  test_idx <- outer_folds[[fold_idx]]
  train_idx <- setdiff(1:nrow(X), test_idx)
  
  X_train <- X[train_idx, , drop = FALSE]
  X_test <- X[test_idx, , drop = FALSE]
  
  y_rfi_train <- y_rfi[train_idx]
  y_rfi_test <- y_rfi[test_idx]
  
  y_death_train <- y_death_no_r[train_idx]
  y_death_test <- y_death_no_r[test_idx]
  
  clinical_train <- clinical_data[train_idx, ]
  clinical_test <- clinical_data[test_idx, ]
  
  cat(sprintf("Train: n=%d (RFi=%d, Death=%d)\n",
              nrow(X_train), 
              sum(clinical_train$RFi_event), 
              sum(clinical_train$DeathNoR_event)))
  cat(sprintf("Test: n=%d (RFi=%d, Death=%d)\n",
              nrow(X_test), 
              sum(clinical_test$RFi_event), 
              sum(clinical_test$DeathNoR_event)))
  
  # ============================================================================
  # INNER CV #1: COXNET FOR RFI (RECURRENCE)
  # ============================================================================
  
  cat(sprintf("\n========== COXNET FOR RFI ==========\n"))
  
  # Prepare features: variance filtering
  X_train_filtered_rfi <- prepare_filtered_features(
    X = X_train,
    vars_preserve = VARS_PRESERVE,
    variance_quantile = VARIANCE_QUANTILE
  )
  
  # Set penalty factors
  penalty_factor_rfi <- rep(1, ncol(X_train_filtered_rfi))
  if (!is.null(VARS_NO_PENALIZATION)) {
    penalty_factor_rfi[colnames(X_train_filtered_rfi) %in% VARS_NO_PENALIZATION] <- 0
  }
  
  # Run CoxNet with inner CV
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
  
  # Extract coefficients from final model
  rfi_coef_res <- extract_nonzero_coefs(rfi_model$final_fit)
  coef_rfi_df <- rfi_coef_res$coef_df
  features_rfi <- rfi_coef_res$features
  
  # --------------------------------------------------------------------------
  # STABILITY FILTERING FOR RFI
  # --------------------------------------------------------------------------
  
  #stability_threshold <- 1.0
  stability_info_rfi <- rfi_model$best_result$stability_info
  
  # Filter non-clinical features by stability + sign consistency
  stable_nonclinical_rfi <- stability_info_rfi$feature[
    !(stability_info_rfi$feature %in% VARS_NO_PENALIZATION) &
      stability_info_rfi$selection_freq >= STABILITY_THRESHOLD &
      stability_info_rfi$sign_consistent == TRUE
  ]
  
  # Always include clinical variables + stable non-clinical features
  features_rfi <- c(VARS_NO_PENALIZATION, stable_nonclinical_rfi)
  
  if (length(stable_nonclinical_rfi) > 0) {
    cat(sprintf("  Applied stability + sign filter (≥%.0f%% & consistent): %d → %d features\n",
                STABILITY_THRESHOLD * 100,
                length(rfi_model$best_result$features),
                length(features_rfi)))
    cat("    Filtered features:", paste(features_rfi, collapse = ", "), "\n")
  } else {
    cat("  Warning: No non-clinical features met stability+sign criteria\n")
  }
  
  cat(sprintf("  BEST RFI MODEL: Alpha=%.2f, Lambda=%.6f, Features=%d\n",
              rfi_model$best_alpha, 
              rfi_model$best_lambda, 
              length(features_rfi)))
  cat("    Selected:", paste(features_rfi, collapse=", "), "\n")
  
  # ============================================================================
  # INNER CV #2: COXNET FOR DEATH WITHOUT RECURRENCE
  # ============================================================================
  
  cat(sprintf("\n========== COXNET FOR DEATH w/o RECURRENCE ==========\n"))
  
  # Prepare features
  X_train_filtered_death <- prepare_filtered_features(
    X = X_train,
    vars_preserve = VARS_PRESERVE,
    variance_quantile = VARIANCE_QUANTILE
  )
  
  # Set penalty factors
  penalty_factor_death <- rep(1, ncol(X_train_filtered_death))
  if (!is.null(VARS_NO_PENALIZATION)) {
    penalty_factor_death[colnames(X_train_filtered_death) %in% VARS_NO_PENALIZATION] <- 0
  }
  
  # Run CoxNet with inner CV
  death_model <- tune_and_fit_coxnet(
    X_train = X_train_filtered_death,
    y_train = y_death_train,
    clinical_train = clinical_train,
    event_col = "DeathNoR_event",
    alpha_grid = ALPHA_GRID,
    penalty_factor = penalty_factor_death,
    n_inner_folds = N_INNER_FOLDS,
    outcome_name = "DeathNoR",
    compute_stability = TRUE
  )
  
  # Extract coefficients
  death_coef_res <- extract_nonzero_coefs(death_model$final_fit)
  coef_death_df <- death_coef_res$coef_df
  features_death <- death_coef_res$features
  
  # --------------------------------------------------------------------------
  # STABILITY FILTERING FOR DEATH
  # --------------------------------------------------------------------------
  
  #stability_threshold <- 1.0
  stability_info_death <- death_model$best_result$stability_info
  
  # Filter non-clinical features by stability + sign consistency
  stable_nonclinical_death <- stability_info_death$feature[
    !(stability_info_death$feature %in% VARS_NO_PENALIZATION) &
      stability_info_death$selection_freq >= STABILITY_THRESHOLD &
      stability_info_death$sign_consistent == TRUE
  ]
  
  # Always include clinical variables + stable non-clinical features
  features_death <- c(VARS_NO_PENALIZATION, stable_nonclinical_death)
  
  if (length(stable_nonclinical_death) > 0) {
    cat(sprintf("  Applied stability + sign filter (≥%.0f%% & consistent): %d → %d features\n",
                STABILITY_THRESHOLD * 100,
                length(death_model$best_result$features),
                length(features_death)))
    cat("    Filtered features:", paste(features_death, collapse = ", "), "\n")
  } else {
    cat("  Warning: No non-clinical features met stability+sign criteria\n")
  }
  
  cat(sprintf("  BEST DeathNoR MODEL: Alpha=%.2f, Lambda=%.6f, Features=%d\n",
              death_model$best_alpha, 
              death_model$best_lambda, 
              length(features_death)))
  cat("    Selected:", paste(features_death, collapse=", "), "\n")
  
  # ============================================================================
  # FEATURE POOLING
  # ============================================================================
  
  features_pooled <- union(features_rfi, features_death)
  
  cat(sprintf("\n--- Feature Pooling ---\n"))
  cat(sprintf("RFI features: %d\n", length(features_rfi)))
  cat(sprintf("Death features: %d\n", length(features_death)))
  cat(sprintf("Overlap: %d\n", length(intersect(features_rfi, features_death))))
  cat(sprintf("Pooled total: %d\n", length(features_pooled)))
  
  # Check if any features selected
  if (length(features_pooled) == 0) {
    warning(sprintf("Fold %d: No features selected. Skipping fold.", fold_idx))
    next
  }
  
  # Prepare pooled feature matrices
  X_pooled_train <- X_train[, features_pooled, drop = FALSE]
  X_pooled_test <- X_test[, features_pooled, drop = FALSE]
  
  # ============================================================================
  # FEATURE SCALING
  # ============================================================================
  
  cat(sprintf("\n--- Scaling Continuous Features ---\n"))
  
  scale_res <- scale_continuous_features(
    X_train = X_pooled_train, 
    X_test = X_pooled_test, 
    dont_scale = if (DATA_MODE == "methylation") NULL else encoded_result$encoded_cols
  )
  
  X_train_scaled <- scale_res$X_train_scaled
  X_test_scaled <- scale_res$X_test_scaled
  
  # ============================================================================
  # FINE-GRAY MODEL FITTING
  # ============================================================================
  
  cat(sprintf("\n--- Fitting Fine-Gray Model ---\n"))
  
  # Prepare training data
  fgr_train_data <- cbind(
    clinical_train[c("time_to_CompRisk_event","CompRisk_event_coded")], 
    X_train_scaled
  )
  
  # Prepare test data
  fgr_test_data <- cbind(
    clinical_test[c("time_to_CompRisk_event","CompRisk_event_coded")], 
    X_test_scaled
  )
  
  # Build formula
  feature_cols <- setdiff(
    colnames(fgr_train_data), 
    c("time_to_CompRisk_event", "CompRisk_event_coded")
  )
  
  formula_str <- paste(
    "Hist(time_to_CompRisk_event, CompRisk_event_coded) ~", 
    paste(feature_cols, collapse = " + ")
  )
  formula_fg <- as.formula(formula_str)
  
  # Fit Fine-Gray model
  fgr1 <- FGR(
    formula = formula_fg,
    data = fgr_train_data,
    cause = 1
  )
  
  cat(sprintf("Fine-Gray model fitted with %d features\n", length(feature_cols)))
  
  # ============================================================================
  # MODEL EVALUATION
  # ============================================================================
  
  score_fgr1 <- Score(
    list("FGR" = fgr1),
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
  auc_fgr <- score_fgr1$AUC$score[model == "FGR"]
  brier_fgr <- score_fgr1$Brier$score[model == "FGR"]
  eval_times <- auc_fgr$times
  
  # Format performance
  auc_cols <- setNames(auc_fgr$AUC, paste0("auc_", eval_times, "yr"))
  brier_cols <- setNames(brier_fgr$Brier, paste0("brier_", eval_times, "yr"))
  
  fgr_performance <- data.frame(
    model = "FGR",
    mean_auc = mean(auc_fgr$AUC, na.rm = TRUE),
    final_ibs = brier_fgr[times == max(times), IBS],
    as.list(auc_cols),
    as.list(brier_cols)
  )
  
  fgr_performance[, -1] <- round(fgr_performance[, -1], 2)
  print(fgr_performance)
  
  # ============================================================================
  # EXTRACT PREDICTIONS
  # ============================================================================
  
  pred_risks <- predictRisk(
    fgr1,
    newdata = X_test_scaled,
    times = EVAL_TIMES,
    cause = 1
  )
  
  fold_predictions <- data.frame(
    fold = fold_idx,
    sample = rownames(X_test_scaled),
    time = clinical_test$time_to_CompRisk_event,
    event_coded = clinical_test$CompRisk_event_coded,
    rfi_event = clinical_test$RFi_event,
    pred_risks
  )
  
  colnames(fold_predictions)[6:ncol(fold_predictions)] <- paste0("risk_", EVAL_TIMES, "yr")
  
  # ============================================================================
  # COEFFICIENT COMPARISON
  # ============================================================================
  
  # Extract Fine-Gray coefficients
  fg_coef <- fgr1$crrFit$coef
  coef_fg_df <- data.frame(
    feature = names(fg_coef),
    fg_coef = as.vector(fg_coef)
  )
  
  # Rename CoxNet coefficient columns
  names(coef_rfi_df) <- c("feature","cox_rfi_coef")
  names(coef_death_df) <- c("feature","cox_death_coef")
  
  # Merge all coefficient sets (all = TRUE keeps all features)
  coef_comparison <- merge(coef_fg_df, coef_rfi_df, by = "feature", all = TRUE)
  coef_comparison <- merge(coef_comparison, coef_death_df, by = "feature", all = TRUE)
  coef_comparison[is.na(coef_comparison)] <- 0
  
  # Add hazard ratio
  coef_comparison$fg_HR <- exp(coef_comparison$fg_coef)
  coef_comparison <- coef_comparison[c("feature","cox_rfi_coef",
                                       "cox_death_coef","fg_coef","fg_HR")]
  
  # Sort by absolute Fine-Gray coefficient
  coef_comparison <- coef_comparison[order(abs(coef_comparison$fg_coef), decreasing = TRUE), ]
  rownames(coef_comparison) <- NULL
  
  # Print ONLY features in FG model
  cat(sprintf("\nFine-Gray Model Coefficients (%d features):\n", length(fg_coef)))
  print(coef_comparison[coef_comparison$fg_coef != 0, ])
  
  # ============================================================================
  # STORE FOLD RESULTS
  # ============================================================================
  
  fold_runtime <- as.numeric(difftime(Sys.time(), fold_start_time, units = "mins"))
  cat(sprintf("\n✓ Fold %d completed in %.1f minutes\n", fold_idx, fold_runtime))
  
  outer_fold_results[[fold_idx]] <- list(
    # Fold information
    fold_idx = fold_idx,
    train_samples = rownames(X_train),
    test_samples = rownames(X_test),
    n_train = nrow(X_train),
    n_test = nrow(X_test),
    fold_runtime = fold_runtime,
    
    # Feature selection results
    features_rfi = features_rfi,
    features_death = features_death,
    features_pooled = features_pooled,
    
    # CoxNet models and hyperparameters
    coxnet_rfi_complete_results = rfi_model,
    coxnet_rfi_model = rfi_model$final_fit,
    coxnet_rfi_cv_result = rfi_model$best_result,
    coxnet_death_complete_results = death_model,
    coxnet_death_model = death_model$final_fit,
    coxnet_death_cv_result = death_model$best_result,
    
    # Fine-Gray model
    finegray_model = fgr1,
    fold_model_coefficients = coef_comparison,
    performance_df = fgr_performance,
    fold_predictions = fold_predictions,
    
    # Scaling parameters
    scale_params = scale_res[c("centers","scales","cont_cols")]
  )
}

################################################################################
# SAVE RAW CV RESULTS (SAFETY BACKUP)
################################################################################

save(outer_fold_results, file = file.path(current_output_dir, 
                                          paste0("outer_fold_results_", run_timestamp, ".RData")))
cat("Outer fold results saved (backup)\n")

################################################################################
# AGGREGATE CV RESULTS
################################################################################

perf_results <- aggregate_cv_performance(outer_fold_results)

stability_results <- assess_finegray_stability(
  outer_fold_results, 
  clinical_features = if (DATA_MODE == "methylation") NULL else clinvars
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
                                          paste0("outer_fold_results_", run_timestamp, ".RData")))
cat("Outer fold results saved (complete with aggregations)\n")


#
#load("~/PhD_Workspace/PredictRecurrence/output/FineGray/ERpHER2n/Methylation/Unadjusted/outer_fold_results_20260130_092851.RData")
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
stable_features <- stability_metrics$feature[
  stability_metrics$is_clinical != TRUE &
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
                                     paste0("final_fg_results_", run_timestamp, ".RData")))
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