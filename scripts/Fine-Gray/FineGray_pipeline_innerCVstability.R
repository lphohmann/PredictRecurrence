#!/usr/bin/env Rscript

################################################################################
# Fine-Gray Competing Risks with Nested Cross-Validation
# Author: Lennart Hohmann
#
## PIPELINE OVERVIEW:
# ==================

################################################################################

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

setwd("~/PhD_Workspace/PredictRecurrence/")
source("./src/finegray_functions.R") # match real name

################################################################################
# SETTINGS - Command Line Arguments
################################################################################

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Set defaults
DEFAULT_COHORT <- "ERpHER2n"
DEFAULT_DATA_MODE <- "combined" #combined
DEFAULT_TRAIN_CPGS <- "./data/set_definitions/CpG_prefiltered_sets/cpg_ids_atac_overlap.txt"
DEFAULT_OUTPUT_DIR <- "./output/FineGray"

# Parse arguments or use defaults
if (length(args) == 0) {
  # No arguments - use defaults
  cat("No command line arguments provided. Using defaults.\n")
  COHORT_NAME <- DEFAULT_COHORT
  DATA_MODE <- DEFAULT_DATA_MODE
  TRAIN_CPGS <- DEFAULT_TRAIN_CPGS
  OUTPUT_BASE_DIR <- DEFAULT_OUTPUT_DIR
  
} else if (length(args) == 2) {
  # Two arguments - cohort and data mode
  COHORT_NAME <- args[1]
  DATA_MODE <- args[2]
  TRAIN_CPGS <- DEFAULT_TRAIN_CPGS      # Use default
  OUTPUT_BASE_DIR <- DEFAULT_OUTPUT_DIR  # Use default
  
#} else if (length(args) == 4) {
#  # All four arguments provided (for backward compatibility)
#  COHORT_NAME <- args[1]
#  DATA_MODE <- args[2]
#  TRAIN_CPGS <- args[3]
#  OUTPUT_BASE_DIR <- args[4]
  
} else {
  # Wrong number of arguments - show usage and exit
  cat("\n=== USAGE ===\n")
  cat("Rscript finegray_pipeline.R <COHORT> <DATA_MODE> <TRAIN_CPGS> <OUTPUT_DIR>\n\n")
  cat("Arguments:\n")
  cat("  COHORT      : 'TNBC', 'ERpHER2n', or 'All'\n")
  cat("  DATA_MODE   : 'methylation' or 'combined'\n")
  cat("  TRAIN_CPGS  : Path to CpG IDs file (or 'NULL' to use all)\n")
  cat("  OUTPUT_DIR  : Base output directory\n\n")
  cat("Example:\n")
  cat("  Rscript finegray_pipeline.R ERpHER2n methylation ./data/cpg_ids.txt ./output/FineGray\n\n")
  cat("Or run without arguments to use defaults:\n")
  cat("  Rscript finegray_pipeline.R\n\n")
  stop("Incorrect number of arguments provided.")
}

# Validate COHORT_NAME
if (!COHORT_NAME %in% c("TNBC", "ERpHER2n", "All")) {
  stop(sprintf("Invalid COHORT_NAME: '%s'. Must be 'TNBC', 'ERpHER2n', or 'All'", COHORT_NAME))
}

# Validate DATA_MODE
if (!DATA_MODE %in% c("methylation", "combined")) {
  stop(sprintf("Invalid DATA_MODE: '%s'. Must be 'methylation' or 'combined'. For clinical-only, use finegray_clinical.R", DATA_MODE))
}

# Handle NULL for TRAIN_CPGS
if (TRAIN_CPGS == "NULL") {
  TRAIN_CPGS <- NULL
}

# Print settings
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
script_name <- "finegray_pipeline_stability"
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

# Feature selection - variance filter only
VARIANCE_QUANTILE <- 0.75  # Keep top 25% most variable features

# Variables to not penalize in elastic net (after one-hot encoding)
# These get coefficient shrinkage but not eliminated (penalty_factor = 0)

# Cross-validation
N_OUTER_FOLDS <- 5#5
N_INNER_FOLDS <- 5
ALPHA_GRID <- c(0.9)#c(0.5,0.7,0.9)  # Elastic net mixing: 0=ridge, 1=lasso, 0.9=mostly lasso

# Performance evaluation timepoints (in years)
EVAL_TIMES <- seq(1, 10)

################################################################################
# LOAD AND PREPARE DATA
################################################################################

start_time <- Sys.time()
cat(sprintf("\n========== LOADING DATA ==========\n"))

# Load data
train_ids <- read_csv(COHORT_TRAIN_IDS_PATHS[[COHORT_NAME]], col_names = FALSE, show_col_types = FALSE)[[1]]

data_list <- load_training_data(train_ids, INFILE_METHYLATION, INFILE_CLINICAL)
beta_matrix <- data_list$beta_matrix
clinical_data <- data_list$clinical_data

# Convert to M-values for better statistical properties
mvals <- beta_to_m(beta_matrix, beta_threshold = 0.001)

# Apply administrative censoring if needed
if (!is.null(ADMIN_CENSORING_CUTOFF)) {
  clinical_data <- apply_admin_censoring(
    clinical_data, "RFi_years", "RFi_event", ADMIN_CENSORING_CUTOFF
  )
}

# Subset to predefined CpGs
if (!is.null(TRAIN_CPGS)) {
  mvals <- subset_methylation(mvals, TRAIN_CPGS)
}

# Combine methylation with clinical variables based on DATA_MODE
if (DATA_MODE == "methylation") {
  # Methylation only - no clinical variables
  X <- mvals
  clinvars <- NULL
  encoded_result <- list(encoded_cols = NULL)  # Create empty for compatibility
  cat(sprintf("Using methylation features only: %d CpGs\n", ncol(X)))
  
} else if (DATA_MODE == "combined") {
  # Combined - methylation + clinical
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

# Variables to preserve during filtering (all clinical variables)
VARS_PRESERVE <- if (DATA_MODE == "methylation") NULL else clinvars
VARS_NO_PENALIZATION <- if (DATA_MODE == "methylation") NULL else clinvars  # Set to NULL to penalize all features

# Validate preserved and non-penalized variables
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
# CREATE COMPETING RISKS OUTCOMES
################################################################################

cat(sprintf("\n========== DEFINING OUTCOMES ==========\n"))

# Create outcomes for competing risks framework
# Event = 1: Recurrence (outcome of interest)
# Event = 2: Death without recurrence (competing risk)
# Event = 0: Censored

clinical_data$DeathNoR_event <- as.integer(
  clinical_data$OS_event == 1 & clinical_data$RFi_event == 0
)
clinical_data$DeathNoR_years <- clinical_data$OS_years

clinical_data$CompRisk_event_coded <- 0
clinical_data$CompRisk_event_coded[clinical_data$RFi_event == 1] <- 1
clinical_data$CompRisk_event_coded[clinical_data$DeathNoR_event == 1] <- 2

# Time to event: minimum of RFI time and death time
clinical_data$time_to_CompRisk_event <- pmin(
  clinical_data$RFi_years, 
  clinical_data$DeathNoR_years
)

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

# Create survival objects for CoxNet models
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
  
  # --------------------------------------------------------------------------
  # INNER CV LOOP #1: CoxNet for RFI
  # --------------------------------------------------------------------------
  
  cat(sprintf("\n========== COXNET FOR RFI ==========\n"))
  
  X_train_filtered_rfi <- prepare_filtered_features(
    X = X_train,
    vars_preserve = VARS_PRESERVE,
    variance_quantile = VARIANCE_QUANTILE)
  
  penalty_factor_rfi <- rep(1, ncol(X_train_filtered_rfi))
  if (!is.null(VARS_NO_PENALIZATION)) {
    penalty_factor_rfi[colnames(X_train_filtered_rfi) %in% VARS_NO_PENALIZATION] <- 0
  }
  
  rfi_model <- tune_and_fit_coxnet(
    X_train = X_train_filtered_rfi,
    y_train = y_rfi_train,
    clinical_train = clinical_train,
    event_col = "RFi_event",
    alpha_grid = ALPHA_GRID,
    penalty_factor = penalty_factor_rfi,
    n_inner_folds = N_INNER_FOLDS,
    outcome_name = "RFI",
    compute_stability = TRUE  # ENABLE STABILITY TRACKING
  )
  
  # Extract coefficients and features from final model
  rfi_coef_res <- extract_nonzero_coefs(rfi_model$final_fit)
  coef_rfi_df <- rfi_coef_res$coef_df
  features_rfi <- rfi_coef_res$features
  
  # ==========================================================================
  # OPTIONAL STABILITY FILTERING FOR RFI
  # Uncomment this block to filter features based on stability
  # ==========================================================================
  
  # Filter by selection frequency AND sign consistency
  # This ensures features are not only frequently selected, but also have
  # consistent coefficient direction (always positive or always negative)
  stability_threshold <- 1.0
  stability_info_rfi <- rfi_model$best_result$stability_info

  
  stable_nonclinical_rfi <- stability_info_rfi$feature[
    !(stability_info_rfi$feature %in% VARS_NO_PENALIZATION) &
      stability_info_rfi$selection_freq >= stability_threshold &
      stability_info_rfi$sign_consistent == TRUE
  ]
  
  # Always include ALL clinical variables + stable non-clinical features
  features_rfi <- c(VARS_NO_PENALIZATION, stable_nonclinical_rfi)

  if (length(stable_nonclinical_rfi) > 0) {
    cat(sprintf("  Applied stability + sign filter (≥%.0f%% & consistent): %d → %d features\n",
                stability_threshold * 100,
                length(rfi_model$best_result$features),
                length(features_rfi)))
    cat("    Filtered features:", paste(features_rfi, collapse = ", "), "\n")
  } else {
    cat("  Warning: No non clinical features met stability+sign criteria\n")
  }
  
  # ==========================================================================
  
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
  
  cat(sprintf("\n========== COXNET FOR DEATH w/o RECURRENCE ==========\n"))
  
  X_train_filtered_death <- prepare_filtered_features(
    X = X_train,
    vars_preserve = VARS_PRESERVE,
    variance_quantile = VARIANCE_QUANTILE)
  
  penalty_factor_death <- rep(1, ncol(X_train_filtered_death))
  if (!is.null(VARS_NO_PENALIZATION)) {
    penalty_factor_death[colnames(X_train_filtered_death) %in% VARS_NO_PENALIZATION] <- 0
  }
  
  death_model <- tune_and_fit_coxnet(
    X_train = X_train_filtered_death,
    y_train = y_death_train,
    clinical_train = clinical_train,
    event_col = "DeathNoR_event",
    alpha_grid = ALPHA_GRID,
    penalty_factor = penalty_factor_death,
    n_inner_folds = N_INNER_FOLDS,
    outcome_name = "DeathNoR",
    compute_stability = TRUE  # ENABLE STABILITY TRACKING
  )
  
  # Extract coefficients and features from final model
  death_coef_res <- extract_nonzero_coefs(death_model$final_fit)
  coef_death_df <- death_coef_res$coef_df
  features_death <- death_coef_res$features
  
  # ==========================================================================
  # OPTIONAL STABILITY FILTERING FOR DEATH
  # Uncomment this block to filter features based on stability
  # ==========================================================================
  
  # Filter by selection frequency AND sign consistency
  stability_threshold <- 1.0
  stability_info_death <- death_model$best_result$stability_info

  stable_nonclinical_death <- stability_info_death$feature[
    !(stability_info_death$feature %in% VARS_NO_PENALIZATION) &
      stability_info_death$selection_freq >= stability_threshold &
      stability_info_death$sign_consistent == TRUE
  ]
  
  # Always include ALL clinical variables + stable non-clinical features
  features_death <- c(VARS_NO_PENALIZATION, stable_nonclinical_death)
  
  if (length(stable_nonclinical_death) > 0) {
    cat(sprintf("  Applied stability + sign filter (≥%.0f%% & consistent): %d → %d features\n",
                stability_threshold * 100,
                length(death_model$best_result$features),
                length(features_death)))
    cat("    Filtered features:", paste(features_death, collapse = ", "), "\n")
  } else {
    cat("  Warning: No non clinical features met stability+sign criteria\n")
  }
  
  # ==========================================================================
  
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
  
  if (length(features_pooled) == 0) {
    warning(sprintf("Fold %d: No features selected. Skipping fold.", fold_idx))
    next
  }
  
  # Prepare Fine-Gray input
  X_pooled_train <- X_train[, features_pooled, drop = FALSE]
  X_pooled_test <- X_test[, features_pooled, drop = FALSE]
  
  # --------------------------------------------------------------------------
  # Scale Input Data for Fine-Gray Model
  # --------------------------------------------------------------------------
  # WHY: Fine-Gray model is sensitive to feature scales
  # Only scale continuous variables, leave one-hot encoded variables as-is
  
  cat(sprintf("\n--- Scaling Continuous Features ---\n"))
  scale_res <- scale_continuous_features(
    X_train = X_pooled_train, 
    X_test = X_pooled_test, 
    dont_scale = if (DATA_MODE == "methylation") NULL else encoded_result$encoded_cols
  )
  
  X_train_scaled <- scale_res$X_train_scaled
  X_test_scaled <- scale_res$X_test_scaled
  
  # --------------------------------------------------------------------------
  # Fit Fine-Gray Model on Pooled Features
  # --------------------------------------------------------------------------
  # Fine-Gray model: Models subdistribution hazard for event of interest
  
  cat(sprintf("\n--- Fitting Fine-Gray Model ---\n"))
  #identical(clinical_train$Sample,rownames(X_train_scaled))
  #identical(clinical_test$Sample,rownames(X_test_scaled))
  
  fgr_train_data <- cbind(
    clinical_train[c("time_to_CompRisk_event","CompRisk_event_coded")], 
    X_train_scaled
  )
  fgr_test_data <- cbind(
    clinical_test[c("time_to_CompRisk_event","CompRisk_event_coded")], 
    X_test_scaled
  )
  
  feature_cols <- setdiff(colnames(fgr_train_data), 
                          c("time_to_CompRisk_event", "CompRisk_event_coded"))
  
  # Build formula explicitly
  formula_str <- paste("Hist(time_to_CompRisk_event, CompRisk_event_coded) ~", 
                       paste(feature_cols, collapse = " + "))
  formula_fg <- as.formula(formula_str)
  
  fgr1 <- FGR(
    formula = formula_fg,
    data    = fgr_train_data,
    cause   = 1
  )
  
  cat(sprintf("Fine-Gray model fitted with %d features\n", length(feature_cols)))
  
  # --------------------------------------------------------------------------
  # Evaluate Fine-Gray Model Performance on Test Set
  # --------------------------------------------------------------------------
  
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
  
  # Extract FGR model results only (exclude null model)
  auc_fgr <- score_fgr1$AUC$score[model == "FGR"]
  brier_fgr <- score_fgr1$Brier$score[model == "FGR"]
  
  # Extract actual evaluation times
  eval_times <- auc_fgr$times
  
  # Create named vectors dynamically based on actual times
  auc_cols <- setNames(auc_fgr$AUC, paste0("auc_", eval_times, "yr"))
  brier_cols <- setNames(brier_fgr$Brier, paste0("brier_", eval_times, "yr"))
  
  # Combine into one-row dataframe
  fgr_performance <- data.frame(
    model = "FGR",
    mean_auc = mean(auc_fgr$AUC, na.rm = TRUE),
    final_ibs = brier_fgr[times == max(times), IBS],
    as.list(auc_cols),
    as.list(brier_cols)
  )
  
  # Round all numeric columns to 2 decimal places
  fgr_performance[, -1] <- round(fgr_performance[, -1], 2)
  
  print(fgr_performance)
  
  # --------------------------------------------------------------------------
  # Extract Predictions on Test Set
  # --------------------------------------------------------------------------
  
  pred_risks <- predictRisk(
    fgr1,
    newdata = X_test_scaled,
    times = EVAL_TIMES,
    cause = 1
  )
  
  # Combine with outcomes
  fold_predictions <- data.frame(
    fold = fold_idx,
    sample = rownames(X_test_scaled),
    time = clinical_test$time_to_CompRisk_event,
    event_coded = clinical_test$CompRisk_event_coded,
    rfi_event = clinical_test$RFi_event,
    pred_risks
  )
  
  colnames(fold_predictions)[6:ncol(fold_predictions)] <- paste0("risk_", EVAL_TIMES, "yr")
  
  # --------------------------------------------------------------------------
  # example predict on new data
  # --------------------------------------------------------------------------
  #  To apply model to new data later:
  # X_new_scaled <- X_new
  # X_new_scaled[, continuous_cols] <- scale(
  #   X_new[, continuous_cols, drop = FALSE],
  #   center = scale_centers,  # Use TRAINING mean
  #   scale = scale_scales      # Use TRAINING SD
  # )
  
  # # predict risks at times
  # predicted_risks <- predictRisk(
  #   object = fgr1,
  #   newdata = new_patients,
  #   times = c(1, 3, 5, 10),
  #   cause = 1
  # )
  # # simple risk score
  # coefs <- fgr1$crrFit$coef
  # linear_predictor <- as.matrix(new_patients[, names(coefs)]) %*% coefs
  
  # --------------------------------------------------------------------------
  # Extract and Display Coefficients
  # --------------------------------------------------------------------------
  
  # Extract FGR coefficients
  fg_coef <- fgr1$crrFit$coef
  coef_fg_df <- data.frame(
    feature = names(fg_coef),
    fg_coef = as.vector(fg_coef)
  )
  names(coef_rfi_df) <- c("feature","cox_rfi_coef")
  names(coef_death_df) <- c("feature","cox_death_coef")
  
  # Merge all three
  coef_comparison <- merge(coef_fg_df, coef_rfi_df, by = "feature")# ,all = TRUE)
  coef_comparison <- merge(coef_comparison, coef_death_df, by = "feature")#, all = TRUE)
  coef_comparison[is.na(coef_comparison)] <- 0
  
  coef_comparison$fg_HR <- exp(coef_comparison$fg_coef)
  coef_comparison <- coef_comparison[c("feature","cox_rfi_coef",
                                       "cox_death_coef","fg_coef",
                                       "fg_HR")]
  
  # Sort by absolute FG coefficient
  coef_comparison <- coef_comparison[order(abs(coef_comparison$fg_coef), decreasing = TRUE), ]
  rownames(coef_comparison) <- NULL
  
  print(coef_comparison)
  
  # --------------------------------------------------------------------------
  # Store Fold Results
  # --------------------------------------------------------------------------
  
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
    coxnet_rfi_complete_results = rfi_model, #rfi_model$cv_results$`0.9`$stability_matrix
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
    
    # Scaling parameters (for future use)
    scale_params = scale_res[c("centers","scales","cont_cols")]
  )
}

################################################################################
# SAVE RAW CV RESULTS (SAFETY BACKUP)
################################################################################

save(outer_fold_results, file = file.path(current_output_dir, "outer_fold_results.RData"))
cat("outer_fold_results results saved (backup)\n")

################################################################################
# AGGREGATE RESULTS ACROSS FOLDS
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
save(cv_results, file = file.path(current_output_dir, "outer_fold_results.RData"))
cat("outer_fold_results results saved (complete with aggregations)\n")

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

# --------------------------------------------------------------------------
# Collect Features Selected During CV
# --------------------------------------------------------------------------

cat(sprintf("\n--- Collecting CV-Discovered Features ---\n"))

stability_metrics <- stability_results$stability_metrics
stable_features <- stability_metrics$feature[
  stability_metrics$is_clinical == FALSE &
  stability_metrics$selection_freq >= 0.8 &
    stability_metrics$direction_consistent == TRUE]

# Always include ALL clinical variables + stable non-clinical features
stable_features <- c(stability_metrics$feature[
  stability_metrics$is_clinical == TRUE], stable_features)

X_pooled_all <- X_all[, stable_features, drop = FALSE]

# --------------------------------------------------------------------------
# Scale Features
# --------------------------------------------------------------------------

cat(sprintf("\n--- Scaling Continuous Features ---\n"))
scale_res_all <- scale_continuous_features(
  X_train = X_pooled_all, 
  X_test = NULL, 
  dont_scale = if (DATA_MODE == "methylation") NULL else encoded_result$encoded_cols
)

X_all_scaled <- scale_res_all$X_train_scaled

# --------------------------------------------------------------------------
# Fit Final Fine-Gray Model
# --------------------------------------------------------------------------

cat(sprintf("\n--- Fitting Fine-Gray Model ---\n"))

fgr_all_data <- cbind(
  clinical_all[c("time_to_CompRisk_event","CompRisk_event_coded")], 
  X_all_scaled
)

feature_cols <- setdiff(colnames(fgr_all_data), 
                        c("time_to_CompRisk_event", "CompRisk_event_coded"))
formula_str <- paste("Hist(time_to_CompRisk_event, CompRisk_event_coded) ~", 
                     paste(feature_cols, collapse = " + "))
formula_fg <- as.formula(formula_str)

fgr_final <- FGR(
  formula = formula_fg,
  data    = fgr_all_data,
  cause   = 1
)

cat(sprintf("Fine-Gray model fitted with %d features\n", length(feature_cols)))

print(fgr_final)
# --------------------------------------------------------------------------
# Calculate Variable Importance
# --------------------------------------------------------------------------

cat("\n--- Fine-Gray Variable Importance ---\n")

vimp_fg_final <- calculate_fgr_importance(
  fgr_model = fgr_final,
  encoded_cols = if (DATA_MODE == "methylation") NULL else encoded_result$encoded_cols,
  verbose = FALSE
)

print(vimp_fg_final)

# Save final model results
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
save(final_results, file = file.path(current_output_dir, "final_fg_results.RData"))