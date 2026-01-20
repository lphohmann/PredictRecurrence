#!/usr/bin/env Rscript

################################################################################
# Fine-Gray Competing Risks with Nested Cross-Validation
# Author: Lennart Hohmann
#
# PIPELINE OVERVIEW:
# ==================
# This pipeline implements nested cross-validation for competing risks analysis
# of breast cancer recurrence using DNA methylation + clinical variables.
#
# 1. DATA PREPARATION
#    - Load methylation beta values (CpG sites) and clinical data
#    - Convert beta → M-values: log2(beta/(1-beta)) for better normal distribution
#    - Apply administrative censoring if specified (e.g., 5 years for TNBC)
#    - One-hot encode categorical clinical variables (NHG, lymph nodes)
#
# 2. COMPETING RISKS DEFINITION
#    - Event 1: Recurrence-free interval (RFI) event - outcome of interest
#    - Event 2: Death without recurrence - competing risk
#    - Event 0: Censored
#
# 3. NESTED CROSS-VALIDATION STRATEGY
#    OUTER LOOP (Performance estimation):
#      - 5-fold stratified CV on RFI events
#      - For each fold:
#        a) Variance filtering on training data (top 75% by variance)
#        b) INNER CV: Tune CoxNet for RFI events
#        c) INNER CV: Tune CoxNet for death without recurrence
#        d) Pool selected features from both models
#        e) Scale continuous features (mean=0, sd=1)
#        f) Fit Fine-Gray subdistribution hazard model
#        g) Predict cumulative incidence functions (CIF) on test set
#        h) Evaluate time-dependent AUC, Brier score, C-index
#
# 4. WHY THIS APPROACH?
#    - CoxNet feature selection: Handles high-dimensional data (many CpGs)
#    - Separate models for RFI and death: Captures features relevant to each outcome
#    - Feature pooling: Ensures Fine-Gray model accounts for both outcomes
#    - Fine-Gray model: Properly models subdistribution hazard for competing risks
#    - Nested CV: Unbiased performance estimates (no data leakage)
#
# KEY STATISTICAL CONCEPTS:
# =========================
# - M-values: Logit-transformed beta values, more suitable for linear models
# - Subdistribution hazard: Hazard of event of interest accounting for competing risks
# - Cumulative Incidence Function (CIF): P(event by time t | baseline covariates)
# - IPCW: Inverse probability of censoring weighting for time-dependent metrics
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
source("./src/finegray_functions.R")

################################################################################
# SETTINGS
################################################################################

setwd("~/PhD_Workspace/PredictRecurrence/")

COHORT_NAME <- "ERpHER2n"
DATA_MODE <- "combined"
TRAIN_CPGS <- "./data/set_definitions/CpG_prefiltered_sets/cpg_ids_atac_overlap.txt"
OUTPUT_BASE_DIR <- "./output/FineGray_fastcmprsk"

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
# PARAMETERS
################################################################################

# Clinical variables - always preserved in filtering
ADMIN_CENSORING_CUTOFF <- if (COHORT_NAME == "TNBC") 5.01 else NULL
CLIN_CATEGORICAL <- c("NHG", "LN")
CLIN_CONT <- c("Age", "Size.mm")

# Feature selection - variance filter only
VARIANCE_QUANTILE <- 0.75  # Keep top 25% most variable features

# Variables to not penalize in elastic net (after one-hot encoding)
# These get coefficient shrinkage but not eliminated (penalty_factor = 0)
VARS_NO_PENALIZATION <- NULL

# Cross-validation
N_OUTER_FOLDS <- 5
N_INNER_FOLDS <- 3
ALPHA_GRID <- c(0.9)  # Elastic net mixing: 0=ridge, 1=lasso, 0.9=mostly lasso

# Performance evaluation timepoints (in years)
EVAL_TIMES <- seq(1, 10)


################################################################################
# LOAD AND PREPARE DATA
################################################################################

start_time <- Sys.time()
cat(sprintf("\n========== LOADING DATA ==========\n"))

# Load data
train_ids <- read_csv(
  COHORT_TRAIN_IDS_PATHS[[COHORT_NAME]], 
  col_names = FALSE, 
  show_col_types = FALSE
)[[1]]

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

# Combine methylation with clinical variables
X <- mvals

if (!is.null(c(CLIN_CONT, CLIN_CATEGORICAL))) {
  clin <- clinical_data[c(CLIN_CONT, CLIN_CATEGORICAL)]
  clin <- clin[rownames(X), , drop = FALSE]
  encoded_result <- onehot_encode_clinical(clin, CLIN_CATEGORICAL)
  X <- cbind(X, encoded_result$encoded_df)
  
  clinvars <- c(CLIN_CONT, encoded_result$encoded_cols)
  cat(sprintf("Added clinical variables: %s\n", 
              paste(clinvars, collapse = ", ")))
}

# Variables to preserve during filtering (all clinical variables)
VARS_PRESERVE <- clinvars

# Validate preserved and non-penalized variables
if (!is.null(VARS_PRESERVE)) {
  missing <- setdiff(VARS_PRESERVE, colnames(X))
  if (length(missing) > 0) {
    warning(sprintf("VARS_PRESERVE contains non-existent columns: %s", 
                    paste(missing, collapse = ", ")))
    VARS_PRESERVE <- intersect(VARS_PRESERVE, colnames(X))
  }
  cat(sprintf("Preserving %d variables from filtering.\n", length(VARS_PRESERVE)))
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
  clinical_data$OS_years
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
all_predictions <- data.frame()  # Will store all test set predictions

################################################################################
# OUTER CV LOOP - Performance Estimation
################################################################################

for (fold_idx in 1:N_OUTER_FOLDS) {
  cat(sprintf("\n========== OUTER FOLD %d/%d ==========\n", 
              fold_idx, N_OUTER_FOLDS))
  
  fold_start_time <- Sys.time()
  
  # --------------------------------------------------------------------------
  # Split data into outer train and test
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
  # Feature Filtering on Outer Training Data
  # --------------------------------------------------------------------------
  # WHY: Reduce dimensionality before elastic net
  # IMPORTANT: Filtering done ONLY on training data to prevent data leakage
  
  cat(sprintf("\n--- Feature Filtering ---\n"))
  
  # Separate preserved vs filterable features
  is_preserved <- colnames(X_train) %in% VARS_PRESERVE
  X_train_preserved <- X_train[, is_preserved, drop = FALSE]
  X_train_to_filter <- X_train[, !is_preserved, drop = FALSE]
  
  cat(sprintf("Features before filtering: %d (preserved: %d)\n", 
              ncol(X_train), sum(is_preserved)))
  
  # Variance filtering
  X_filtered <- filter_by_variance(X_train_to_filter, variance_quantile = VARIANCE_QUANTILE)
  
  # Combine filtered features with preserved features
  X_train_filtered <- cbind(X_filtered, X_train_preserved)
  X_test_filtered <- X_test[, colnames(X_train_filtered), drop = FALSE]
  
  cat(sprintf("Total features for modeling: %d\n", ncol(X_train_filtered)))
  
  # Create penalty factor (0 = not penalized, 1 = penalized)
  # Unpenalized features can still be shrunk but won't be eliminated
  penalty_factor <- rep(1, ncol(X_train_filtered))
  if (!is.null(VARS_NO_PENALIZATION)) {
    penalty_factor[colnames(X_train_filtered) %in% VARS_NO_PENALIZATION] <- 0
  }
  
  # --------------------------------------------------------------------------
  # INNER CV LOOP #1: CoxNet for RFI
  # --------------------------------------------------------------------------
  # PURPOSE: Select features predictive of recurrence
  # Uses Cox proportional hazards with elastic net regularization
  
  rfi_model <- tune_and_fit_coxnet(
    X_train = X_train_filtered,
    y_train = y_rfi_train,
    clinical_train = clinical_train,
    event_col = "RFi_event",
    alpha_grid = ALPHA_GRID,
    penalty_factor = penalty_factor,
    n_inner_folds = N_INNER_FOLDS,
    outcome_name = "RFI"
  )
  
  # Extract coefficients and features
  coef_rfi <- coef(rfi_model$final_fit)
  coef_rfi_df <- data.frame(
    feature = rownames(coef_rfi)[as.vector(coef_rfi != 0)],
    coefficient = as.vector(coef_rfi)[as.vector(coef_rfi != 0)])
  coef_rfi_df <- coef_rfi_df[order(abs(coef_rfi_df$coefficient), decreasing = TRUE), ]
  rownames(coef_rfi_df) <- NULL
  features_rfi <- coef_rfi_df$feature
  
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
  # PURPOSE: Select features predictive of death (competing risk)
  # This ensures we account for features that predict competing events
  death_model <- tune_and_fit_coxnet(
    X_train = X_train_filtered,
    y_train = y_death_train,
    clinical_train = clinical_train,
    event_col = "DeathNoR_event",
    alpha_grid = ALPHA_GRID,
    penalty_factor = penalty_factor,
    n_inner_folds = N_INNER_FOLDS,
    outcome_name = "DeathNoR"
  )
  
  # Extract coefficients and features
  coef_death <- coef(death_model$final_fit)
  coef_death_df <- data.frame(
    feature = rownames(coef_death)[as.vector(coef_death != 0)],
    coefficient = as.vector(coef_death)[as.vector(coef_death != 0)])
  coef_death_df <- coef_death_df[order(abs(coef_death_df$coefficient), decreasing = TRUE), ]
  rownames(coef_death_df) <- NULL
  features_death <- coef_death_df$feature
  
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
  # WHY: Fine-Gray model should include features predictive of EITHER outcome
  # This ensures we model the subdistribution hazard properly
  source("./src/finegray_functions.R")
  
  features_pooled <- union(features_rfi, features_death)
  
  # prepare FG input
  X_pooled_train <- X_train_filtered[, features_pooled, drop = FALSE]
  X_pooled_test <- X_test_filtered[, features_pooled, drop = FALSE]
  
  cat(sprintf("\n--- Feature Pooling ---\n"))
  cat(sprintf("RFI features: %d\n", length(features_rfi)))
  cat(sprintf("Death features: %d\n", length(features_death)))
  cat(sprintf("Overlap: %d\n", length(intersect(features_rfi, features_death))))
  cat(sprintf("Pooled total: %d\n", length(features_pooled)))
  
  # --------------------------------------------------------------------------
  # Scale Input Data for Fine-Gray Model
  # --------------------------------------------------------------------------
  # WHY: Fine-Gray model is sensitive to feature scales
  # Only scale continuous variables, leave one-hot encoded variables as-is
  
  cat(sprintf("\n--- Scaling Cont. Features ---\n"))
  scale_res <- scale_continuous_features(X_train = X_pooled_train, 
                            X_test = X_pooled_test, 
                            dont_scale = encoded_result$encoded_cols)
  
  X_train_scaled <- scale_res$X_train_scaled
  X_test_scaled <- scale_res$X_test_scaled
  
  # --------------------------------------------------------------------------
  # Fit Fine-Gray Model on Pooled Features
  # --------------------------------------------------------------------------
  # Fine-Gray model: Models subdistribution hazard for event of interest
  
  cat(sprintf("\n--- Fitting Fine-Gray Model ---\n"))
  
  fit_fg <- crr(
    ftime   = clinical_train$time_to_CompRisk_event,
    fstatus = clinical_train$CompRisk_event_coded,
    cov1    = as.matrix(X_train_scaled),
    failcode = 1,  # Event of interest (RFI)
    cencode  = 0   # Censored
  )
  
  cat(sprintf("Fine-Gray model fitted with %d features\n", ncol(X_train_scaled)))
  
  # --------------------------------------------------------------------------
  # Extract and Display Coefficients
  # --------------------------------------------------------------------------
  
  coef_fg <- fit_fg$coef
  hr_fg   <- exp(coef_fg)  # Subdistribution hazard ratios
  
  coef_table <- data.frame(
    Variable = names(coef_fg),
    Coef     = coef_fg,
    SHR      = hr_fg  # Subdistribution Hazard Ratio
  )
  rownames(coef_table) <- NULL
  
  # Sort by absolute coefficient size (largest effect first)
  coef_table <- coef_table[order(-abs(coef_table$Coef)), ]
  
  cat("\nTop 15 Features by Effect Size:\n")
  print(head(coef_table, 15))
  
  # --------------------------------------------------------------------------
  # Predict on Test Set
  # --------------------------------------------------------------------------
  
  cat(sprintf("\n--- Test Set Predictions ---\n"))
  
  # Calculate linear predictor (risk score)
  # This is log-relative subdistribution hazard for each patient
  risk_scores_test <- as.matrix(X_test_scaled) %*% fit_fg$coef
  
  cat(sprintf("Generated risk scores for %d test samples\n", nrow(X_test_scaled)))
  cat(sprintf("Risk score range: [%.3f, %.3f]\n", 
              min(risk_scores_test), max(risk_scores_test)))
  
  # Predict cumulative incidence functions (CIF)
  # pred_fg[,1] = unique event times from training data
  # pred_fg[,2:n] = CIF curves for each test patient
  pred_fg <- predict(fit_fg, cov1 = as.matrix(X_test_scaled))
  
  timepoints <- pred_fg[, 1]      # Column 1 = unique failure times
  cif_curves <- pred_fg[, -1]     # Columns 2:n = CIF for each patient
  
  cat(sprintf("CIF timepoints: %d (from training events)\n", length(timepoints)))
  cat(sprintf("Test patients: %d\n", ncol(cif_curves)))
  
  # --------------------------------------------------------------------------
  # Evaluate Performance at Specific Timepoints
  # --------------------------------------------------------------------------
  
  # Only evaluate at timepoints < max observed time in test set
  max_time <- max(clinical_test$time_to_CompRisk_event)
  eval_times_use <- EVAL_TIMES[EVAL_TIMES < max_time]
  
  n_pat <- ncol(cif_curves)
  
  cat(sprintf("\nInterpolating CIF for %d patients at %d timepoints\n", 
              n_pat, length(eval_times_use)))
  
  # Interpolate CIF at requested evaluation times
  # This gives us P(RFI by time t) for each patient
  pred_cif_subset <- matrix(NA, nrow = n_pat, ncol = length(eval_times_use))
  colnames(pred_cif_subset) <- paste0("t", eval_times_use, "y")
  
  for (p in 1:n_pat) {
    interp <- approx(
      x = timepoints,                            # Training event times
      y = cif_curves[, p],                       # Patient's CIF curve
      xout = eval_times_use,                     # Evaluation timepoints
      yleft = 0,                                 # CIF = 0 before first event
      yright = cif_curves[nrow(cif_curves), p],  # Carry forward after last event
      method = "linear"
    )
    pred_cif_subset[p, ] <- interp$y
  }
  
  if (length(eval_times_use) >= 1) {
    cat(sprintf("CIF range at %gy: [%.4f, %.4f]\n", 
                eval_times_use[1], 
                min(pred_cif_subset[,1]), 
                max(pred_cif_subset[,1])))
  }
  if (length(eval_times_use) >= 2) {
    cat(sprintf("CIF range at %gy: [%.4f, %.4f]\n", 
                eval_times_use[2], 
                min(pred_cif_subset[,2]), 
                max(pred_cif_subset[,2])))
  }
  
  # --------------------------------------------------------------------------
  # Calculate Performance Metrics
  # --------------------------------------------------------------------------
  # Uses riskRegression::Score() which properly handles competing risks
  # - AUC: Time-dependent discrimination (higher = better)
  # - Brier: Time-dependent calibration (lower = better)
  # - Both use inverse probability of censoring weighting (IPCW)
  
  score_fg <- Score(
    list("FG_model" = pred_cif_subset),
    formula = Hist(time_to_CompRisk_event, CompRisk_event_coded) ~ 1,
    data    = clinical_test,
    times   = eval_times_use,
    cause   = 1,  # RFI is event of interest
    metrics = c("auc", "brier"),
    se.fit  = TRUE,
    cens.model = "cox"  # Cox model for censoring weights
  )
  
  # --------------------------------------------------------------------------
  # Display Performance Metrics
  # --------------------------------------------------------------------------
  
  cat("\n========================================\n")
  cat("Fine-Gray Model Performance (Test Set)\n")
  cat("========================================\n")
  
  # Extract results
  auc_fg <- score_fg$AUC$score[score_fg$AUC$score$model == "FG_model", ]
  brier_fg <- score_fg$Brier$score[score_fg$Brier$score$model == "FG_model", ]
  
  # Time-specific AUC
  cat("\nTime-dependent AUC:\n")
  for (i in 1:nrow(auc_fg)) {
    cat(sprintf("  %g years: %.3f (SE: %.4f)\n", 
                auc_fg$times[i], auc_fg$AUC[i], auc_fg$se[i]))
  }
  
  # Time-specific Brier
  cat("\nBrier Score:\n")
  for (i in 1:nrow(brier_fg)) {
    cat(sprintf("  %g years: %.4f (SE: %.5f)\n",
                brier_fg$times[i], brier_fg$Brier[i], brier_fg$se[i]))
  }
  
  # Integrated metrics
  cat("\nIntegrated Metrics:\n")
  
  # Mean AUC (average over eval times, ignoring any NA values)
  mean_auc <- NA
  if (nrow(auc_fg) > 0) {
    mean_auc <- mean(auc_fg$AUC, na.rm = TRUE)
    if (!is.na(mean_auc)) {
      cat(sprintf("  Mean AUC: %.3f (across %d timepoints)\n", 
                  mean_auc, sum(!is.na(auc_fg$AUC))))
    }
  }
  
  # IBS (Integrated Brier Score) - trapezoidal integration
  ibs <- NA
  if (nrow(brier_fg) > 1) {
    # Only use non-NA Brier scores
    valid_brier <- !is.na(brier_fg$Brier)
    if (sum(valid_brier) > 1) {
      brier_valid <- brier_fg[valid_brier, ]
      time_diffs <- diff(brier_valid$times)
      avg_brier <- (brier_valid$Brier[-1] + brier_valid$Brier[-nrow(brier_valid)]) / 2
      ibs <- sum(time_diffs * avg_brier) / diff(range(brier_valid$times))
      cat(sprintf("  IBS: %.4f (across %d timepoints)\n", ibs, nrow(brier_valid)))
    }
  }
  
  # C-index using concordance on risk scores
  cindex <- concordance(
    Surv(time_to_CompRisk_event, CompRisk_event_coded == 1) ~ risk_scores_test,
    data = clinical_test
  )
  cat(sprintf("  C-index: %.3f (SE: %.4f)\n", 
              cindex$concordance, sqrt(cindex$var)))
  
  cat("\n")
  
  fold_time <- difftime(Sys.time(), fold_start_time, units = "mins")
  cat(sprintf("Fold completed in %.1f minutes\n", fold_time))
  
  # --------------------------------------------------------------------------
  # SAVE FOLD RESULTS
  # --------------------------------------------------------------------------
  
  # Store comprehensive results for this fold
  outer_fold_results[[fold_idx]] <- list(
    # Fold information
    fold_idx = fold_idx,
    train_idx = train_idx,
    test_idx = test_idx,
    train_samples = rownames(X_train),
    test_samples = rownames(X_test),
    n_train = nrow(X_train),
    n_test = nrow(X_test),
    
    # Feature selection results
    features_rfi = features_rfi,
    features_death = features_death,
    features_pooled = features_pooled,
    n_features_final = length(features_pooled),
    variance_threshold = var_threshold,
    
    # CoxNet models and hyperparameters
    coxnet_rfi_model = final_fit_outer_rfi,
    coxnet_rfi_alpha = best_alpha_rfi,
    coxnet_rfi_lambda = best_lambda_rfi,
    coxnet_rfi_cv_result = best_rfi_result,
    coxnet_rfi_coefficients = data.frame(
      feature = features_rfi,
      coefficient = as.vector(coef_rfi[features_rfi, ])
    ),
    
    coxnet_death_model = final_fit_outer_death,
    coxnet_death_alpha = best_alpha_death,
    coxnet_death_lambda = best_lambda_death,
    coxnet_death_cv_result = best_death_result,
    coxnet_death_coefficients = data.frame(
      feature = features_death,
      coefficient = as.vector(coef_death[features_death, ])
    ),
    
    # Fine-Gray model
    finegray_model = fit_fg,
    fg_coefficients = coef_table,
    
    # Performance metrics
    auc_by_time = auc_fg,
    brier_by_time = brier_fg,
    mean_auc = mean_auc,
    ibs = ibs,
    cindex = cindex$concordance,
    cindex_se = sqrt(cindex$var),
    
    # Predictions
    risk_scores = as.vector(risk_scores_test),
    pred_cif = pred_cif_subset,
    eval_times = eval_times_use,
    
    # Scaling parameters (for future use)
    scale_centers = centers,
    scale_scales = scales,
    continuous_cols = cont_cols,
    
    # Timing
    fold_duration_mins = as.numeric(fold_time)
  )
  
  # Store predictions with actual outcomes for later aggregation
  fold_predictions <- data.frame(
    sample_id = rownames(X_test),
    fold = fold_idx,
    risk_score = as.vector(risk_scores_test),
    time = clinical_test$time_to_CompRisk_event,
    event = clinical_test$CompRisk_event_coded,
    rfi_event = clinical_test$RFi_event,
    death_event = clinical_test$DeathNoR_event
  )
  
  # Add CIF predictions
  for (i in 1:length(eval_times_use)) {
    fold_predictions[[paste0("CIF_", eval_times_use[i], "y")]] <- pred_cif_subset[, i]
  }
  
  all_predictions <- rbind(all_predictions, fold_predictions)
  
  # Save individual fold results
  fold_output_file <- file.path(
    current_output_dir, 
    sprintf("fold_%d_results.rds", fold_idx)
  )
  saveRDS(outer_fold_results[[fold_idx]], fold_output_file)
  cat(sprintf("Saved fold results to: %s\n", fold_output_file))
}

################################################################################
# AGGREGATE RESULTS ACROSS FOLDS
################################################################################

cat(sprintf("\n========== AGGREGATING RESULTS ==========\n"))

total_time <- difftime(Sys.time(), start_time, units = "mins")

# Create fold assignment dataframe
cat("\nCreating fold assignment table...\n")
fold_assignments <- data.frame()
for (fold_idx in 1:N_OUTER_FOLDS) {
  fold_res <- outer_fold_results[[fold_idx]]
  
  # Train samples
  train_df <- data.frame(
    sample_id = fold_res$train_samples,
    fold = fold_idx,
    set = "train"
  )
  
  # Test samples
  test_df <- data.frame(
    sample_id = fold_res$test_samples,
    fold = fold_idx,
    set = "test"
  )
  
  fold_assignments <- rbind(fold_assignments, train_df, test_df)
}

cat(sprintf("Created fold assignments for %d samples across %d folds\n", 
            nrow(fold_assignments), N_OUTER_FOLDS))

# Extract performance metrics from each fold
all_aucs <- data.frame()
all_briers <- data.frame()
all_cindices <- data.frame()

for (fold_idx in 1:N_OUTER_FOLDS) {
  fold_res <- outer_fold_results[[fold_idx]]
  
  # AUC
  if (!is.null(fold_res$auc_by_time) && nrow(fold_res$auc_by_time) > 0) {
    fold_auc <- fold_res$auc_by_time
    fold_auc$fold <- fold_idx
    all_aucs <- rbind(all_aucs, fold_auc)
  }
  
  # Brier
  if (!is.null(fold_res$brier_by_time) && nrow(fold_res$brier_by_time) > 0) {
    fold_brier <- fold_res$brier_by_time
    fold_brier$fold <- fold_idx
    all_briers <- rbind(all_briers, fold_brier)
  }
  
  # C-index
  all_cindices <- rbind(all_cindices, data.frame(
    fold = fold_idx,
    cindex = fold_res$cindex,
    cindex_se = fold_res$cindex_se
  ))
}

# Calculate mean and SD across folds for each timepoint
cat("\n========================================\n")
cat("OVERALL PERFORMANCE (ACROSS ALL FOLDS)\n")
cat("========================================\n")

if (nrow(all_aucs) > 0) {
  cat("\nTime-dependent AUC (Mean ± SD):\n")
  auc_summary <- all_aucs %>%
    group_by(times) %>%
    summarise(
      mean_AUC = mean(AUC, na.rm = TRUE),
      sd_AUC = sd(AUC, na.rm = TRUE),
      n = n()
    )
  
  for (i in 1:nrow(auc_summary)) {
    cat(sprintf("  %g years: %.3f ± %.3f (n=%d folds)\n",
                auc_summary$times[i],
                auc_summary$mean_AUC[i],
                auc_summary$sd_AUC[i],
                auc_summary$n[i]))
  }
}

if (nrow(all_briers) > 0) {
  cat("\nBrier Score (Mean ± SD):\n")
  brier_summary <- all_briers %>%
    group_by(times) %>%
    summarise(
      mean_Brier = mean(Brier, na.rm = TRUE),
      sd_Brier = sd(Brier, na.rm = TRUE),
      n = n()
    )
  
  for (i in 1:nrow(brier_summary)) {
    cat(sprintf("  %g years: %.4f ± %.4f (n=%d folds)\n",
                brier_summary$times[i],
                brier_summary$mean_Brier[i],
                brier_summary$sd_Brier[i],
                brier_summary$n[i]))
  }
}

cat("\nC-index (Mean ± SD):\n")
cat(sprintf("  %.3f ± %.3f (n=%d folds)\n",
            mean(all_cindices$cindex),
            sd(all_cindices$cindex),
            nrow(all_cindices)))

# Feature selection stability
cat("\n========================================\n")
cat("FEATURE SELECTION STABILITY\n")
cat("========================================\n")

# Count how often each feature was selected across folds
all_selected_features <- unlist(lapply(outer_fold_results, 
                                       function(x) x$features_pooled))
feature_counts <- table(all_selected_features)
feature_freq <- feature_counts / N_OUTER_FOLDS

cat(sprintf("\nTotal unique features selected: %d\n", length(feature_freq)))
cat(sprintf("Features selected in all %d folds: %d\n", 
            N_OUTER_FOLDS, 
            sum(feature_freq == 1)))
cat(sprintf("Features selected in ≥ %d folds: %d\n", 
            ceiling(N_OUTER_FOLDS / 2),
            sum(feature_freq >= 0.5)))

# Show most frequently selected features
cat("\nMost Stable Features (selected in ≥50% of folds):\n")
stable_features <- names(feature_freq[feature_freq >= 0.5])
stable_features <- stable_features[order(-feature_freq[feature_freq >= 0.5])]

if (length(stable_features) > 0) {
  for (feat in head(stable_features, 20)) {
    cat(sprintf("  %s: %.0f%%\n", feat, feature_freq[feat] * 100))
  }
} else {
  cat("  No features selected in ≥50% of folds\n")
}

# Detailed feature selection by model type
cat("\n========================================\n")
cat("FEATURE SELECTION BY MODEL TYPE\n")
cat("========================================\n")

# Track RFI-specific features
all_rfi_features <- unlist(lapply(outer_fold_results, function(x) x$features_rfi))
rfi_feature_counts <- table(all_rfi_features)
rfi_feature_freq <- rfi_feature_counts / N_OUTER_FOLDS

cat(sprintf("\nRFI Model Features:\n"))
cat(sprintf("  Total unique: %d\n", length(rfi_feature_freq)))
cat(sprintf("  Selected in all folds: %d\n", sum(rfi_feature_freq == 1)))
cat(sprintf("  Selected in ≥50%% folds: %d\n", sum(rfi_feature_freq >= 0.5)))

# Track Death-specific features
all_death_features <- unlist(lapply(outer_fold_results, function(x) x$features_death))
death_feature_counts <- table(all_death_features)
death_feature_freq <- death_feature_counts / N_OUTER_FOLDS

cat(sprintf("\nDeath Model Features:\n"))
cat(sprintf("  Total unique: %d\n", length(death_feature_freq)))
cat(sprintf("  Selected in all folds: %d\n", sum(death_feature_freq == 1)))
cat(sprintf("  Selected in ≥50%% folds: %d\n", sum(death_feature_freq >= 0.5)))

################################################################################
# SAVE ALL RESULTS
################################################################################

cat(sprintf("\n========== SAVING RESULTS ==========\n"))

# Save aggregated results
results_summary <- list(
  # Configuration
  cohort_name = COHORT_NAME,
  data_mode = DATA_MODE,
  n_outer_folds = N_OUTER_FOLDS,
  n_inner_folds = N_INNER_FOLDS,
  alpha_grid = ALPHA_GRID,
  eval_times = EVAL_TIMES,
  variance_quantile = VARIANCE_QUANTILE,
  clinical_vars_continuous = CLIN_CONT,
  clinical_vars_categorical = CLIN_CATEGORICAL,
  
  # Fold assignments
  fold_assignments = fold_assignments,
  
  # Overall performance
  auc_summary = auc_summary,
  brier_summary = brier_summary,
  cindex_summary = all_cindices,
  mean_cindex = mean(all_cindices$cindex, na.rm = TRUE),
  sd_cindex = sd(all_cindices$cindex, na.rm = TRUE),
  
  # Feature selection
  feature_frequency = feature_freq,
  stable_features = stable_features,
  rfi_feature_frequency = rfi_feature_freq,
  death_feature_frequency = death_feature_freq,
  
  # Detailed fold results
  fold_results = outer_fold_results,
  
  # All predictions
  all_predictions = all_predictions,
  
  # Timing
  total_time_mins = as.numeric(total_time)
)

# Save main results file
main_results_file <- file.path(current_output_dir, "nCV_results_summary.rds")
saveRDS(results_summary, main_results_file)
cat(sprintf("Saved summary results to: %s\n", main_results_file))

# Save predictions as CSV for easy viewing
pred_file <- file.path(current_output_dir, "all_predictions.csv")
write.csv(all_predictions, pred_file, row.names = FALSE)
cat(sprintf("Saved predictions to: %s\n", pred_file))

# Save fold assignments
fold_assign_file <- file.path(current_output_dir, "fold_assignments.csv")
write.csv(fold_assignments, fold_assign_file, row.names = FALSE)
cat(sprintf("Saved fold assignments to: %s\n", fold_assign_file))

# Save performance summaries as CSV
if (exists("auc_summary")) {
  auc_file <- file.path(current_output_dir, "auc_summary.csv")
  write.csv(auc_summary, auc_file, row.names = FALSE)
}

if (exists("brier_summary")) {
  brier_file <- file.path(current_output_dir, "brier_summary.csv")
  write.csv(brier_summary, brier_file, row.names = FALSE)
}

cindex_file <- file.path(current_output_dir, "cindex_by_fold.csv")
write.csv(all_cindices, cindex_file, row.names = FALSE)

# Save feature stability
feature_stability <- data.frame(
  feature = names(feature_freq),
  selection_frequency = as.numeric(feature_freq),
  n_folds_selected = as.numeric(feature_counts)
)
feature_stability <- feature_stability[order(-feature_stability$selection_frequency), ]

feat_file <- file.path(current_output_dir, "feature_stability.csv")
write.csv(feature_stability, feat_file, row.names = FALSE)
cat(sprintf("Saved feature stability to: %s\n", feat_file))

# Save RFI-specific feature selection
rfi_feature_stability <- data.frame(
  feature = names(rfi_feature_freq),
  selection_frequency = as.numeric(rfi_feature_freq),
  n_folds_selected = as.numeric(rfi_feature_counts)
)
rfi_feature_stability <- rfi_feature_stability[order(-rfi_feature_stability$selection_frequency), ]

rfi_feat_file <- file.path(current_output_dir, "feature_stability_rfi_model.csv")
write.csv(rfi_feature_stability, rfi_feat_file, row.names = FALSE)
cat(sprintf("Saved RFI feature stability to: %s\n", rfi_feat_file))

# Save death-specific feature selection
death_feature_stability <- data.frame(
  feature = names(death_feature_freq),
  selection_frequency = as.numeric(death_feature_freq),
  n_folds_selected = as.numeric(death_feature_counts)
)
death_feature_stability <- death_feature_stability[order(-death_feature_stability$selection_frequency), ]

death_feat_file <- file.path(current_output_dir, "feature_stability_death_model.csv")
write.csv(death_feature_stability, death_feat_file, row.names = FALSE)
cat(sprintf("Saved death feature stability to: %s\n", death_feat_file))

# Save hyperparameter summary across folds
hyperparam_summary <- data.frame()
for (fold_idx in 1:N_OUTER_FOLDS) {
  fold_res <- outer_fold_results[[fold_idx]]
  hyperparam_summary <- rbind(hyperparam_summary, data.frame(
    fold = fold_idx,
    rfi_alpha = fold_res$coxnet_rfi_alpha,
    rfi_lambda = fold_res$coxnet_rfi_lambda,
    rfi_n_features = length(fold_res$features_rfi),
    death_alpha = fold_res$coxnet_death_alpha,
    death_lambda = fold_res$coxnet_death_lambda,
    death_n_features = length(fold_res$features_death),
    pooled_n_features = length(fold_res$features_pooled)
  ))
}

hyperparam_file <- file.path(current_output_dir, "coxnet_hyperparameters.csv")
write.csv(hyperparam_summary, hyperparam_file, row.names = FALSE)
cat(sprintf("Saved CoxNet hyperparameters to: %s\n", hyperparam_file))

################################################################################
# FINAL SUMMARY
################################################################################

cat(sprintf("\n========================================\n"))
cat(sprintf("PIPELINE COMPLETED SUCCESSFULLY\n"))
cat(sprintf("========================================\n"))
cat(sprintf("Total runtime: %.1f minutes\n", total_time))
cat(sprintf("Output directory: %s\n", current_output_dir))
cat(sprintf("\nGenerated files:\n"))
cat(sprintf("  Core Results:\n"))
cat(sprintf("    - nCV_results_summary.rds (complete results object)\n"))
cat(sprintf("    - fold_X_results.rds (individual fold results with models)\n"))
cat(sprintf("\n  Predictions & Assignments:\n"))
cat(sprintf("    - all_predictions.csv (test set predictions)\n"))
cat(sprintf("    - fold_assignments.csv (sample-to-fold mapping)\n"))
cat(sprintf("\n  Performance Metrics:\n"))
cat(sprintf("    - auc_summary.csv (AUC by timepoint)\n"))
cat(sprintf("    - brier_summary.csv (Brier by timepoint)\n"))
cat(sprintf("    - cindex_by_fold.csv (C-index per fold)\n"))
cat(sprintf("\n  Feature Selection:\n"))
cat(sprintf("    - feature_stability.csv (pooled feature frequency)\n"))
cat(sprintf("    - feature_stability_rfi_model.csv (RFI model features)\n"))
cat(sprintf("    - feature_stability_death_model.csv (death model features)\n"))
cat(sprintf("\n  Model Information:\n"))
cat(sprintf("    - coxnet_hyperparameters.csv (alpha/lambda per fold)\n"))
cat(sprintf("\n"))