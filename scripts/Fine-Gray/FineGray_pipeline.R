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
library(prodlim) #prodlim::Hist()

setwd("~/PhD_Workspace/PredictRecurrence/")
source("./src/finegray_functions.R") # match real name

################################################################################
# SETTINGS
################################################################################


COHORT_NAME <- "ERpHER2n"
DATA_MODE <- "combined"
TRAIN_CPGS <- "./data/set_definitions/CpG_prefiltered_sets/cpg_ids_atac_overlap.txt"
OUTPUT_BASE_DIR <- "./output/FineGray"

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
script_name <- "finegray_pipeline"
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
CLIN_CATEGORICAL <- c("NHG", "LN")
CLIN_CONT <- c("Age", "Size.mm")

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

# Combine methylation with clinical variables
X <- mvals

if (!is.null(c(CLIN_CONT, CLIN_CATEGORICAL))) {
  # Recode N+ to Np to avoid special character issues in formulas
  clinical_data$LN <- gsub("N\\+", "Np", clinical_data$LN)
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
VARS_NO_PENALIZATION <- clinvars  # Set to NULL to penalize all features

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
  # Feature Filtering on Outer Training Data
  # --------------------------------------------------------------------------
  # WHY: Reduce dimensionality before elastic net
  # IMPORTANT: Filtering done ONLY on training data to prevent data leakage
  
  cat(sprintf("\n--- Feature Filtering ---\n"))
  
  X_train_filtered <- prepare_filtered_features(
    X_train,
    vars_preserve = VARS_PRESERVE, 
    variance_quantile = VARIANCE_QUANTILE
  )
  
  X_test_filtered <- X_test[, colnames(X_train_filtered), drop = FALSE]
  
  # --------------------------------------------------------------------------
  # Create Penalty Factor for Elastic Net
  # --------------------------------------------------------------------------
  
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
  rfi_coef_res <- extract_nonzero_coefs(rfi_model$final_fit)
  coef_rfi_df <- rfi_coef_res$coef_df
  features_rfi <- rfi_coef_res$features
  
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
  # WHY: Fine-Gray model should include features predictive of EITHER outcome
  # This ensures we model the subdistribution hazard properly
  
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
  X_pooled_train <- X_train_filtered[, features_pooled, drop = FALSE]
  X_pooled_test <- X_test_filtered[, features_pooled, drop = FALSE]
  
  # --------------------------------------------------------------------------
  # Scale Input Data for Fine-Gray Model
  # --------------------------------------------------------------------------
  # WHY: Fine-Gray model is sensitive to feature scales
  # Only scale continuous variables, leave one-hot encoded variables as-is
  
  cat(sprintf("\n--- Scaling Continuous Features ---\n"))
  scale_res <- scale_continuous_features(
    X_train = X_pooled_train, 
    X_test = X_pooled_test, 
    dont_scale = encoded_result$encoded_cols
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
  coef_comparison <- merge(coef_fg_df, coef_rfi_df, by = "feature", all = TRUE)
  coef_comparison <- merge(coef_comparison, coef_death_df, by = "feature", all = TRUE)
  coef_comparison[is.na(coef_comparison)] <- 0
  
  coef_comparison$fg_HR <- exp(coef_comparison$fg_coef)
  coef_comparison <- coef_comparison[c("feature","cox_rfi_coef",
                                       "cox_death_coef","fg_coef",
                                       "fg_HR")]
  
  # Sort by absolute FG coefficient
  coef_comparison <- coef_comparison[order(abs(coef_comparison$fg_HR), decreasing = TRUE), ]
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
    coxnet_rfi_model = rfi_model$final_fit,
    coxnet_rfi_cv_result = rfi_model$best_result,
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
# AGGREGATE RESULTS ACROSS FOLDS
################################################################################

# Get aggregated performance
perf_results <- aggregate_cv_performance(outer_fold_results)

# Feature selection stability
stability_results <- assess_feature_stability(outer_fold_results, min_folds = N_OUTER_FOLDS-1)

################################################################################
# TRAIN FINAL MODEL ON ALL DATA
################################################################################

cat(sprintf("\n========== TRAINING FINAL MODEL ==========\n"))

# Use same procedure as outer loop but no performance evaluation
X_all <- X
y_rfi_all <- y_rfi
y_death_all <- y_death_no_r
clinical_all <- clinical_data

cat(sprintf("Training samples: n=%d (RFi=%d, Death=%d)\n",
            nrow(X_all), 
            sum(clinical_all$RFi_event), 
            sum(clinical_all$DeathNoR_event)))

# --------------------------------------------------------------------------
# Feature Filtering
# --------------------------------------------------------------------------

X_all_filtered <- prepare_filtered_features(
  X_all,
  vars_preserve = VARS_PRESERVE, 
  variance_quantile = VARIANCE_QUANTILE
)

# --------------------------------------------------------------------------
# Create Penalty Factor
# --------------------------------------------------------------------------

penalty_factor <- rep(1, ncol(X_all_filtered))
if (!is.null(VARS_NO_PENALIZATION)) {
  penalty_factor[colnames(X_all_filtered) %in% VARS_NO_PENALIZATION] <- 0
}

# --------------------------------------------------------------------------
# CoxNet for RFI
# --------------------------------------------------------------------------

rfi_model_all <- tune_and_fit_coxnet(
  X_train = X_all_filtered,
  y_train = y_rfi_all,
  clinical_train = clinical_all,
  event_col = "RFi_event",
  alpha_grid = ALPHA_GRID,
  penalty_factor = penalty_factor,
  n_inner_folds = N_INNER_FOLDS,
  outcome_name = "RFI"
)

rfi_coef_res_all <- extract_nonzero_coefs(rfi_model_all$final_fit)
coef_rfi_df_all <- rfi_coef_res_all$coef_df
features_rfi_all <- rfi_coef_res_all$features

cat(sprintf(
  "  BEST RFI MODEL: Alpha=%.2f, Lambda=%.6f, Features=%d\n",
  rfi_model_all$best_alpha, 
  rfi_model_all$best_lambda, 
  length(features_rfi_all)
))
cat("    Selected:", paste(features_rfi_all, collapse=", "), "\n")

# --------------------------------------------------------------------------
# CoxNet for Death
# --------------------------------------------------------------------------

death_model_all <- tune_and_fit_coxnet(
  X_train = X_all_filtered,
  y_train = y_death_all,
  clinical_train = clinical_all,
  event_col = "DeathNoR_event",
  alpha_grid = ALPHA_GRID,
  penalty_factor = penalty_factor,
  n_inner_folds = N_INNER_FOLDS,
  outcome_name = "DeathNoR"
)

death_coef_res_all <- extract_nonzero_coefs(death_model_all$final_fit)
coef_death_df_all <- death_coef_res_all$coef_df
features_death_all <- death_coef_res_all$features

cat(sprintf(
  "  BEST DEATH MODEL: Alpha=%.2f, Lambda=%.6f, Features=%d\n",
  death_model_all$best_alpha, 
  death_model_all$best_lambda, 
  length(features_death_all)
))
cat("    Selected:", paste(features_death_all, collapse=", "), "\n")

# --------------------------------------------------------------------------
# Pool Features
# --------------------------------------------------------------------------

features_pooled_all <- union(features_rfi_all, features_death_all)
X_pooled_all <- X_all_filtered[, features_pooled_all, drop = FALSE]

cat(sprintf("\n--- Feature Pooling ---\n"))
cat(sprintf("RFI features: %d\n", length(features_rfi_all)))
cat(sprintf("Death features: %d\n", length(features_death_all)))
cat(sprintf("Overlap: %d\n", length(intersect(features_rfi_all, features_death_all))))
cat(sprintf("Pooled total: %d\n", length(features_pooled_all)))

# --------------------------------------------------------------------------
# Scale Features
# --------------------------------------------------------------------------

cat(sprintf("\n--- Scaling Continuous Features ---\n"))
scale_res_all <- scale_continuous_features(
  X_train = X_pooled_all, 
  X_test = NULL, 
  dont_scale = encoded_result$encoded_cols
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

# --------------------------------------------------------------------------
# Extract Final Model Coefficients
# --------------------------------------------------------------------------

fg_final_coef <- fgr_final$crrFit$coef
coef_fg_final_df <- data.frame(
  feature = names(fg_final_coef),
  fg_final_coef = as.vector(fg_final_coef)
)
names(coef_rfi_df_all) <- c("feature","cox_rfi_final_coef") 
names(coef_death_df_all) <- c("feature","cox_death_final_coef")

# Merge all three
coef_comparison_final <- merge(coef_fg_final_df, coef_rfi_df_all, by = "feature", all = TRUE)
coef_comparison_final <- merge(coef_comparison_final, coef_death_df_all, by = "feature", all = TRUE)
coef_comparison_final[is.na(coef_comparison_final)] <- 0
coef_comparison_final$fg_final_HR <- exp(coef_comparison_final$fg_final_coef)
coef_comparison_final <- coef_comparison_final[c("feature","cox_rfi_final_coef",
                                                 "cox_death_final_coef","fg_final_coef",
                                                 "fg_final_HR")]

# Sort by absolute Fine-Gray hazard ratio
coef_comparison_final <- coef_comparison_final[order(
  abs(coef_comparison_final$fg_final_HR), decreasing = TRUE), ]
rownames(coef_comparison_final) <- NULL

print(coef_comparison_final)

# --------------------------------------------------------------------------
# Calculate Variable Importance
# --------------------------------------------------------------------------

cat("\n--- Fine-Gray Variable Importance ---\n")

vimp_fg_final <- calculate_fgr_importance(
  fgr_model = fgr_final,
  encoded_cols = encoded_result$encoded_cols,
  verbose = TRUE
)

print(vimp_fg_final)

################################################################################
# SAVE RESULTS
################################################################################

cat("\n========== SAVING RESULTS ==========\n")

# Create output directory
results_dir <- file.path(current_output_dir, "final_results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# --------------------------------------------------------------------------
# Combine CV Predictions from All Folds
# --------------------------------------------------------------------------

cat("Combining CV predictions from all folds...\n")

cv_predictions <- do.call(rbind, lapply(outer_fold_results, function(fold) {
  fold$fold_predictions
}))

# --------------------------------------------------------------------------
# Save All Results in .RData File
# --------------------------------------------------------------------------

save(
  # Performance
  perf_results,
  cv_predictions,
  
  # Stability
  stability_results,
  
  # Final model objects
  fgr_final,
  rfi_model_all,
  death_model_all,
  
  # Coefficients and importance
  coef_comparison_final,
  vimp_fg_final,
  
  # Features
  features_rfi_all,
  features_death_all,
  features_pooled_all,
  
  # Scaling parameters (critical for applying model to new data)
  scale_res_all,
  encoded_result,
  
  # CV results
  outer_fold_results,
  
  # Parameters
  N_OUTER_FOLDS,
  N_INNER_FOLDS,
  ALPHA_GRID,
  EVAL_TIMES,
  COHORT_NAME,
  clinvars,
  
  file = file.path(results_dir, "all_results.RData")
)

# --------------------------------------------------------------------------
# Save Key CSVs for Easy Viewing
# --------------------------------------------------------------------------

write.csv(perf_results$summary, 
          file.path(results_dir, "performance_summary.csv"), 
          row.names = FALSE)

write.csv(perf_results$all_folds, 
          file.path(results_dir, "performance_all_folds.csv"), 
          row.names = FALSE)

write.csv(cv_predictions, 
          file.path(results_dir, "cv_predictions.csv"), 
          row.names = FALSE)

write.csv(vimp_fg_final, 
          file.path(results_dir, "variable_importance.csv"), 
          row.names = FALSE)

write.csv(coef_comparison_final, 
          file.path(results_dir, "final_coefficients.csv"), 
          row.names = FALSE)

write.csv(stability_results$finegray_stability, 
          file.path(results_dir, "stability_finegray.csv"), 
          row.names = FALSE)

# --------------------------------------------------------------------------
# Print Summary
# --------------------------------------------------------------------------

cat(sprintf("\n✓ Results saved to: %s\n", results_dir))
cat("  - all_results.RData (complete workspace)\n")
cat("  - performance_summary.csv\n")
cat("  - performance_all_folds.csv\n")
cat("  - cv_predictions.csv (for ROC/calibration plots)\n")
cat("  - variable_importance.csv\n")
cat("  - final_coefficients.csv\n")
cat("  - stability_finegray.csv\n")

cat(sprintf("\nTotal runtime: %.1f minutes\n", 
            as.numeric(difftime(Sys.time(), start_time, units = "mins"))))

################################################################################
# SESSION INFO
################################################################################

cat(sprintf("\n%s\n", paste(rep("=", 80), collapse = "")))
cat("SESSION INFO\n")
cat(sprintf("%s\n", paste(rep("=", 80), collapse = "")))
print(sessionInfo())

################################################################################
# CLOSE LOG FILE
################################################################################

cat(sprintf("\n%s\n", paste(rep("=", 80), collapse = "")))
cat(sprintf("Completed: %s\n", Sys.time()))
cat(sprintf("Log saved to: %s\n", log_path))
cat(sprintf("%s\n", paste(rep("=", 80), collapse = "")))

# Close log connections
sink(type = "message")
sink(type = "output")
close(log_con)

cat(sprintf("\n✓ Pipeline completed successfully!\n"))
cat(sprintf("✓ Log file: %s\n", log_path))