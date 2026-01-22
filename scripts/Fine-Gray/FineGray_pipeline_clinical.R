#!/usr/bin/env Rscript

################################################################################
# Fine-Gray Competing Risks - Clinical Variables Only
# Author: Lennart Hohmann
#
# PIPELINE OVERVIEW:
# ==================
# This pipeline implements simple cross-validation for competing risks analysis
# of breast cancer recurrence using ONLY clinical variables.
#
# 1. DATA PREPARATION
#    - Load clinical data
#    - Apply administrative censoring if specified (e.g., 5 years for TNBC)
#    - One-hot encode categorical clinical variables (NHG, lymph nodes)
#
# 2. COMPETING RISKS DEFINITION
#    - Event 1: Recurrence-free interval (RFI) event - outcome of interest
#    - Event 2: Death without recurrence - competing risk
#    - Event 0: Censored
#
# 3. CROSS-VALIDATION STRATEGY
#    - 10-fold stratified CV on RFI events (can use more folds since fast)
#    - For each fold:
#        a) Scale continuous features (mean=0, sd=1)
#        b) Fit Fine-Gray subdistribution hazard model with ALL clinical variables
#        c) Predict cumulative incidence functions (CIF) on test set
#        d) Evaluate time-dependent AUC, Brier score
#
# 4. WHY THIS APPROACH?
#    - Small number of features (~6-10): No feature selection needed
#    - All variables are clinically relevant: Include all in model
#    - Simple CV: Appropriate when not doing hyperparameter tuning
#    - Fine-Gray model: Properly models subdistribution hazard for competing risks
#
# KEY STATISTICAL CONCEPTS:
# =========================
# - Subdistribution hazard: Hazard of event of interest accounting for competing risks
# - Cumulative Incidence Function (CIF): P(event by time t | baseline covariates)
# - IPCW: Inverse probability of censoring weighting for time-dependent metrics
################################################################################

################################################################################
# LIBRARIES
################################################################################

library(readr)        # Fast CSV reading
library(dplyr)        # Data manipulation
library(survival)     # Survival analysis (Surv objects)
library(cmprsk)       # Competing risks regression
library(caret)        # Cross-validation fold creation
library(riskRegression) # Score() function for competing risks metrics
library(prodlim)      # prodlim::Hist()

setwd("~/PhD_Workspace/PredictRecurrence/")
source("./src/finegray_functions.R")

################################################################################
# SETTINGS
################################################################################

COHORT_NAME <- "ERpHER2n"  # "TNBC", "ERpHER2n", or "All"
OUTPUT_BASE_DIR <- "./output/FineGray"

################################################################################
# INPUT/OUTPUT SETTINGS
################################################################################

# Input files
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
  "Clinical"
)
dir.create(current_output_dir, recursive = TRUE, showWarnings = FALSE)

################################################################################
# SETUP LOGGING
################################################################################

script_name <- "finegray_clinical"
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
log_filename <- sprintf("%s_%s.log", script_name, timestamp)
log_path <- file.path(current_output_dir, log_filename)

# Start logging
log_con <- file(log_path, open = "wt")
sink(log_con, type = "output", split = TRUE)
sink(log_con, type = "message")

cat(sprintf("%s\n", paste(rep("=", 80), collapse = "")))
cat(sprintf("FINE-GRAY COMPETING RISKS - CLINICAL VARIABLES ONLY\n"))
cat(sprintf("Started: %s\n", Sys.time()))
cat(sprintf("Cohort: %s\n", COHORT_NAME))
cat(sprintf("Log file: %s\n", log_path))
cat(sprintf("%s\n\n", paste(rep("=", 80), collapse = "")))

################################################################################
# PARAMETERS
################################################################################

# Clinical variables
ADMIN_CENSORING_CUTOFF <- if (COHORT_NAME == "TNBC") 5.01 else NULL
CLIN_CATEGORICAL <- c("NHG", "LN")
CLIN_CONT <- c("Age", "Size.mm")

# Cross-validation
N_FOLDS <- 5  # Can use more folds since no inner CV

# Performance evaluation timepoints (in years)
EVAL_TIMES <- seq(1, 10)

################################################################################
# LOAD AND PREPARE DATA
################################################################################

start_time <- Sys.time()
cat(sprintf("\n========== LOADING DATA ==========\n"))

# Load clinical data only
train_ids <- read_csv(
  COHORT_TRAIN_IDS_PATHS[[COHORT_NAME]], 
  col_names = FALSE, 
  show_col_types = FALSE
)[[1]]

clinical_data <- read_csv(INFILE_CLINICAL, show_col_types = FALSE)
clinical_data <- as.data.frame(clinical_data)
rownames(clinical_data) <- clinical_data$Sample
clinical_data <- clinical_data[train_ids, , drop = FALSE]

cat(sprintf("Loaded %d samples\n", nrow(clinical_data)))

# Apply administrative censoring if needed
if (!is.null(ADMIN_CENSORING_CUTOFF)) {
  clinical_data <- apply_admin_censoring(
    clinical_data, "RFi_years", "RFi_event", ADMIN_CENSORING_CUTOFF
  )
}

# Prepare clinical variables
clinical_data$LN <- gsub("N\\+", "Np", clinical_data$LN)
clin <- clinical_data[c(CLIN_CONT, CLIN_CATEGORICAL)]

# One-hot encode categorical variables
encoded_result <- onehot_encode_clinical(clin, CLIN_CATEGORICAL)
X <- encoded_result$encoded_df
clinvars <- c(CLIN_CONT, encoded_result$encoded_cols)

cat(sprintf("\nClinical variables (%d total):\n", length(clinvars)))
cat(sprintf("  Continuous: %s\n", paste(CLIN_CONT, collapse=", ")))
cat(sprintf("  Categorical (one-hot): %s\n", 
            paste(encoded_result$encoded_cols, collapse=", ")))

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

################################################################################
# CROSS-VALIDATION SETUP
################################################################################

cat(sprintf("\n========== CROSS-VALIDATION SETUP ==========\n"))
cat(sprintf("Folds: %d\n", N_FOLDS))
cat(sprintf("Strategy: Stratified on RFI events\n"))

# Create stratified folds based on RFI events
set.seed(123)
cv_folds <- createFolds(
  y = clinical_data$RFi_event,
  k = N_FOLDS,
  list = TRUE,
  returnTrain = FALSE  # Returns test indices
)

# Storage for results
fold_results <- list()

################################################################################
# CROSS-VALIDATION LOOP
################################################################################

for (fold_idx in 1:N_FOLDS) {
  cat(sprintf("\n========== FOLD %d/%d ==========\n", fold_idx, N_FOLDS))
  
  fold_start_time <- Sys.time()
  
  # --------------------------------------------------------------------------
  # Split Data
  # --------------------------------------------------------------------------
  
  test_idx <- cv_folds[[fold_idx]]
  train_idx <- setdiff(1:nrow(X), test_idx)
  
  X_train <- X[train_idx, , drop = FALSE]
  X_test <- X[test_idx, , drop = FALSE]
  
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
  # Scale Continuous Features
  # --------------------------------------------------------------------------
  # Scale Age and Size.mm using training set parameters
  # Leave one-hot encoded variables (NHG, LN) as 0/1
  
  cat(sprintf("\n--- Scaling Features ---\n"))
  scale_res <- scale_continuous_features(
    X_train = X_train, 
    X_test = X_test, 
    dont_scale = encoded_result$encoded_cols
  )
  
  X_train_scaled <- scale_res$X_train_scaled
  X_test_scaled <- scale_res$X_test_scaled
  
  # --------------------------------------------------------------------------
  # Fit Fine-Gray Model
  # --------------------------------------------------------------------------
  # Fit on ALL clinical variables - no feature selection
  
  cat(sprintf("\n--- Fitting Fine-Gray Model ---\n"))
  
  fgr_train_data <- cbind(
    clinical_train[c("time_to_CompRisk_event", "CompRisk_event_coded")], 
    X_train_scaled
  )
  fgr_test_data <- cbind(
    clinical_test[c("time_to_CompRisk_event", "CompRisk_event_coded")], 
    X_test_scaled
  )
  
  # Build formula with all clinical features
  feature_cols <- colnames(X_train_scaled)
  formula_str <- paste("Hist(time_to_CompRisk_event, CompRisk_event_coded) ~", 
                       paste(feature_cols, collapse = " + "))
  formula_fg <- as.formula(formula_str)
  
  fgr_model <- FGR(
    formula = formula_fg,
    data = fgr_train_data,
    cause = 1
  )
  
  cat(sprintf("Fine-Gray model fitted with %d features\n", length(feature_cols)))
  
  # --------------------------------------------------------------------------
  # Evaluate Performance on Test Set
  # --------------------------------------------------------------------------
  
  score_result <- Score(
    list("FGR" = fgr_model),
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
  auc_fgr <- score_result$AUC$score[model == "FGR"]
  brier_fgr <- score_result$Brier$score[model == "FGR"]
  
  # Extract actual evaluation times
  eval_times <- auc_fgr$times
  
  # Create named vectors dynamically based on actual times
  auc_cols <- setNames(auc_fgr$AUC, paste0("auc_", eval_times, "yr"))
  brier_cols <- setNames(brier_fgr$Brier, paste0("brier_", eval_times, "yr"))
  
  # Combine into one-row dataframe
  performance <- data.frame(
    model = "FGR",
    mean_auc = mean(auc_fgr$AUC, na.rm = TRUE),
    final_ibs = brier_fgr[times == max(times), IBS],
    as.list(auc_cols),
    as.list(brier_cols)
  )
  
  # Round all numeric columns
  performance[, -1] <- round(performance[, -1], 3)
  
  print(performance)
  
  # --------------------------------------------------------------------------
  # Extract Predictions
  # --------------------------------------------------------------------------
  
  pred_risks <- predictRisk(
    fgr_model,
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
  
  # --------------------------------------------------------------------------
  # Extract Coefficients
  # --------------------------------------------------------------------------
  
  fg_coef <- fgr_model$crrFit$coef
  coef_df <- data.frame(
    feature = names(fg_coef),
    coefficient = as.vector(fg_coef),
    HR = exp(as.vector(fg_coef))
  )
  
  # Sort by absolute HR
  coef_df <- coef_df[order(abs(coef_df$HR - 1), decreasing = TRUE), ]
  rownames(coef_df) <- NULL
  
  cat("\nModel Coefficients:\n")
  print(coef_df)
  
  # --------------------------------------------------------------------------
  # Store Fold Results
  # --------------------------------------------------------------------------
  
  fold_runtime <- as.numeric(difftime(Sys.time(), fold_start_time, units = "secs"))
  cat(sprintf("\n✓ Fold %d completed in %.1f seconds\n", fold_idx, fold_runtime))
  
  fold_results[[fold_idx]] <- list(
    fold_idx = fold_idx,
    train_samples = rownames(X_train),
    test_samples = rownames(X_test),
    n_train = nrow(X_train),
    n_test = nrow(X_test),
    fold_runtime = fold_runtime,
    
    # Model and results
    finegray_model = fgr_model,
    coefficients = coef_df,
    performance = performance,
    predictions = fold_predictions,
    
    # Scaling parameters (for applying to new data)
    scale_params = scale_res[c("centers", "scales", "cont_cols")]
  )
}

################################################################################
# AGGREGATE RESULTS ACROSS FOLDS
################################################################################

cat(sprintf("\n========== AGGREGATED PERFORMANCE ==========\n"))

# Extract all performance metrics
performance_list <- lapply(fold_results, function(fold) fold$performance)
all_performance <- do.call(rbind, performance_list)

# Calculate summary statistics
metric_cols <- setdiff(names(all_performance), "model")
performance_summary <- data.frame(
  metric = metric_cols,
  mean = sapply(metric_cols, function(col) mean(all_performance[[col]], na.rm = TRUE)),
  se = sapply(metric_cols, function(col) {
    sd(all_performance[[col]], na.rm = TRUE) / sqrt(nrow(all_performance))
  }),
  sd = sapply(metric_cols, function(col) sd(all_performance[[col]], na.rm = TRUE))
)

# Add 95% CI
z_score <- qnorm(0.975)
performance_summary$ci_lower <- performance_summary$mean - z_score * performance_summary$se
performance_summary$ci_upper <- performance_summary$mean + z_score * performance_summary$se

# Round
performance_summary[, c("mean", "se", "sd", "ci_lower", "ci_upper")] <- 
  round(performance_summary[, c("mean", "se", "sd", "ci_lower", "ci_upper")], 3)

rownames(performance_summary) <- NULL

cat("\nIndividual Fold Performance:\n")
print(all_performance)

cat("\nPerformance Summary (Mean ± SE, 95% CI):\n")
for (i in 1:nrow(performance_summary)) {
  cat(sprintf("%15s: %.3f ± %.3f (95%% CI: %.3f - %.3f)\n",
              performance_summary$metric[i],
              performance_summary$mean[i],
              performance_summary$se[i],
              performance_summary$ci_lower[i],
              performance_summary$ci_upper[i]))
}

# Combine predictions from all folds
cv_predictions <- do.call(rbind, lapply(fold_results, function(fold) {
  fold$predictions
}))

################################################################################
# ASSESS COEFFICIENT STABILITY
################################################################################

cat(sprintf("\n========== COEFFICIENT STABILITY ==========\n"))

# Extract coefficients from all folds
all_coefs <- lapply(fold_results, function(fold) fold$coefficients)

# Get all features
all_features <- unique(unlist(lapply(all_coefs, function(df) df$feature)))

# Calculate stability metrics
coef_stability <- lapply(all_features, function(feat) {
  coefs <- sapply(all_coefs, function(df) {
    idx <- which(df$feature == feat)
    if (length(idx) == 0) return(NA)
    df$coefficient[idx]
  })
  
  data.frame(
    feature = feat,
    mean_coef = mean(coefs, na.rm = TRUE),
    sd_coef = sd(coefs, na.rm = TRUE),
    mean_HR = exp(mean(coefs, na.rm = TRUE)),
    stringsAsFactors = FALSE
  )
})

coef_stability <- do.call(rbind, coef_stability)
coef_stability <- coef_stability[order(abs(coef_stability$mean_HR - 1), decreasing = TRUE), ]
rownames(coef_stability) <- NULL

coef_stability$mean_coef <- round(coef_stability$mean_coef, 3)
coef_stability$sd_coef <- round(coef_stability$sd_coef, 3)
coef_stability$mean_HR <- round(coef_stability$mean_HR, 3)

cat("\nCoefficient Stability Across Folds:\n")
print(coef_stability)

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
# Scale Features
# --------------------------------------------------------------------------

cat(sprintf("\n--- Scaling Features ---\n"))
scale_res_all <- scale_continuous_features(
  X_train = X_all, 
  X_test = NULL, 
  dont_scale = encoded_result$encoded_cols
)

X_all_scaled <- scale_res_all$X_train_scaled

# --------------------------------------------------------------------------
# Fit Final Fine-Gray Model
# --------------------------------------------------------------------------

cat(sprintf("\n--- Fitting Final Fine-Gray Model ---\n"))

fgr_all_data <- cbind(
  clinical_all[c("time_to_CompRisk_event", "CompRisk_event_coded")], 
  X_all_scaled
)

feature_cols <- colnames(X_all_scaled)
formula_str <- paste("Hist(time_to_CompRisk_event, CompRisk_event_coded) ~", 
                     paste(feature_cols, collapse = " + "))
formula_fg <- as.formula(formula_str)

fgr_final <- FGR(
  formula = formula_fg,
  data = fgr_all_data,
  cause = 1
)

cat(sprintf("Final model fitted with %d features\n", length(feature_cols)))

# --------------------------------------------------------------------------
# Extract Final Model Coefficients
# --------------------------------------------------------------------------

fg_final_coef <- fgr_final$crrFit$coef
coef_final <- data.frame(
  feature = names(fg_final_coef),
  coefficient = as.vector(fg_final_coef),
  HR = exp(as.vector(fg_final_coef))
)

# Sort by absolute HR
coef_final <- coef_final[order(abs(coef_final$HR - 1), decreasing = TRUE), ]
rownames(coef_final) <- NULL

coef_final$coefficient <- round(coef_final$coefficient, 3)
coef_final$HR <- round(coef_final$HR, 3)

cat("\nFinal Model Coefficients:\n")
print(coef_final)

# --------------------------------------------------------------------------
# Calculate Variable Importance
# --------------------------------------------------------------------------

cat("\n--- Fine-Gray Variable Importance ---\n")

vimp_final <- calculate_fgr_importance(
  fgr_model = fgr_final,
  encoded_cols = encoded_result$encoded_cols,
  verbose = TRUE
)

################################################################################
# SAVE RESULTS
################################################################################

cat("\n========== SAVING RESULTS ==========\n")

results_dir <- file.path(current_output_dir, "final_results")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

# --------------------------------------------------------------------------
# Save RData File
# --------------------------------------------------------------------------

save(
  # Performance
  performance_summary,
  all_performance,
  cv_predictions,
  
  # Stability
  coef_stability,
  
  # Final model
  fgr_final,
  coef_final,
  vimp_final,
  
  # Scaling parameters (critical for new data)
  scale_res_all,
  encoded_result,
  
  # CV results
  fold_results,
  
  # Parameters
  N_FOLDS,
  EVAL_TIMES,
  COHORT_NAME,
  clinvars,
  
  file = file.path(results_dir, "all_results.RData")
)

# --------------------------------------------------------------------------
# Save CSVs
# --------------------------------------------------------------------------

write.csv(performance_summary, 
          file.path(results_dir, "performance_summary.csv"), 
          row.names = FALSE)

write.csv(all_performance, 
          file.path(results_dir, "performance_all_folds.csv"), 
          row.names = FALSE)

write.csv(cv_predictions, 
          file.path(results_dir, "cv_predictions.csv"), 
          row.names = FALSE)

write.csv(vimp_final, 
          file.path(results_dir, "variable_importance.csv"), 
          row.names = FALSE)

write.csv(coef_final, 
          file.path(results_dir, "final_coefficients.csv"), 
          row.names = FALSE)

write.csv(coef_stability, 
          file.path(results_dir, "coefficient_stability.csv"), 
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
cat("  - coefficient_stability.csv\n")

total_runtime <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
cat(sprintf("\nTotal runtime: %.1f minutes (%.1f seconds per fold)\n", 
            total_runtime, 
            (total_runtime * 60) / N_FOLDS))

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

sink(type = "message")
sink(type = "output")
close(log_con)

cat(sprintf("\n✓ Pipeline completed successfully!\n"))
cat(sprintf("✓ Log file: %s\n", log_path))