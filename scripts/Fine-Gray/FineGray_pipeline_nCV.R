#!/usr/bin/env Rscript

################################################################################
# Fine-Gray Competing Risks with Nested Cross-Validation
# Author: Lennart Hohmann
################################################################################

################################################################################
# LIBRARIES
################################################################################

library(readr)
library(dplyr)
library(survival)
library(glmnet)
library(cmprsk)
library(caret)
library(timeROC)
library(data.table)

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
current_output_dir <- file.path(OUTPUT_BASE_DIR, COHORT_NAME, 
                                tools::toTitleCase(DATA_MODE), "Unadjusted")
dir.create(current_output_dir, recursive = TRUE, showWarnings = FALSE)

################################################################################
# PARAMETERS
################################################################################

# Clinical variables
ADMIN_CENSORING_CUTOFF <- if (COHORT_NAME == "TNBC") 5.01 else NULL
CLINVARS_INCLUDED <- c("Age", "Size.mm", "NHG", "LN")
CLIN_CATEGORICAL <- c("NHG", "LN")

# Feature selection - variance filter only
VARIANCE_QUANTILE <- 0.75

# Variables to preserve during filtering (after one-hot encoding)
VARS_PRESERVE <- "Size.mm"  # e.g., c("Age", "Size.mm", "NHG3", "LNN+")

# Variables to not penalize in elastic net (after one-hot encoding)
VARS_NO_PENALIZATION <- "Size.mm"  # e.g., c("Age", "Size.mm", "NHG3", "LNN+")

# Cross-validation
N_OUTER_FOLDS <- 5
N_INNER_FOLDS <- 3
ALPHA_GRID <- c(0.5, 0.7, 0.9)

# Performance evaluation timepoints (in years)
EVAL_TIMES <- c(1, 3, 5, 10)

################################################################################
# HELPER FUNCTIONS
################################################################################

load_training_data <- function(train_ids, beta_path, clinical_path) {
  
  # Load clinical data
  clinical_data <- read_csv(clinical_path, show_col_types = FALSE)
  clinical_data <- as.data.frame(clinical_data)
  rownames(clinical_data) <- clinical_data$Sample
  clinical_data <- clinical_data[train_ids, , drop = FALSE]
  
  # Load methylation data - use fread instead of read_csv
  cat("Loading methylation data...\n")
  beta_matrix <- fread(beta_path, header = TRUE, data.table = FALSE)
  
  rownames(beta_matrix) <- beta_matrix[[1]]
  beta_matrix <- beta_matrix[, -1]
  
  # Transpose
  cat("Transposing...\n")
  beta_matrix <- t(beta_matrix)
  
  # Subset to train_ids
  beta_matrix <- beta_matrix[train_ids, , drop = FALSE]
  beta_matrix <- as.data.frame(beta_matrix)
  
  cat(sprintf("Loaded %d samples x %d CpGs\n", nrow(beta_matrix), ncol(beta_matrix)))
  
  return(list(beta_matrix = beta_matrix, clinical_data = clinical_data))
}

beta_to_m <- function(beta, beta_threshold = 1e-3) {
  # Convert beta values to M-values
  was_df <- is.data.frame(beta)
  if (was_df) {
    row_names <- rownames(beta)
    col_names <- colnames(beta)
    beta <- as.matrix(beta)
  }
  
  # Prevent log of 0 or 1
  beta <- pmax(pmin(beta, 1 - beta_threshold), beta_threshold)
  m_values <- log2(beta / (1 - beta))
  
  if (was_df) {
    m_values <- as.data.frame(m_values)
    rownames(m_values) <- row_names
    colnames(m_values) <- col_names
  }
  return(m_values)
}

apply_admin_censoring <- function(df, time_col, event_col, time_cutoff) {
  # Apply administrative censoring at specified time
  mask <- df[[time_col]] > time_cutoff
  df[mask, time_col] <- time_cutoff
  df[mask, event_col] <- 0
  cat(sprintf("Applied censoring at %.1f for %s/%s (n=%d).\n", 
              time_cutoff, time_col, event_col, sum(mask)))
  return(df)
}

subset_methylation <- function(mval_matrix, cpg_ids_file) {
  # Subset to predefined CpGs
  cpg_ids <- trimws(readLines(cpg_ids_file))
  cpg_ids <- cpg_ids[cpg_ids != ""]
  valid_cpgs <- cpg_ids[cpg_ids %in% colnames(mval_matrix)]
  cat(sprintf("Subsetted to %d CpGs (from %d in file).\n", 
              length(valid_cpgs), length(cpg_ids)))
  return(mval_matrix[, valid_cpgs, drop = FALSE])
}

onehot_encode_clinical <- function(clin, clin_categorical) {
  # One-hot encode categorical variables
  # Continuous variables are kept as-is
  
  for (var in clin_categorical) {
    if (var == "LN") {
      clin[[var]] <- factor(clin[[var]], levels = c("N0", "N+"))
    } else {
      clin[[var]] <- as.factor(clin[[var]])
    }
  }
  
  # Keep continuous variables
  continuous_vars <- setdiff(colnames(clin), clin_categorical)
  clin_encoded <- if (length(continuous_vars) > 0) {
    clin[, continuous_vars, drop = FALSE]
  } else {
    data.frame(row.names = rownames(clin))
  }
  
  # One-hot encode categorical
  encoded_cols <- c()
  for (var in clin_categorical) {
    dummy_df <- as.data.frame(model.matrix(
      as.formula(paste("~", var, "- 1")), 
      data = clin
    ))
    rownames(dummy_df) <- rownames(clin)
    encoded_cols <- c(encoded_cols, colnames(dummy_df))
    clin_encoded <- cbind(clin_encoded, dummy_df)
  }
  
  return(list(
    encoded_df = clin_encoded, 
    encoded_cols = encoded_cols
  ))
}

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

# Convert to M-values
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

if (!is.null(CLINVARS_INCLUDED)) {
  clin <- clinical_data[CLINVARS_INCLUDED]
  clin <- clin[rownames(X), , drop = FALSE]
  encoded_result <- onehot_encode_clinical(clin, CLIN_CATEGORICAL)
  X <- cbind(X, encoded_result$encoded_df)
  
  cat(sprintf("Added clinical variables: %s\n", 
              paste(colnames(encoded_result$encoded_df), collapse = ", ")))
}

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

# Create outcomes for competing risks
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
cat(sprintf("  Death without RFI: %d (%.1f%%)\n", 
            sum(clinical_data$CompRisk_event_coded == 2),
            100 * mean(clinical_data$CompRisk_event_coded == 2)))

# Create survival objects
y_rfi <- Surv(clinical_data$RFi_years, clinical_data$RFi_event)
y_death_no_r <- Surv(clinical_data$DeathNoR_years, clinical_data$DeathNoR_event)

################################################################################
# NESTED CROSS-VALIDATION
################################################################################

cat(sprintf("\n========== NESTED CROSS-VALIDATION ==========\n"))
cat(sprintf("Outer folds: %d\n", N_OUTER_FOLDS))
cat(sprintf("Inner folds: %d\n", N_INNER_FOLDS))
cat(sprintf("Alpha grid: %s\n", paste(ALPHA_GRID, collapse = ", ")))

# Create stratified outer folds based on RFI events
set.seed(123)
outer_folds <- createFolds(
  y = clinical_data$RFi_event,
  k = N_OUTER_FOLDS,
  list = TRUE,
  returnTrain = FALSE  # Returns test indices
)

# Storage for results
outer_fold_results <- list()
outer_fold_predictions <- list()

################################################################################
# OUTER CV LOOP - Performance estimation
################################################################################
fold_idx=1
for (fold_idx in 1:N_OUTER_FOLDS) {
  cat(sprintf("\n========== OUTER FOLD %d/%d ==========\n", 
              fold_idx, N_OUTER_FOLDS))
  
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
  # Feature filtering on outer training data
  # --------------------------------------------------------------------------
  
  cat(sprintf("\n--- Feature Filtering ---\n"))
  
  # Separate preserved vs filterable features
  is_preserved <- colnames(X_train) %in% VARS_PRESERVE
  X_train_preserved <- X_train[, is_preserved, drop = FALSE]
  X_train_to_filter <- X_train[, !is_preserved, drop = FALSE]
  
  cat(sprintf("Features before filtering: %d (preserved: %d)\n", 
              ncol(X_train), sum(is_preserved)))
  
  # Variance filtering
  variances <- apply(X_train_to_filter, 2, var, na.rm = TRUE)
  var_threshold <- quantile(variances, probs = VARIANCE_QUANTILE, na.rm = TRUE)
  keep_var <- names(variances[variances >= var_threshold])
  X_filtered <- X_train_to_filter[, keep_var, drop = FALSE]
  
  cat(sprintf("After variance filter (>= %.0fth percentile): %d features\n", 
              VARIANCE_QUANTILE * 100, length(keep_var)))
  
  # Combine filtered features with preserved features
  X_train_filtered <- cbind(X_filtered, X_train_preserved)
  X_test_filtered <- X_test[, colnames(X_train_filtered), drop = FALSE]
  
  cat(sprintf("Total features for modeling: %d\n", ncol(X_train_filtered)))
  
  # --------------------------------------------------------------------------
  # INNER CV LOOP #1: CoxNet for RFI
  # --------------------------------------------------------------------------
  
  cat(sprintf("\n--- Inner CV: Tuning CoxNet for RFI ---\n"))
  
  # Create penalty factor (0 = not penalized, 1 = penalized)
  penalty_factor_rfi <- rep(1, ncol(X_train_filtered))
  if (!is.null(VARS_NO_PENALIZATION)) {
    penalty_factor_rfi[colnames(X_train_filtered) %in% VARS_NO_PENALIZATION] <- 0
  }
  
  # Create stratified inner folds
  set.seed(123)
  inner_folds_rfi <- createFolds(
    y = clinical_train$RFi_event,
    k = N_INNER_FOLDS,
    list = TRUE,
    returnTrain = FALSE
  )
  
  # Create foldid vector for cv.glmnet
  foldid_rfi <- rep(NA, nrow(X_train_filtered))
  for (i in 1:N_INNER_FOLDS) {
    foldid_rfi[inner_folds_rfi[[i]]] <- i
  }
  
  # Test each alpha value
  cv_rfi_inner <- list()
  for (alpha_val in ALPHA_GRID) {
    set.seed(123)
    cv_fit <- cv.glmnet(
      x = as.matrix(X_train_filtered),
      y = y_rfi_train,
      family = "cox",
      alpha = alpha_val,
      penalty.factor = penalty_factor_rfi,
      foldid = foldid_rfi
      #nfolds = N_INNER_FOLDS # info in foldid
    )
    
    # performance at lambda.min
    perf_min <- cv_fit$cvm[cv_fit$lambda == cv_fit$lambda.min]
    
    # selected features
    beta <- coef(cv_fit, s = "lambda.min")
    selected_features <- rownames(beta)[as.vector(beta != 0)]
    
    cv_rfi_inner[[as.character(alpha_val)]] <- list(
      alpha = alpha_val,
      lambda = cv_fit$lambda.min,
      cvm_min = perf_min,
      cvm_se = cv_fit$cvsd[cv_fit$lambda == cv_fit$lambda.min],
      n_features = length(selected_features),
      features = selected_features,
      fit = cv_fit
    )
    
    cat(sprintf(
      "  Alpha=%.2f: CV-deviance=%.4f (SE=%.4f), Lambda=%.6f, Features=%d\n",
      alpha_val,
      perf_min,
      cv_fit$cvsd[cv_fit$lambda == cv_fit$lambda.min],
      cv_fit$lambda.min,
      length(selected_features)
    ))
    
  }
  
  # Select best hyperparams based on CV performance
  best_rfi_idx <- which.min(sapply(cv_rfi_inner, function(x) x$cvm_min))
  best_rfi_result <- cv_rfi_inner[[best_rfi_idx]]
  best_alpha <- best_rfi_result$alpha
  best_lambda <- best_rfi_result$lambda
  
  
  # fit model on whole trainig set
  final_fit_outer <- glmnet(
    x = as.matrix(X_train_filtered),
    y = y_rfi_train,
    family = "cox",
    alpha = best_alpha,
    lambda = best_lambda,
    penalty.factor = penalty_factor_rfi
  )
  
  # Extract selected features in that final model
  coef_rfi <- coef(final_fit_outer)
  features_rfi <- rownames(coef_rfi)[as.vector(coef_rfi != 0)]
  
  cat(sprintf(
    "  BEST FINAL MODEL: Alpha=%.2f, Lambda=%.6f, Selected %d RFI features\n",
    best_alpha, best_lambda, length(features_rfi)
  ))
  cat("    Selected Features:", paste(features_rfi, collapse=", "), "\n")
  
  # --------------------------------------------------------------------------
  # INNER CV LOOP #2: CoxNet for Death without Recurrence
  # --------------------------------------------------------------------------
  
  cat(sprintf("\n--- Inner CV: Tuning CoxNet for Death ---\n"))
  
  # Create penalty factor
  penalty_factor_death <- rep(1, ncol(X_train_filtered))
  if (!is.null(VARS_NO_PENALIZATION)) {
    penalty_factor_death[colnames(X_train_filtered) %in% VARS_NO_PENALIZATION] <- 0
  }
  
  # Create stratified inner folds
  set.seed(124)  # Different seed for independent split
  inner_folds_death <- createFolds(
    y = clinical_train$DeathNoR_event,
    k = N_INNER_FOLDS,
    list = TRUE,
    returnTrain = FALSE
  )
  
  # Create foldid vector
  foldid_death <- rep(NA, nrow(X_train_filtered))
  for (i in 1:N_INNER_FOLDS) {
    foldid_death[inner_folds_death[[i]]] <- i
  }
  
  # Test each alpha value
  cv_death_inner <- list()
  for (alpha_val in ALPHA_GRID) {
    set.seed(124)
    cv_fit <- cv.glmnet(
      x = as.matrix(X_train_filtered),
      y = y_death_train,
      family = "cox",
      alpha = alpha_val,
      penalty.factor = penalty_factor_death,
      foldid = foldid_death,
      nfolds = N_INNER_FOLDS
    )
    
    best_idx <- which.min(cv_fit$cvm)
    cv_death_inner[[as.character(alpha_val)]] <- list(
      fit = cv_fit,
      best_cvm = cv_fit$cvm[best_idx],
      alpha = alpha_val,
      lambda = cv_fit$lambda.min,
      n_features = sum(coef(cv_fit, s = "lambda.min") != 0)
    )
    
    cat(sprintf("  Alpha=%.1f: CV-deviance=%.4f, Lambda=%.6f, Features=%d\n",
                alpha_val, 
                cv_fit$cvm[best_idx],
                cv_fit$lambda.min,
                sum(coef(cv_fit, s = "lambda.min") != 0)))
  }
  
  # Select best alpha
  best_death_idx <- which.min(sapply(cv_death_inner, function(x) x$best_cvm))
  best_death_result <- cv_death_inner[[best_death_idx]]
  
  # Extract selected features
  coef_death <- coef(best_death_result$fit, s = "lambda.min")
  features_death <- rownames(coef_death)[coef_death[, 1] != 0]
  
  cat(sprintf("  BEST: Alpha=%.1f, Selected %d Death features\n",
              best_death_result$alpha, length(features_death)))
  
  # --------------------------------------------------------------------------
  # Pool features from both models
  # --------------------------------------------------------------------------
  
  features_pooled <- union(features_rfi, features_death)
  
  cat(sprintf("\n--- Feature Pooling ---\n"))
  cat(sprintf("RFI features: %d\n", length(features_rfi)))
  cat(sprintf("Death features: %d\n", length(features_death)))
  cat(sprintf("Overlap: %d\n", length(intersect(features_rfi, features_death))))
  cat(sprintf("Pooled total: %d\n", length(features_pooled)))
  
  # --------------------------------------------------------------------------
  # Fit Fine-Gray model on pooled features
  # --------------------------------------------------------------------------
  
  cat(sprintf("\n--- Fitting Fine-Gray Model ---\n"))
  
  X_pooled_train <- X_train_filtered[, features_pooled, drop = FALSE]
  X_pooled_test <- X_test_filtered[, features_pooled, drop = FALSE]
  
  fit_fg <- crr(
    ftime = clinical_train$time_to_CompRisk_event,
    fstatus = clinical_train$CompRisk_event_coded,
    cov1 = as.matrix(X_pooled_train),
    failcode = 1,  # Event of interest (RFI)
    cencode = 0    # Censored
  )
  
  cat(sprintf("Fine-Gray model fitted with %d features\n", ncol(X_pooled_train)))
  
  # --------------------------------------------------------------------------
  # Predict on test set
  # --------------------------------------------------------------------------
  
  cat(sprintf("\n--- Test Set Predictions ---\n"))
  
  # Predict cumulative incidence function
  pred_fg <- predict(fit_fg, cov1 = as.matrix(X_pooled_test))
  
  # Calculate linear predictor (risk score)
  risk_scores_test <- as.matrix(X_pooled_test) %*% fit_fg$coef
  
  cat(sprintf("Generated predictions for %d test samples\n", nrow(X_test)))
  
  # --------------------------------------------------------------------------
  # Store results
  # --------------------------------------------------------------------------
  
  outer_fold_results[[fold_idx]] <- list(
    fold = fold_idx,
    train_idx = train_idx,
    test_idx = test_idx,
    n_train = nrow(X_train),
    n_test = nrow(X_test),
    n_events_train_rfi = sum(clinical_train$RFi_event),
    n_events_train_death = sum(clinical_train$DeathNoR_event),
    n_events_test_rfi = sum(clinical_test$RFi_event),
    n_events_test_death = sum(clinical_test$DeathNoR_event),
    # Feature selection
    n_features_after_variance = ncol(X_after_var_filter),
    n_features_after_univariate = ncol(X_filtered),
    n_features_preserved = ncol(X_train_preserved),
    n_features_total = ncol(X_train_filtered),
    features_rfi = features_rfi,
    features_death = features_death,
    features_pooled = features_pooled,
    # Hyperparameters
    alpha_rfi = best_rfi_result$alpha,
    lambda_rfi = best_rfi_result$lambda,
    alpha_death = best_death_result$alpha,
    lambda_death = best_death_result$lambda,
    # Models
    model_fg = fit_fg,
    cv_fit_rfi = best_rfi_result$fit,
    cv_fit_death = best_death_result$fit
  )
  
  outer_fold_predictions[[fold_idx]] <- data.frame(
    fold = fold_idx,
    sample_idx = test_idx,
    sample_id = rownames(X_test),
    risk_score = as.numeric(risk_scores_test),
    time = clinical_test$time_to_CompRisk_event,
    event = clinical_test$CompRisk_event_coded,
    rfi_event = clinical_test$RFi_event,
    death_event = clinical_test$DeathNoR_event
  )
  
  cat(sprintf("Fold %d complete.\n", fold_idx))
}

################################################################################
# AGGREGATE AND SUMMARIZE RESULTS
################################################################################

cat(sprintf("\n\n========== AGGREGATED RESULTS ==========\n"))

# Combine all predictions
all_predictions <- do.call(rbind, outer_fold_predictions)

# Feature selection stability across folds
all_features <- unique(unlist(lapply(outer_fold_results, 
                                     function(x) x$features_pooled)))
feature_selection_freq <- sapply(all_features, function(feat) {
  sum(sapply(outer_fold_results, function(x) feat %in% x$features_pooled))
})

cat(sprintf("\n--- Feature Selection Stability ---\n"))
cat(sprintf("Total unique features selected: %d\n", length(all_features)))
cat(sprintf("Features in all %d folds: %d\n", 
            N_OUTER_FOLDS, sum(feature_selection_freq == N_OUTER_FOLDS)))
cat(sprintf("Features in >= %d folds: %d\n", 
            ceiling(N_OUTER_FOLDS / 2), 
            sum(feature_selection_freq >= ceiling(N_OUTER_FOLDS / 2))))

# Show most stable features
n_show <- min(20, length(feature_selection_freq))
stable_features <- names(sort(feature_selection_freq, decreasing = TRUE)[1:n_show])

cat(sprintf("\nMost stable features (top %d):\n", n_show))
stable_df <- data.frame(
  Feature = stable_features,
  Frequency = feature_selection_freq[stable_features],
  Percentage = sprintf("%.0f%%", 100 * feature_selection_freq[stable_features] / N_OUTER_FOLDS)
)
print(stable_df, row.names = FALSE)

# Model complexity summary
cat(sprintf("\n--- Model Complexity Summary ---\n"))
n_features_pooled <- sapply(outer_fold_results, function(x) length(x$features_pooled))
cat(sprintf("Features per fold (pooled): %.1f ± %.1f [%d-%d]\n",
            mean(n_features_pooled), sd(n_features_pooled),
            min(n_features_pooled), max(n_features_pooled)))

n_features_rfi <- sapply(outer_fold_results, function(x) length(x$features_rfi))
cat(sprintf("Features per fold (RFI only): %.1f ± %.1f [%d-%d]\n",
            mean(n_features_rfi), sd(n_features_rfi),
            min(n_features_rfi), max(n_features_rfi)))

n_features_death <- sapply(outer_fold_results, function(x) length(x$features_death))
cat(sprintf("Features per fold (Death only): %.1f ± %.1f [%d-%d]\n",
            mean(n_features_death), sd(n_features_death),
            min(n_features_death), max(n_features_death)))

################################################################################
# SAVE RESULTS
################################################################################

nested_cv_results <- list(
  # Results
  outer_fold_results = outer_fold_results,
  predictions = all_predictions,
  feature_stability = feature_selection_freq,
  
  # Parameters
  cv_params = list(
    n_outer_folds = N_OUTER_FOLDS,
    n_inner_folds = N_INNER_FOLDS,
    alpha_grid = ALPHA_GRID,
    variance_quantile = VARIANCE_QUANTILE,
    n_univariate_keep = N_UNIVARIATE_KEEP,
    vars_preserve = VARS_PRESERVE,
    vars_no_penalization = VARS_NO_PENALIZATION
  ),
  
  # Data info
  cohort_info = list(
    cohort_name = COHORT_NAME,
    n_samples = nrow(X),
    n_features_original = ncol(X),
    n_rfi_events = sum(clinical_data$RFi_event),
    n_death_events = sum(clinical_data$DeathNoR_event),
    n_censored = sum(clinical_data$CompRisk_event_coded == 0)
  )
)

# Save results
output_file <- file.path(current_output_dir, "nested_cv_results.rds")
saveRDS(nested_cv_results, output_file)
cat(sprintf("\nSaved results to: %s\n", output_file))

# Save predictions as CSV for easy access
pred_file <- file.path(current_output_dir, "nested_cv_predictions.csv")
write_csv(all_predictions, pred_file)
cat(sprintf("Saved predictions to: %s\n", pred_file))

# Save feature stability
feature_stability_df <- data.frame(
  Feature = names(feature_selection_freq),
  Selection_Frequency = feature_selection_freq,
  Percentage = 100 * feature_selection_freq / N_OUTER_FOLDS
) %>% arrange(desc(Selection_Frequency), Feature)

feature_file <- file.path(current_output_dir, "feature_stability.csv")
write_csv(feature_stability_df, feature_file)
cat(sprintf("Saved feature stability to: %s\n", feature_file))

cat(sprintf("\n========== PIPELINE COMPLETE ==========\n"))
cat(sprintf("Total runtime: %.1f minutes\n", 
            as.numeric(difftime(Sys.time(), start_time, units = "mins"))))