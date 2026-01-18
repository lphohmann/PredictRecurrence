#!/usr/bin/env Rscript

################################################################################
# Script: Fine-Gray Competing Risks Pipeline with fastcmprsk
# Author: lennart hohmann
################################################################################

################################################################################
# IMPORTS
################################################################################

library(readr)
library(dplyr)
library(survival)
library(glmnet)
#library(fastcmprsk)  # For penalized Fine-Gray models
library(cmprsk)

# Source custom functions
setwd("~/PhD_Workspace/PredictRecurrence/")

# ==============================================================================
# USER SETTINGS
# ==============================================================================

COHORT_NAME <- "ERpHER2n"
DATA_MODE <- "combined"
TRAIN_CPGS <- "./data/set_definitions/CpG_prefiltered_sets/cpg_ids_atac_overlap.txt"
OUTPUT_BASE_DIR <- "./output/FineGray_fastcmprsk"

# ==============================================================================
# INPUT FILES
# ==============================================================================

INFILE_METHYLATION <- "./data/train/train_methylation_unadjusted.csv"
COHORT_TRAIN_IDS_PATHS <- list(
  TNBC = "./data/train/train_subcohorts/TNBC_train_ids.csv",
  ERpHER2n = "./data/train/train_subcohorts/ERpHER2n_train_ids.csv",
  All = "./data/train/train_subcohorts/All_train_ids.csv"
)
INFILE_CLINICAL <- "./data/train/train_clinical.csv"

# ==============================================================================
# OUTPUT DIRECTORY
# ==============================================================================

current_output_dir <- file.path(OUTPUT_BASE_DIR, COHORT_NAME, 
                                tools::toTitleCase(DATA_MODE), "Unadjusted")
dir.create(current_output_dir, recursive = TRUE, showWarnings = FALSE)

# Output files
outfile_coxnet_rfi <- file.path(current_output_dir, "coxnet_rfi_model.rds")
outfile_coxnet_death <- file.path(current_output_dir, "coxnet_death_model.rds")
outfile_finegray_model <- file.path(current_output_dir, "finegray_final_model.rds")

# Logfile
#logfile_path <- file.path(current_output_dir, "pipeline_run.log")
#log_con <- file(logfile_path, open = "wt")
#sink(log_con, type = "output")
#sink(log_con, type = "message")

# ==============================================================================
# PARAMETERS
# ==============================================================================

ADMIN_CENSORING_CUTOFF <- if (COHORT_NAME == "TNBC") 5.01 else NULL
CLINVARS_INCLUDED <- c("Age", "Size.mm", "NHG", "LN")
CLIN_CATEGORICAL <- c("NHG", "LN")
VARIANCE_QUANTILE <- 0.75
N_UNIVARIATE_KEEP <- 1000
ALPHA_GRID <- c(0.7)#c(0.3, 0.5, 0.7)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

load_training_data <- function(train_ids, beta_path, clinical_path) {
  clinical_data <- read_csv(clinical_path, show_col_types = FALSE)
  clinical_data <- as.data.frame(clinical_data)
  rownames(clinical_data) <- clinical_data$Sample
  clinical_data <- clinical_data[train_ids, , drop = FALSE]
  
  beta_matrix <- read_csv(beta_path, show_col_types = FALSE)
  beta_matrix <- as.data.frame(beta_matrix)
  rownames(beta_matrix) <- beta_matrix[[1]]
  beta_matrix <- as.data.frame(t(beta_matrix[, -1]))
  beta_matrix <- beta_matrix[train_ids, , drop = FALSE]
  
  cat("Loaded training data.\n")
  return(list(beta_matrix = beta_matrix, clinical_data = clinical_data))
}

beta_to_m <- function(beta, beta_threshold = 1e-3) {
  was_df <- is.data.frame(beta)
  if (was_df) {
    row_names <- rownames(beta)
    col_names <- colnames(beta)
    beta <- as.matrix(beta)
  }
  beta <- pmax(pmin(beta, 1 - beta_threshold), beta_threshold)
  m_values <- log2(beta / (1 - beta))
  if (was_df) {
    m_values <- as.data.frame(m_values)
    rownames(m_values) <- row_names
    colnames(m_values) <- col_names
  }
  return(m_values)
}

apply_admin_censoring <- function(df, time_col, event_col, time_cutoff = 5.0) {
  mask <- df[[time_col]] > time_cutoff
  df[mask, time_col] <- time_cutoff
  df[mask, event_col] <- 0
  cat(sprintf("Applied censoring at %.1f for %s/%s.\n", time_cutoff, time_col, event_col))
  return(df)
}

subset_methylation <- function(mval_matrix, cpg_ids_file) {
  cpg_ids <- trimws(readLines(cpg_ids_file)[readLines(cpg_ids_file) != ""])
  valid_cpgs <- cpg_ids[cpg_ids %in% colnames(mval_matrix)]
  cat(sprintf("Subsetted to %d CpGs.\n", length(valid_cpgs)))
  return(mval_matrix[, valid_cpgs, drop = FALSE])
}

onehot_encode_clinical <- function(clin, clin_categorical) {
  for (var in clin_categorical) {
    clin[[var]] <- if (var == "LN") factor(clin[[var]], levels = c("N0", "N+")) else as.factor(clin[[var]])
  }
  continuous_vars <- setdiff(colnames(clin), clin_categorical)
  clin_encoded <- if (length(continuous_vars) > 0) clin[, continuous_vars, drop = FALSE] else data.frame(row.names = rownames(clin))
  encoded_cols <- c()
  
  for (var in clin_categorical) {
    dummy_df <- as.data.frame(model.matrix(as.formula(paste("~", var, "- 1")), data = clin))
    rownames(dummy_df) <- rownames(clin)
    encoded_cols <- c(encoded_cols, colnames(dummy_df))
    clin_encoded <- cbind(clin_encoded, dummy_df)
  }
  return(list(encoded_df = clin_encoded, encoded_cols = encoded_cols))
}

variance_filter <- function(x, variance_quantile, keep_vars = NULL) {
  keep_list <- if (is.null(keep_vars)) character(0) else as.character(keep_vars)
  pool_cols <- setdiff(colnames(x), keep_list)
  if (is.null(variance_quantile)) stop("variance_quantile required")
  
  if (length(pool_cols) > 0) {
    variances <- apply(x[, pool_cols, drop = FALSE], 2, stats::var, na.rm = TRUE)
    var_threshold <- quantile(variances, probs = variance_quantile, na.rm = TRUE)
    selected <- names(variances[variances >= var_threshold])
    cat(sprintf("\t%d features selected by variance\n", length(selected)))
  } else {
    selected <- character(0)
  }
  return(c(keep_list, selected))
}

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

start_time <- Sys.time()
cat(sprintf("Pipeline started: %s\n", start_time))

# ==============================================================================
# LOAD DATA
# ==============================================================================

train_ids <- read_csv(COHORT_TRAIN_IDS_PATHS[[COHORT_NAME]], col_names = FALSE, show_col_types = FALSE)[[1]]
data_list <- load_training_data(train_ids, INFILE_METHYLATION, INFILE_CLINICAL)
beta_matrix <- data_list$beta_matrix
clinical_data <- data_list$clinical_data

mvals <- beta_to_m(beta_matrix, beta_threshold = 0.001)

if (!is.null(ADMIN_CENSORING_CUTOFF)) {
  clinical_data <- apply_admin_censoring(clinical_data, "RFi_years", "RFi_event", ADMIN_CENSORING_CUTOFF)
}

if (!is.null(TRAIN_CPGS)) {
  mvals <- subset_methylation(mvals, TRAIN_CPGS)
}

# ==============================================================================
# ONE-HOT ENCODING
# ==============================================================================

X <- mvals
if (!is.null(CLINVARS_INCLUDED)) {
  clin <- clinical_data[CLINVARS_INCLUDED]
  clin <- clin[rownames(X), , drop = FALSE]
  encoded_result <- onehot_encode_clinical(clin, CLIN_CATEGORICAL)
  X <- cbind(X, encoded_result$encoded_df)
  clinvars_included_encoded <- c(setdiff(CLINVARS_INCLUDED, CLIN_CATEGORICAL), encoded_result$encoded_cols)
  
  cat(sprintf("Added clinical variables: %s\n", paste(clinvars_included_encoded, collapse = ", ")))
} else {
  clinvars_included_encoded <- NULL
}

# ==============================================================================
# DEFINE VARIABLES TO PRESERVE AND NOT PENALIZE (after encoding)
# ==============================================================================

# Variables that should never be filtered out during variance/univariate filtering
VARS_PRESERVE <- NULL#c("Age", "Size.mm", "NHG3", "LNN+")

# Variables that should not be penalized in CoxNet and Fine-Gray models
# Can include all clinical variables or a subset
VARS_NO_PENALIZATION <- NULL#c("Age", "Size.mm", "NHG3", "LNN+")

cat(sprintf("\nVARS_PRESERVE: %s\n", paste(VARS_PRESERVE, collapse = ", ")))
cat(sprintf("VARS_NO_PENALIZATION: %s\n", paste(VARS_NO_PENALIZATION, collapse = ", ")))

# Validate that these variables exist in X
missing_preserve <- setdiff(VARS_PRESERVE, colnames(X))
missing_no_pen <- setdiff(VARS_NO_PENALIZATION, colnames(X))

if (length(missing_preserve) > 0) {
  warning(sprintf("VARS_PRESERVE contains non-existent columns: %s", 
                  paste(missing_preserve, collapse = ", ")))
  VARS_PRESERVE <- intersect(VARS_PRESERVE, colnames(X))
}

if (length(missing_no_pen) > 0) {
  warning(sprintf("VARS_NO_PENALIZATION contains non-existent columns: %s",
                  paste(missing_no_pen, collapse = ", ")))
  VARS_NO_PENALIZATION <- intersect(VARS_NO_PENALIZATION, colnames(X))
}

# ==============================================================================
# CREATE COMPETING RISKS OUTCOMES
# ==============================================================================

clinical_data$DeathNoR_event <- as.integer(clinical_data$OS_event == 1 & clinical_data$RFi_event == 0)
clinical_data$DeathNoR_years <- clinical_data$OS_years
clinical_data$CompRisk_event_coded <- 0
clinical_data$CompRisk_event_coded[clinical_data$RFi_event == 1] <- 1
clinical_data$CompRisk_event_coded[clinical_data$OS_event == 1 & clinical_data$RFi_event == 0] <- 2
clinical_data$time_to_CompRisk_event <- pmin(clinical_data$RFi_years, clinical_data$OS_years)

cat(sprintf("Competing risks: Censored=%d, RFI=%d, Death=%d\n",
            sum(clinical_data$CompRisk_event_coded == 0),
            sum(clinical_data$CompRisk_event_coded == 1),
            sum(clinical_data$CompRisk_event_coded == 2)))

y_rfi <- Surv(clinical_data$RFi_years, clinical_data$RFi_event)
y_death_no_r <- Surv(clinical_data$DeathNoR_years, clinical_data$DeathNoR_event)

# ==============================================================================
# VARIANCE FILTERING
# ==============================================================================

cat("\n=== Variance Filtering ===\n")
selected_features <- variance_filter(X, variance_quantile = VARIANCE_QUANTILE, keep_vars = VARS_PRESERVE)
X_filtered <- X[, selected_features, drop = FALSE]

# ==============================================================================
# UNIVARIATE FILTERING, weird to only do for rfi i think
# ==============================================================================

cat("\n=== Univariate Cox Filtering ===\n")
is_preserved <- colnames(X_filtered) %in% VARS_PRESERVE
X_preserved <- X_filtered[, is_preserved, drop = FALSE]
X_to_filter <- X_filtered[, !is_preserved, drop = FALSE]
X_matrix <- as.matrix(X_to_filter)[rownames(clinical_data), , drop = FALSE]

univariate_pvals <- apply(X_matrix, 2, function(x) {
  summary(coxph(y_rfi ~ x))$coefficients[1, "Pr(>|z|)"]
})

n_keep <- min(N_UNIVARIATE_KEEP, ncol(X_matrix))
top_features_univ <- names(sort(univariate_pvals)[1:n_keep])
X_matrix_filtered <- cbind(X_matrix[, top_features_univ, drop = FALSE],
                           as.matrix(X_preserved[rownames(X_matrix), , drop = FALSE]))

cat(sprintf("After univariate: %d features\n", ncol(X_matrix_filtered)))

# ==============================================================================
# DUAL COXNET FEATURE SELECTION
# ==============================================================================

cat("\n=== Dual CoxNet Feature Selection ===\n")

# Create penalty factor: 0 for unpenalized variables, 1 for others
penalty_factor <- rep(1, ncol(X_matrix_filtered))
names(penalty_factor) <- colnames(X_matrix_filtered)

if (!is.null(VARS_NO_PENALIZATION)) {
  penalty_factor[colnames(X_matrix_filtered) %in% VARS_NO_PENALIZATION] <- 0
  cat(sprintf("Penalty factor: %d unpenalized, %d penalized\n",
              sum(penalty_factor == 0), sum(penalty_factor == 1)))
} else {
  cat("All variables will be penalized (VARS_NO_PENALIZATION = NULL)\n")
}

# CoxNet #1: RFI
cat("\n--- CoxNet #1: RFI ---\n")
cv_rfi_results <- list()
for (alpha_val in ALPHA_GRID) {
  set.seed(123)
  cv_fit <- cv.glmnet(x = X_matrix_filtered, y = y_rfi, family = "cox", 
                      alpha = alpha_val, penalty.factor = penalty_factor, nfolds = 5)
  best_idx <- which.min(cv_fit$cvm)
  cv_rfi_results[[as.character(alpha_val)]] <- list(
    fit = cv_fit, best_cvm = cv_fit$cvm[best_idx], alpha = alpha_val,
    n_features = sum(coef(cv_fit, s = "lambda.min") != 0)
  )
  cat(sprintf("Alpha %.1f: Deviance=%.4f, Features=%d\n", alpha_val, cv_fit$cvm[best_idx],
              sum(coef(cv_fit, s = "lambda.min") != 0)))
}
best_rfi_idx <- which.min(sapply(cv_rfi_results, function(x) x$best_cvm))
best_rfi_fit <- cv_rfi_results[[best_rfi_idx]]$fit
coef_rfi <- coef(best_rfi_fit, s = "lambda.min")
features_rfi <- rownames(coef_rfi)[coef_rfi[,1] != 0]
cat(sprintf("\nRFI selected %d features\n", length(features_rfi)))

# CoxNet #2: Death without Recurrence
cat("\n--- CoxNet #2: Death without Recurrence ---\n")
cv_death_results <- list()
for (alpha_val in ALPHA_GRID) {
  set.seed(123)
  cv_fit <- cv.glmnet(x = X_matrix_filtered, y = y_death_no_r, family = "cox", 
                      alpha = alpha_val, penalty.factor = penalty_factor, nfolds = 5)
  best_idx <- which.min(cv_fit$cvm)
  cv_death_results[[as.character(alpha_val)]] <- list(
    fit = cv_fit, best_cvm = cv_fit$cvm[best_idx], alpha = alpha_val,
    n_features = sum(coef(cv_fit, s = "lambda.min") != 0)
  )
  cat(sprintf("Alpha %.1f: Deviance=%.4f, Features=%d\n", alpha_val, cv_fit$cvm[best_idx],
              sum(coef(cv_fit, s = "lambda.min") != 0)))
}
best_death_idx <- which.min(sapply(cv_death_results, function(x) x$best_cvm))
best_death_fit <- cv_death_results[[best_death_idx]]$fit
coef_death <- coef(best_death_fit, s = "lambda.min")
features_death <- rownames(coef_death)[coef_death[,1] != 0]
cat(sprintf("\nDeath selected %d features\n", length(features_death)))

# Pool features
features_pooled <- union(features_rfi, features_death)
cat(sprintf("\n=== Pooled Features ===\n"))
cat(sprintf("Total: %d (RFI: %d, Death: %d, Overlap: %d)\n",
            length(features_pooled), length(features_rfi), length(features_death),
            length(intersect(features_rfi, features_death))))

# Save CoxNet models
saveRDS(list(rfi = best_rfi_fit, features = features_rfi, alpha = cv_rfi_results[[best_rfi_idx]]$alpha), 
        outfile_coxnet_rfi)
saveRDS(list(death = best_death_fit, features = features_death, alpha = cv_death_results[[best_death_idx]]$alpha), 
        outfile_coxnet_death)

# ==============================================================================
# UNPENALIZED FINE-GRAY MODEL
# ==============================================================================

cat("\n=== Unpenalized Fine-Gray Model ===\n")

# Prepare data for Fine-Gray model
X_pooled <- as.matrix(X_matrix_filtered[, features_pooled, drop = FALSE])

cat(sprintf("Fitting Fine-Gray model with %d pooled features\n", ncol(X_pooled)))

# Fit unpenalized Fine-Gray model using crr() from cmprsk package
library(cmprsk)

fit_fg <- crr(
  ftime = clinical_data$time_to_CompRisk_event,
  fstatus = clinical_data$CompRisk_event_coded,
  cov1 = X_pooled,
  failcode = 1,  # Event of interest (recurrence)
  cencode = 0    # Censoring code
)

# ==============================================================================
# MODEL SUMMARY
# ==============================================================================

cat("\n=== Fine-Gray Model Summary ===\n")
cat(sprintf("Number of observations: %d\n", nrow(X_pooled)))
cat(sprintf("Number of features: %d\n", ncol(X_pooled)))
cat(sprintf("  From RFI CoxNet: %d\n", length(features_rfi)))
cat(sprintf("  From Death CoxNet: %d\n", length(features_death)))
cat(sprintf("  Overlap: %d\n", length(intersect(features_rfi, features_death))))

cat(sprintf("\nEvent summary:\n"))
cat(sprintf("  Recurrence events: %d\n", sum(clinical_data$CompRisk_event_coded == 1)))
cat(sprintf("  Competing events (death): %d\n", sum(clinical_data$CompRisk_event_coded == 2)))
cat(sprintf("  Censored: %d\n", sum(clinical_data$CompRisk_event_coded == 0)))

# Print model summary
summary(fit_fg)

# Extract coefficients
coefs_fg <- fit_fg$coef
names(coefs_fg) <- features_pooled

# Get significant features (p < 0.05)
pvals <- summary(fit_fg)$coef[, "p-value"]
sig_features <- names(coefs_fg)[pvals < 0.05]

cat(sprintf("\n=== Coefficients ===\n"))
cat(sprintf("Significant features (p < 0.05): %d out of %d\n", 
            length(sig_features), length(coefs_fg)))

cat("\nAll coefficients (sorted by magnitude):\n")
print(sort(coefs_fg, decreasing = TRUE))

cat("\nSignificant coefficients only:\n")
if (length(sig_features) > 0) {
  print(sort(coefs_fg[sig_features], decreasing = TRUE))
} else {
  cat("No significant features at p < 0.05\n")
}

# ==============================================================================
# SAVE FINAL MODEL
# ==============================================================================

# Save final model
final_results <- list(
  model = fit_fg,
  coefficients = coefs_fg,
  significant_features = sig_features,
  pooled_features = features_pooled,
  features_rfi = features_rfi,
  features_death = features_death,
  summary = summary(fit_fg)
)

saveRDS(final_results, outfile_finegray_model)
cat(sprintf("\nSaved final model to: %s\n", outfile_finegray_model))

# ==============================================================================
# CLEANUP
# ==============================================================================

end_time <- Sys.time()
cat(sprintf("\nPipeline ended: %s\n", end_time))
cat(sprintf("Total time: %.2f minutes\n", as.numeric(difftime(end_time, start_time, units = "mins"))))

# Close log connections if they exist
if (exists("log_con")) {
  sink(type = "message")
  sink(type = "output")
  close(log_con)
}