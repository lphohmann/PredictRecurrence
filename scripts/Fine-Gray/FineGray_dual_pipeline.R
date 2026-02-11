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

################################################################################
# COMMAND LINE ARGUMENTS
################################################################################

args <- commandArgs(trailingOnly = TRUE)

# Defaults
DEFAULT_COHORT <- "TNBC"
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

script_name <- "finegray_dual_pipeline"
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

# Feature selection - variance filter only
VARIANCE_QUANTILE <- 0.75

# Cross-validation
N_OUTER_FOLDS <- 5
N_INNER_FOLDS <- 3
ALPHA_GRID <- c(0.5, 0.7, 0.9)

# Stability threshold for final model
STABILITY_THRESHOLD_FG <- 0.4

# Performance evaluation timepoints (in years)
EVAL_TIMES <- seq(1, 10)
if (COHORT_NAME =="TNBC") {
  EVAL_TIMES <- EVAL_TIMES[1:5]
  ALPHA_GRID <- c(0.5)
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
#encoded_result$encoded_df
# i have to combined only for trainign th unpen fg models at the en of nCV outer folds
#X <- cbind(X, encoded_result$encoded_df)
#clinvars <- c(CLIN_CONT, encoded_result$encoded_cols)

#VARS_PRESERVE <- NULL
#VARS_NO_PENALIZATION <- NULL

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
  
  X_train_filtered <- prepare_filtered_features(
    X = X_train,
    vars_preserve = NULL,
    variance_quantile = VARIANCE_QUANTILE,
    apply_cox_filter = FALSE)
  
  # PURPOSE: Select features predictive of recurrence
  # Cox proportional hazards with elastic net regularization
  rfi_model <- tune_and_fit_coxnet(
    X_train = X_train_filtered,
    y_train = y_rfi_train,
    clinical_train = clinical_train,
    event_col = "RFi_event",
    alpha_grid = ALPHA_GRID,
    penalty_factor = rep(1, ncol(X_train_filtered)),
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
  # INNER CV LOOP #2: CoxNet for Death without Recurrence
  # --------------------------------------------------------------------------
  
  # Feature Filtering on Outer Training Data
  cat(sprintf("\n========== COXNET FOR DEATH w/o RECURRENCE ==========\n"))
  
  # PURPOSE: Select features predictive of death (competing risk)
  # This ensures we account for features that predict competing events
  
  death_model <- tune_and_fit_coxnet(
    X_train = X_train_filtered,
    y_train = y_death_train,
    clinical_train = clinical_train,
    event_col = "DeathNoR_event",
    alpha_grid = ALPHA_GRID,
    penalty_factor = rep(1, ncol(X_train_filtered)),
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
  # --------------------------------------------------------------------------
  
  features_pooled <- union(features_rfi, features_death)
  
  cat(sprintf("\n--- Feature Pooling ---\n"))
  cat(sprintf("RFI features: %d\n", length(features_rfi)))
  cat(sprintf("Death features: %d\n", length(features_death)))
  cat(sprintf("Overlap: %d\n", length(intersect(features_rfi, features_death))))
  cat(sprintf("Pooled total: %d\n", length(features_pooled)))
  
  # Check if any features were selected
  if (length(features_pooled) == 0) {
    cat(sprintf("Fold %d: No features selected by either model. Skipping fold.", fold_idx))
    warning(sprintf("Fold %d: No features selected by either model. Skipping fold.", fold_idx))
    
    outer_fold_results[[fold_idx]] <- list(
      metadata = list(
        fold_idx = fold_idx,
        skipped = TRUE,
        reason = "No features selected by either CoxNet model"
      )
    )
    next
    
  }
  
  # Prepare Fine-Gray input
  X_pooled_train <- X_train[, features_pooled, drop = FALSE]
  X_pooled_test <- X_test[, features_pooled, drop = FALSE]
  
  # --------------------------------------------------------------------------
  # Fit Penalized Fine-Gray Model (MRS Construction)
  # --------------------------------------------------------------------------
  # dont do feature selection only do RS calcualtion
  
  cat(sprintf("\n========== PENALIZED FINE-GRAY (MRS CONSTRUCTION) ==========\n"))
  
  pFG_res <- fit_penalized_finegray_cv(
    X_input = X_pooled_train[, features_pooled, drop = FALSE],
    clinical_input = clinical_train,
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
  mrs_train <- calculate_methylation_risk_score(
    X_data = X_pooled_train,
    cpg_coefficients = pFG_cpg_coefs,
    scaling_params = pFG_cpg_scaling,
    verbose = TRUE
  )
  
  # Test MRS (using same scaling)
  mrs_test <- calculate_methylation_risk_score(
    X_data = X_pooled_test,
    cpg_coefficients = pFG_cpg_coefs,
    scaling_params = pFG_cpg_scaling,
    verbose = TRUE
  )
  
  MRS_train <- mrs_train$mrs
  MRS_test <- mrs_test$mrs
  
  # Scale the MRS itself (optional)

  mrs_train_scaled <- scale(MRS_train)
  mrs_scaling <- list(
    center = attr(mrs_train_scaled, "scaled:center"),
    scale = attr(mrs_train_scaled, "scaled:scale")
  )
  
  MRS_train <- as.numeric(mrs_train_scaled)
  MRS_test <- as.numeric(scale(MRS_test, 
                               center = mrs_scaling$center, 
                               scale = mrs_scaling$scale))
  
  cat("  MRS scaled to mean=0, sd=1\n")
  
  # --------------------------------------------------------------------------
  # Prepare Data for Final Unpenalized Fine-Gray Model
  # --------------------------------------------------------------------------

  cat(sprintf("\n--- Preparing Final Model Data ---\n"))
  
  # Verify row alignment
  if (!identical(rownames(clinical_train), rownames(X_pooled_train))) {
    stop("Row names don't match between clinical_train and X_pooled_train")
  }
  
  # Build training data
  fgr_train_data <- data.frame(
    time_to_CompRisk_event = clinical_train$time_to_CompRisk_event,
    CompRisk_event_coded = clinical_train$CompRisk_event_coded,
    encoded_result$encoded_df[rownames(X_pooled_train), , drop = FALSE],
    methylation_risk_score = MRS_train,
    row.names = rownames(clinical_train)
  )
  
  # Build test data
  fgr_test_data <- data.frame(
    time_to_CompRisk_event = clinical_test$time_to_CompRisk_event,
    CompRisk_event_coded = clinical_test$CompRisk_event_coded,
    encoded_result$encoded_df[rownames(X_pooled_test), , drop = FALSE],
    methylation_risk_score = MRS_test,
    row.names = rownames(clinical_test)
  )
  
  # Scale clinical continuous variables
  clin_train_scaled <- scale(fgr_train_data[, CLIN_CONT, drop = FALSE])
  clin_scaling <- list(
    variables = CLIN_CONT,
    center = attr(clin_train_scaled, "scaled:center"),
    scale = attr(clin_train_scaled, "scaled:scale")
  )
  
  clin_test_scaled <- scale(
    fgr_test_data[, CLIN_CONT, drop = FALSE],
    center = clin_scaling$center,
    scale = clin_scaling$scale
  )
  
  fgr_train_data[, CLIN_CONT] <- clin_train_scaled
  fgr_test_data[, CLIN_CONT] <- clin_test_scaled
  
  cat(sprintf("  Final model predictors: %s\n", 
              paste(setdiff(colnames(fgr_train_data), 
                            c("time_to_CompRisk_event", "CompRisk_event_coded")), 
                    collapse=", ")))
  
  # --------------------------------------------------------------------------
  # Fit Unpenalized Fine-Gray Model 1: only MRS

  cat(sprintf("\n========== UNPENALIZED FINE-GRAY MODELS ==========\n"))
  
  fgr_MRS <- fit_fine_gray_model(
    fgr_data = fgr_train_data[,!colnames(fgr_train_data) %in% names(encoded_result$encoded_df)],
    cr_time = "time_to_CompRisk_event",
    cr_event = "CompRisk_event_coded",
    cause = 1
  )
  
  fgr_MRS <- fgr_MRS$model
  
  # Fit Unpenalized Fine-Gray Model 2: only clin
  
  fgr_CLIN <- fit_fine_gray_model(
    fgr_data = fgr_train_data[,colnames(fgr_train_data) != "methylation_risk_score"],
    cr_time = "time_to_CompRisk_event",
    cr_event = "CompRisk_event_coded",
    cause = 1
  )
  
  fgr_CLIN <- fgr_CLIN$model
  
  # Fit Unpenalized Fine-Gray Model 3: both

  fgr_COMBINED <- fit_fine_gray_model(
    fgr_data = fgr_train_data,
    cr_time = "time_to_CompRisk_event",
    cr_event = "CompRisk_event_coded",
    cause = 1
  )
  
  fgr_COMBINED <- fgr_COMBINED$model
  
  # --------------------------------------------------------------------------
  # Evaluate Final Model Performance on Test Set
  # --------------------------------------------------------------------------
  
  cat(sprintf("\n--- Test Set Performance ---\n"))
  
  score_fg <- Score(
    list("FGR_MRS" = fgr_MRS,
         "FGR_CLIN" = fgr_CLIN,
         "FGR_COMBINED" = fgr_COMBINED),
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
  
  fgr_performances <- list()
  for(fgmodel in c("FGR_CLIN","FGR_MRS","FGR_COMBINED")) {
    
    # Extract performance metrics
    auc_model <- score_fg$AUC$score[model == fgmodel]
    brier_model <- score_fg$Brier$score[model == fgmodel]
    eval_times <- auc_model$times
    
    auc_cols <- setNames(auc_model$AUC, paste0("auc_", eval_times, "yr"))
    brier_cols <- setNames(brier_model$Brier, paste0("brier_", eval_times, "yr"))
    
    fgr_perf <- data.frame(
      model = fgmodel,
      mean_auc = mean(auc_model$AUC, na.rm = TRUE),
      final_ibs = brier_model[times == max(times), IBS],
      as.list(auc_cols),
      as.list(brier_cols)
    )
    
    fgr_perf[, -1] <- round(fgr_perf[, -1], 2)
    fgr_performances <- append(fgr_performances,list(fgr_perf))
  }
  fgr_performances <- do.call(rbind,fgr_performances)
  print(fgr_performances)
  
  # --------------------------------------------------------------------------
  # Extract and Store Coefficients
  # --------------------------------------------------------------------------
  
  cat(sprintf("\n--- Extracting Model Coefficients ---\n"))
  
  # 1. CoxNet: Comparison table
  names(coef_rfi_df) <- c("feature", "cox_rfi_coef")
  names(coef_death_df) <- c("feature", "cox_death_coef")
  coef_coxnet <- merge(coef_rfi_df, coef_death_df, by = "feature", all = TRUE)
  coef_coxnet$cox_rfi_HR <- ifelse(is.na(coef_coxnet$cox_rfi_coef), 
                                   NA, 
                                   exp(coef_coxnet$cox_rfi_coef))
  coef_coxnet$cox_death_HR <- ifelse(is.na(coef_coxnet$cox_death_coef), 
                                     NA, 
                                     exp(coef_coxnet$cox_death_coef))
  coef_coxnet <- coef_coxnet[c("feature", "cox_rfi_coef", "cox_rfi_HR", 
                               "cox_death_coef", "cox_death_HR")]
  rownames(coef_coxnet) <- NULL
  
  # 2. Penalized FG: CpG coefficients for MRS
  coef_penalized_fg <- data.frame(
    feature = pFG_selected_cpgs,
    pen_fg_coef = pFG_cpg_coefs,
    pen_fg_HR = exp(pFG_cpg_coefs),
    stringsAsFactors = FALSE,
    row.names = NULL
  )

  # 3. FG coefficients
  fg_models <- list(CLIN = fgr_CLIN,MRS = fgr_MRS,COMBINED = fgr_COMBINED)
  fg_coefs <- list()
  
  for (nm in names(fg_models)) {
    fgm <- fg_models[[nm]]
    
    fg_coef <- fgm$crrFit$coef
    
    fg_coefs[[nm]] <- data.frame(
      feature = names(fg_coef),
      fg_coef = as.vector(fg_coef),
      fg_HR = exp(as.vector(fg_coef)),
      stringsAsFactors = FALSE
    )
  }
  #fg_coefs$MRS
  #fg_coefs$COMBINED
  
  # --------------------------------------------------------------------------
  # Store Fold Results
  # --------------------------------------------------------------------------

  outer_fold_results[[fold_idx]] <- list(
    
    # ========== FOLD METADATA ==========
    metadata = list(
      fold_idx = fold_idx,
      included_clinical_features = c(encoded_result$encoded_cols,CLIN_CONT),
      train_samples = rownames(X_train),
      test_samples = rownames(X_test)
    ),
    
    # ========== COXNET FEATURE SELECTION ==========
    coxnet = list(
      features_rfi = features_rfi,
      features_death = features_death,
      coefficients = coef_coxnet
    ),
    
    # ========== PENALIZED FINE-GRAY (MRS) ==========
    penalized_fg = list(
      input_cpgs = pFG_selected_cpgs,
      coefficients = coef_penalized_fg
    ),
    
    # ========== FINAL FINE-GRAY MODEL ==========
    unpenalized_fgs = list(
      train_data = fgr_train_data,
      test_data = fgr_test_data,
      models = fg_models,
      coefficients = fg_coefs,
      performances = fgr_performances)
  )
  
  fold_runtime <- as.numeric(difftime(Sys.time(), fold_start_time, units = "mins"))
  cat(sprintf("\n✓ Fold %d completed in %.1f minutes\n", fold_idx, fold_runtime))
}

################################################################################
# SAVE RAW CV RESULTS (SAFETY BACKUP)
################################################################################
#load("./output/FineGray/ERpHER2n/Combined/Unadjusted/outer_fold_results.RData")

save(outer_fold_results, file = file.path(current_output_dir, 
                                          "outer_fold_results.RData"))
cat("Outer fold results saved (backup)\n")
# 
# ################################################################################
# # AGGREGATE CV RESULTS
# ################################################################################
# 
# perf_results <- aggregate_cv_performance(outer_fold_results)
# 
# stability_results <- assess_finegray_stability(
#   outer_fold_results
# )
# 
# ################################################################################
# # SAVE COMPLETE CV RESULTS (OVERWRITES BACKUP)
# ################################################################################
# 
# cv_results <- list(
#   outer_fold_results = outer_fold_results,
#   perf_results = perf_results,
#   stability_results = stability_results,
#   metadata = list(
#     N_OUTER_FOLDS = N_OUTER_FOLDS,
#     EVAL_TIMES = EVAL_TIMES,
#     clinvars = clinvars,
#     COHORT_NAME = COHORT_NAME,
#     DATA_MODE = DATA_MODE
#   )
# )
# 
# save(cv_results, file = file.path(current_output_dir, 
#                                   "outer_fold_results.RData"))
# cat("Outer fold results saved (complete with aggregations)\n")
# 
# 
# #
# #load("~/PhD_Workspace/PredictRecurrence/output/FineGray/ERpHER2n/Methylation/Unadjusted/outer_fold_results.RData")
# #load(file.path(current_output_dir, "outer_fold_results.RData"))
# 
# ################################################################################
# # TRAIN FINAL MODEL ON ALL DATA
# ################################################################################
# 
# cat(sprintf("\n========== TRAINING FINAL MODEL ==========\n"))
# 
# X_all <- X
# clinical_all <- clinical_data
# 
# cat(sprintf("Training samples: n=%d (RFi=%d, Death=%d)\n",
#             nrow(X_all),
#             sum(clinical_all$RFi_event),
#             sum(clinical_all$DeathNoR_event)))
# 
# # ----------------------------------------------------------------------------
# # Collect CV-Discovered Features
# # ----------------------------------------------------------------------------
# cat(sprintf("\n--- Collecting CV-Discovered Features ---\n"))
# 
# stability_metrics <- stability_results$stability_metrics
# 
# # Select stable non-clinical features
# stable_cpgs <- stability_metrics$feature[
#   stability_metrics$selection_freq >= STABILITY_THRESHOLD_FG &
#     stability_metrics$direction_consistent == TRUE
# ]
# 
# cat(sprintf(
#   "Including CpGs with selection_freq ≥ %.2f and consistent direction: n = %d\n",
#   STABILITY_THRESHOLD_FG,
#   length(stable_cpgs)
# ))
# 
# # Always include clinical variables
# stable_features <- c(
#   included_clinvars, 
#   stable_cpgs
# )
# 
# #X_all <- X_all[, stable_features, drop = FALSE]
# #X_all_stable_features[, stable_cpgs, drop = FALSE]
# #--------------------------------------------------------------------------
# # Fit Penalized Fine-Gray Model (MRS Construction)
# # --------------------------------------------------------------------------
# coxnet_selected_cpgs <- setdiff(colnames(X_pooled_train), included_clinvars)
# 
# cat(sprintf("\n========== PENALIZED FINE-GRAY (MRS CONSTRUCTION) ==========\n"))
# 
# pFG_res <- fit_penalized_finegray_cv(
#   X_input = X_all[, stable_cpgs, drop = FALSE],
#   clinical_input = clinical_all,
#   alpha_seq = 0,#c(0.1, 0.2, 0.3, 0.4, 0.5),
#   lambda_seq = NULL,
#   cr_time = "time_to_CompRisk_event",
#   cr_event = "CompRisk_event_coded",
#   penalty = "RIDGE",#"ENET",#
#   n_inner_folds = 3
# )
# 
# # Extract selected CpGs and their coefficients
# pFG_selected_cpgs <- pFG_res$results_table$feature[pFG_res$results_table$selected]
# pFG_cpg_coefs <- setNames(
#   pFG_res$results_table$pFG_coefficient[pFG_res$results_table$selected],
#   pFG_selected_cpgs
# )
# 
# # Extract scaling parameters (CpGs were scaled inside pFG model)
# pFG_cpg_scaling <- list(
#   center = setNames(
#     pFG_res$results_table$scale_center[pFG_res$results_table$selected],
#     pFG_selected_cpgs
#   ),
#   scale = setNames(
#     pFG_res$results_table$scale_scale[pFG_res$results_table$selected],
#     pFG_selected_cpgs
#   )
# )
# 
# cat(sprintf("  BEST PENALIZED FG: Alpha=%.2f, Lambda=%.6f, CpGs=%d\n",
#             pFG_res$best_alpha,
#             pFG_res$best_lambda,
#             length(pFG_selected_cpgs)))
# cat("    Selected CpGs:", paste(pFG_selected_cpgs, collapse=", "), "\n")
# 
# # --------------------------------------------------------------------------
# # Calculate Methylation Risk Score (MRS)
# # --------------------------------------------------------------------------
# 
# # Training MRS
# mrs_all <- calculate_methylation_risk_score(
#   X_data = X_all[, stable_cpgs, drop = FALSE],
#   cpg_coefficients = pFG_cpg_coefs,
#   scaling_params = pFG_cpg_scaling,
#   verbose = TRUE
# )
# MRS_all <- mrs_all$mrs
# 
# # Scale the MRS itself (optional)
# scale_mrs <- TRUE
# mrs_scaling <- NULL
# if (scale_mrs) {
#   mrs_train_scaled <- scale(MRS_all)
#   mrs_scaling <- list(
#     center = attr(mrs_train_scaled, "scaled:center"),
#     scale = attr(mrs_train_scaled, "scaled:scale")
#   )
#   
#   MRS_all <- as.numeric(mrs_train_scaled)
#   cat("  MRS scaled to mean=0, sd=1\n")
# }
# 
# # --------------------------------------------------------------------------
# # Prepare Data for Final Unpenalized Fine-Gray Model
# # --------------------------------------------------------------------------
# 
# cat(sprintf("\n--- Preparing Final Model Data ---\n"))
# 
# # Verify row alignment
# if (!identical(rownames(clinical_all), rownames(X_all))) {
#   stop("Row names don't match between clinical_train and X_pooled_train")
# }
# 
# # Build training data
# fgr_all_data <- data.frame(
#   time_to_CompRisk_event = clinical_all$time_to_CompRisk_event,
#   CompRisk_event_coded = clinical_all$CompRisk_event_coded,
#   X_all[, included_clinvars, drop = FALSE],
#   methylation_risk_score = MRS_all,
#   row.names = rownames(clinical_all)
# )
# 
# if (DATA_MODE != "methylation") {
#   # Scale clinical continuous variables
#   clin_all_scaled <- scale(fgr_all_data[, CLIN_CONT, drop = FALSE])
#   clin_scaling <- list(
#     variables = CLIN_CONT,
#     center = attr(clin_all_scaled, "scaled:center"),
#     scale = attr(clin_all_scaled, "scaled:scale")
#   )
#   fgr_all_data[, CLIN_CONT] <- clin_all_scaled
# }
# 
# cat(sprintf("  Final model predictors: %s\n", 
#             paste(setdiff(colnames(fgr_all_data), 
#                           c("time_to_CompRisk_event", "CompRisk_event_coded")), 
#                   collapse=", ")))
# 
# # --------------------------------------------------------------------------
# # Fit Unpenalized Fine-Gray Model (FINAL MODEL)
# # --------------------------------------------------------------------------
# 
# cat(sprintf("\n========== UNPENALIZED FINE-GRAY (FINAL MODEL) ==========\n"))
# 
# fgr_final_all_result <- fit_fine_gray_model(
#   fgr_data = fgr_all_data,
#   cr_time = "time_to_CompRisk_event",
#   cr_event = "CompRisk_event_coded",
#   cause = 1
# )
# 
# fgr_final_all_model <- fgr_final_all_result$model
# print(fgr_final_all_model)
# 
# # ----------------------------------------------------------------------------
# # Calculate Variable Importance
# # ----------------------------------------------------------------------------
# 
# cat("\n--- Fine-Gray Variable Importance ---\n")
# 
# vimp_fg_final <- calculate_fgr_importance(
#   fgr_model = fgr_final_all_model,
#   encoded_cols = if (DATA_MODE == "methylation") NULL else encoded_result$encoded_cols,
#   verbose = FALSE
# )
# 
# print(vimp_fg_final)
# 
# ################################################################################
# # SAVE FINAL MODEL RESULTS
# ################################################################################
# 
# final_results <- list(
#   fgr_final = fgr_final_all_model,
#   vimp_fg_final = vimp_fg_final,
#   metadata = list(
#     N_OUTER_FOLDS = N_OUTER_FOLDS,
#     EVAL_TIMES = EVAL_TIMES,
#     clinvars = clinvars,
#     COHORT_NAME = COHORT_NAME,
#     DATA_MODE = DATA_MODE
#   )
# )
# 
# save(final_results, file = file.path(current_output_dir, 
#                                      "final_fg_results.RData"))
# cat("Final model results saved\n")
# 
# 
# 
# ################################################################################
# # CLEANUP AND FINISH
# ################################################################################
# 
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
