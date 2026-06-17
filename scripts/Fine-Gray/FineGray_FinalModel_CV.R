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

COHORT_NAME <- "ERpHER2n" # 'TNBC', 'ERpHER2n', 'All'
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
#log_con <- file(log_path, open = "wt")
#sink(log_con, type = "output", split = TRUE)
#sink(log_con, type = "message")

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
predictor_clin <- colnames(encoded_result$encoded_df)

fgr_MRS_data <- fgr_data[, c("time_to_CompRisk_event", "CompRisk_event_coded", "MeRS")]
fgr_CLIN_data <- fgr_data[, c("time_to_CompRisk_event", "CompRisk_event_coded", predictor_clin)]
fgr_COMBINED_data <- fgr_data[, c("time_to_CompRisk_event", "CompRisk_event_coded", predictor_clin, "MeRS")]

fgr_COMBINED <- fit_fine_gray_model(
  fgr_data = fgr_COMBINED_data,
  cr_time = "time_to_CompRisk_event",
  cr_event = "CompRisk_event_coded",
  cause = 1
)

fgr_COMBINED <- fgr_COMBINED$model

# mers
fgr_MRS <- fit_fine_gray_model(
  fgr_data = fgr_MRS_data,
  cr_time = "time_to_CompRisk_event",
  cr_event = "CompRisk_event_coded",
  cause = 1
)
fgr_MRS <- fgr_MRS$model

# only clin
fgr_CLIN <- fit_fine_gray_model(
  fgr_data = fgr_CLIN_data,
  cr_time = "time_to_CompRisk_event",
  cr_event = "CompRisk_event_coded",
  cause = 1
)

fgr_CLIN <- fgr_CLIN$model

###############################################################################
# Variable Importance for Combined model
###############################################################################

# # Calculate Variable Importance
# cat("\n--- Fine-Gray Variable Importance ---\n")
# vimp_fgr_COMBINED <- calculate_fgr_importance(
#   fgr_model = fgr_COMBINED,
#   encoded_cols = encoded_result$encoded_cols,
#   verbose = FALSE
# )
# print(vimp_fgr_COMBINED)

###############################################################################
# CALCULATE RISK SCORE CUTOFF FOR DICHOTOMIZATION
###############################################################################
cat(sprintf("\n--- Timepoint Risk Score Cutoffs ---\n"))

fgr_data$RFi_event <- clinical_data$RFi_event[match(rownames(fgr_data), clinical_data$Sample)]
fgr_data$RFi_years <- clinical_data$RFi_years[match(rownames(fgr_data), clinical_data$Sample)]

mrs_cutoff_res  <- compute_risk_cutoffs(fgr_MRS,  fgr_MRS_data,  fgr_data, EVAL_TIMES)
clin_cutoff_res <- compute_risk_cutoffs(fgr_CLIN, fgr_CLIN_data, fgr_data, EVAL_TIMES)

# Log diagnostics
for (res in list(MRS = mrs_cutoff_res, Clinical = clin_cutoff_res)) {
  cat(sprintf("\n%s model cutoffs:\n", names(res)))  # won't work directly; see note
}
print(mrs_cutoff_res$diagnostics)
print(clin_cutoff_res$diagnostics)

# Optional plots
MAKE_PLOTS=FALSE
if (MAKE_PLOTS) {
  pdf(file.path(current_output_dir, "cutoff_selection_plots.pdf"),
      width = 10, height = 4 * length(EVAL_TIMES))
  plot_cutoff_diagnostics(fgr_MRS,  fgr_data, EVAL_TIMES, mrs_cutoff_res$cutoffs)
  dev.off()
}

selected_cutoffs_mrs  <- mrs_cutoff_res$cutoffs
selected_cutoffs_clin <- clin_cutoff_res$cutoffs

# ###############################################################################
# # COMPARE MODELS pam50, used in thesis book
# ###############################################################################
# 
# 
# risk_time_cutoff = 10 #delete
# for(risk_time_cutoff in cutoff_time_points) {
# 
#   # pam50 exploration
#   #pdf(file.path("~/Desktop/", "pam50_boxplot.pdf"), # width = 6, height = 6)
#   #dev.off()
# 
#   # compare models
#   risk_comb <- as.numeric(predictRisk(fgr_COMBINED, 
#                                       newdata = fgr_COMBINED_data, 
#                                       times = risk_time_cutoff, 
#                                       cause = 1)[,1])
#   risk_clin <- as.numeric(predictRisk(fgr_CLIN,     
#                                       newdata = fgr_CLIN_data,     
#                                       times = risk_time_cutoff, 
#                                       cause = 1)[,1])
#   risk_mrs  <- as.numeric(predictRisk(fgr_MRS,      
#                                       newdata = fgr_MRS_data,      
#                                       times = risk_time_cutoff, 
#                                       cause = 1)[,1])
#   
#   summary(risk_comb - risk_clin)
#   summary(risk_comb - risk_mrs)
#   summary(risk_clin - risk_mrs)
#   
#   max(abs(risk_comb - risk_clin), na.rm = TRUE)
#   max(abs(risk_comb - risk_mrs), na.rm = TRUE)
#   max(abs(risk_clin - risk_mrs), na.rm = TRUE)
#   
#   cor(risk_comb, risk_clin, use = "complete.obs") # 0.27 # low
#   cor(risk_comb, risk_mrs,  use = "complete.obs") # 0.97 very high
#   cor(risk_clin, risk_mrs,  use = "complete.obs") # 0.17 # low
#   
#   library(corrplot)
#   
#   # build matrix once
#   risk_mat <- cbind(
#     Clinical    = risk_clin,
#     Methylation = risk_mrs,
#     Combined    = risk_comb
#   )
#   
#   # correlations
#   cor_pearson  <- cor(risk_mat, use = "complete.obs", method = "pearson")
#   cor_spearman <- cor(risk_mat, use = "complete.obs", method = "spearman")
#   
#   # plot side by side
#   par(mfrow = c(1,1))
#   pdf(file.path("~/Desktop/", "corr_erpher2n.pdf"), width = 6, height = 6, onefile = TRUE)
#   corrplot(cor_pearson,
#            method = "color",
#            addCoef.col = "black",
#            tl.col = "black",
#            col = colorRampPalette(c("#2166ac", "white", "#b2182b"))(200),
#            cl.lim = c(-1, 1),
#            title = "Pearson",
#            mar = c(0,0,2,0))
#   dev.off()
#   
#   corrplot(cor_spearman,
#            method = "color",
#            addCoef.col = "black",
#            tl.col = "black",
#            col = colorRampPalette(c("#2166ac", "white", "#b2182b"))(200),
#            cl.lim = c(-1, 1),
#            title = "Spearman",
#            mar = c(0,0,2,0))
#   
#   # pam50 comp
#   risk_comb_named <- setNames(risk_comb, rownames(fgr_COMBINED_data))
#   risk_clin_named <- setNames(risk_clin, rownames(fgr_CLIN_data))
#   risk_mrs_named  <- setNames(risk_mrs,  rownames(fgr_MRS_data))
#   
#   par(mfrow = c(1, 3))
#   plot_pam50_risk_boxplot(risk_comb_named, clinical_data, risk_time_cutoff, main = "Combined")
#   plot_pam50_risk_boxplot(risk_clin_named, clinical_data, risk_time_cutoff, main = "Clinical")
#   plot_pam50_risk_boxplot(risk_mrs_named,  clinical_data, risk_time_cutoff, main = "Methylation")
#   
#   # to save
#   par(mfrow = c(1, 1))
#   pdf(file.path("~/Desktop/", "pam50_boxplot_clin.pdf"), width = 6, height = 6, onefile = TRUE)
#   plot_pam50_risk_boxplot(risk_comb_named, clinical_data, risk_time_cutoff, main = "Combined")
#   plot_pam50_risk_boxplot(risk_clin_named, clinical_data, risk_time_cutoff, main = "Clinical")
#   plot_pam50_risk_boxplot(risk_mrs_named,  clinical_data, risk_time_cutoff, main = "Methylation")
#   dev.off()
# }

###############################################################################
# SAVE DEPLOYMENT OBJECT (for applying to independent data)
###############################################################################

# Everything here is the minimum needed to go from raw test data -> predictions.
# Step 1 (MRS path):  cpg_scaling -> cpg_coefs -> mrs_scaling -> fgr_MRS -> cutoff
# Step 2 (Clin path): clinical_scaling + encoded_cols -> fgr_CLIN -> cutoff

deployment_object <- list(
  
  mrs_params = list(
    required_cpgs = pFG_selected_cpgs,   # which CpGs to extract from test methylation
    cpg_coefs     = pFG_cpg_coefs,       # weights for linear combination -> raw MRS
    cpg_scaling   = pFG_cpg_scaling,     # center/scale applied to CpGs before scoring
    mrs_scaling   = mrs_scaling          # center/scale to standardise final MRS
  ),
  
  clinical_params = list(
    continuous_vars  = CLIN_CONT,
    categorical_vars = CLIN_CATEGORICAL,
    encoded_cols     = encoded_result$encoded_cols,  # expected dummy columns after encoding
    clinical_scaling = clin_scaling                  # center/scale for continuous vars
  ),
  
  mrs_model = list(
    model      = fgr_MRS,
    rs_cutoffs = selected_cutoffs_mrs
  ),
  
  clin_model = list(
    model      = fgr_CLIN,
    rs_cutoffs = selected_cutoffs_clin
  )
)
save(deployment_object, file = file.path(current_output_dir, "deployment_object.RData"))

cat("\nSaved:\n")
cat("  - deployment_object.RData (minimal object for test set predictions)\n")

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

