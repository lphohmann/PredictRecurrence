#!/usr/bin/env Rscript
################################################################################
# Fine-Gray Model Application to Test Set
# Author: Lennart Hohmann
################################################################################
# LIBRARIES
################################################################################
setwd("~/PhD_Workspace/PredictRecurrence/")

library(readr)
library(dplyr)
library(survival)
library(cmprsk)
library(riskRegression)
library(prodlim)
library(fastcmprsk)

source("./src/finegray_functions.R")

################################################################################
# INPUT / OUTPUT
################################################################################
SCRIPT_NAME <- "finegray_TestSet_Application"
COHORT_NAME <- "ERpHER2n"

INFILE_BETA_TEST     <- "./data/train/TestSetSamples_Singlets_betaMatrix_V1_V2_reduced_717459commonCpGs_n2601.RData"
INFILE_CLINICAL_TEST <- "./data/train/test_clinical.csv"
INFILE_DEPLOY        <- "./output/FineGray/ERpHER2n/deployment_object.RData"

EVAL_TIMES <- 1:10

current_output_dir <- "./output/FineGray/ERpHER2n/TestSet/"
dir.create(current_output_dir, recursive = TRUE, showWarnings = FALSE)

################################################################################
# RUN INFO
################################################################################
start_time <- Sys.time()
cat(paste(rep("=", 80), collapse = ""), "\n")
cat(sprintf("%s\n", toupper(SCRIPT_NAME)))
cat(sprintf("Started:  %s\n", start_time))
cat(sprintf("Cohort:   %s\n", COHORT_NAME))
cat(paste(rep("=", 80), collapse = ""), "\n\n")

################################################################################
# LOAD DEPLOYMENT OBJECT
################################################################################
cat("========== LOADING DEPLOYMENT OBJECT ==========\n")
load(INFILE_DEPLOY)
# deployment_object contains:
#   mrs_params:      required_cpgs, cpg_coefs, cpg_scaling, mrs_scaling
#   clinical_params: continuous_vars, categorical_vars, encoded_cols, clinical_scaling
#   mrs_model:       model, rs_cutoffs
#   clin_model:      model, rs_cutoffs

CLIN_CONT        <- deployment_object$clinical_params$continuous_vars
CLIN_CATEGORICAL <- deployment_object$clinical_params$categorical_vars
rs_cpgs          <- deployment_object$mrs_params$required_cpgs

cat(sprintf("Required CpGs:        %d\n", length(rs_cpgs)))
cat(sprintf("Continuous clinical:  %s\n", paste(CLIN_CONT, collapse = ", ")))
cat(sprintf("Categorical clinical: %s\n", paste(CLIN_CATEGORICAL, collapse = ", ")))

################################################################################
# LOAD AND FILTER TEST DATA
################################################################################
cat("\n========== LOADING TEST DATA ==========\n")

# Clinical
clinical_test <- read.csv(INFILE_CLINICAL_TEST, stringsAsFactors = FALSE)
clinical_test <- clinical_test[clinical_test$Group == "ER+HER2-", ]
cat(sprintf("Clinical test samples (ER+HER2-): %d\n", nrow(clinical_test)))

# Methylation
load(INFILE_BETA_TEST)  # loads beta.matrix.singlet (CpGs x samples)
colnames(beta.matrix.singlet) <- sub("\\..*$", "", colnames(beta.matrix.singlet))
beta.matrix.singlet <- t(beta.matrix.singlet)  # now samples x CpGs

# Align samples: keep only those present in both clinical and methylation
common_samples <- intersect(clinical_test$Sample, rownames(beta.matrix.singlet))
clinical_test  <- clinical_test[clinical_test$Sample %in% common_samples, ]
beta_test       <- beta.matrix.singlet[clinical_test$Sample, rs_cpgs, drop = FALSE]

cat(sprintf("Samples after alignment:          %d\n", nrow(beta_test)))
cat(sprintf("CpGs extracted:                   %d\n", ncol(beta_test)))

################################################################################
# DEFINE COMPETING RISKS OUTCOME
################################################################################
cat("\n========== DEFINING OUTCOMES ==========\n")

clinical_test$DeathNoR_event <- as.integer(
  clinical_test$OS_event == 1 & clinical_test$RFi_event == 0
)
clinical_test$DeathNoR_years <- clinical_test$OS_years

clinical_test$CompRisk_event_coded <- 0L
clinical_test$CompRisk_event_coded[clinical_test$RFi_event == 1]        <- 1L
clinical_test$CompRisk_event_coded[clinical_test$DeathNoR_event == 1]   <- 2L

clinical_test$time_to_CompRisk_event <- pmin(
  clinical_test$RFi_years,
  clinical_test$DeathNoR_years
)

cat(sprintf("Censored:                  %d (%.1f%%)\n",
            sum(clinical_test$CompRisk_event_coded == 0),
            100 * mean(clinical_test$CompRisk_event_coded == 0)))
cat(sprintf("RFI events:                %d (%.1f%%)\n",
            sum(clinical_test$CompRisk_event_coded == 1),
            100 * mean(clinical_test$CompRisk_event_coded == 1)))
cat(sprintf("Death without recurrence:  %d (%.1f%%)\n",
            sum(clinical_test$CompRisk_event_coded == 2),
            100 * mean(clinical_test$CompRisk_event_coded == 2)))

################################################################################
# TEST SET EVENT BREAKDOWN BY YEAR
################################################################################
cat("\n========== TEST SET EVENT BREAKDOWN BY YEAR ==========\n")

# Safety check: independent test evaluation should only include ER+HER2- samples
stopifnot(all(clinical_test$Group == "ER+HER2-"))

event_breakdown <- do.call(rbind, lapply(EVAL_TIMES, function(t) {
  data.frame(
    year = t,
    rfi_events_up_to_year = sum(
      clinical_test$CompRisk_event_coded == 1 &
        clinical_test$time_to_CompRisk_event <= t,
      na.rm = TRUE
    ),
    death_without_recurrence_up_to_year = sum(
      clinical_test$CompRisk_event_coded == 2 &
        clinical_test$time_to_CompRisk_event <= t,
      na.rm = TRUE
    ),
    censored_up_to_year = sum(
      clinical_test$CompRisk_event_coded == 0 &
        clinical_test$time_to_CompRisk_event <= t,
      na.rm = TRUE
    ),
    still_under_observation_after_year = sum(
      clinical_test$time_to_CompRisk_event > t,
      na.rm = TRUE
    )
  )
}))

cat("\nCumulative test-set event breakdown:\n")
print(event_breakdown, row.names = FALSE)

################################################################################
# PREPARE METHYLATION FEATURES (MRS)
################################################################################
cat("\n========== PREPARING METHYLATION FEATURES ==========\n")

# Beta -> M-values
mvals_test <- beta_to_m(beta_test, beta_threshold = 0.001)

# Calculate MRS using deployment object parameters
mrs_res_test <- calculate_methylation_risk_score(
  X_data         = mvals_test,
  cpg_coefficients = deployment_object$mrs_params$cpg_coefs,
  scaling_params   = deployment_object$mrs_params$cpg_scaling,
  verbose          = TRUE
)
MRS_test <- mrs_res_test$mrs

# Apply the same standardisation as training (fixed center/scale from training)
MRS_test <- (MRS_test - deployment_object$mrs_params$mrs_scaling$center) /
  deployment_object$mrs_params$mrs_scaling$scale
cat(sprintf("MRS: mean=%.3f, sd=%.3f (training was mean=0, sd=1)\n",
            mean(MRS_test), sd(MRS_test)))

################################################################################
# PREPARE CLINICAL FEATURES
################################################################################
cat("\n========== PREPARING CLINICAL FEATURES ==========\n")

clinical_test$LN <- gsub("N\\+", "Np", clinical_test$LN)
rownames(clinical_test) <- clinical_test$Sample

clin_test <- clinical_test[, c(CLIN_CONT, CLIN_CATEGORICAL), drop = FALSE]

# Apply training scaling to continuous vars (fixed center/scale — do NOT refit)
clin_scaling     <- deployment_object$clinical_params$clinical_scaling
clin_test_scaled <- clin_test
clin_test_scaled[, CLIN_CONT] <- scale(
  clin_test[, CLIN_CONT, drop = FALSE],
  center = clin_scaling$center,
  scale  = clin_scaling$scale
)

# One-hot encode using training schema
encoded_test   <- onehot_encode_clinical(clin_test_scaled, CLIN_CATEGORICAL)
expected_cols  <- deployment_object$clinical_params$encoded_cols

# Ensure column alignment with training (add missing dummies as 0 if needed)
missing_cols <- setdiff(expected_cols, colnames(encoded_test$encoded_df))
if (length(missing_cols) > 0) {
  cat(sprintf("WARNING: %d dummy columns missing in test data, filling with 0: %s\n",
              length(missing_cols), paste(missing_cols, collapse = ", ")))
  encoded_test$encoded_df[, missing_cols] <- 0L
}
encoded_test_df <- encoded_test$encoded_df[, expected_cols, drop = FALSE]

################################################################################
# BUILD MODEL INPUT DATA FRAMES
################################################################################
cat("\n========== BUILDING MODEL INPUT ==========\n")

# Verify row alignment
stopifnot(identical(rownames(clin_test_scaled), rownames(mvals_test)))

fgr_data_test <- data.frame(
  time_to_CompRisk_event = clinical_test[rownames(mvals_test), "time_to_CompRisk_event"],
  CompRisk_event_coded   = clinical_test[rownames(mvals_test), "CompRisk_event_coded"],
  RFi_event              = clinical_test[rownames(mvals_test), "RFi_event"],
  RFi_years              = clinical_test[rownames(mvals_test), "RFi_years"],
  clin_test_scaled[, CLIN_CONT, drop = FALSE],
  encoded_test_df,
  MeRS = MRS_test,
  row.names = rownames(mvals_test)
)

fgr_MRS_data_test  <- fgr_data_test[, c("time_to_CompRisk_event", "CompRisk_event_coded", "MeRS")]
fgr_CLIN_data_test <- fgr_data_test[, c("time_to_CompRisk_event", "CompRisk_event_coded",
                                        CLIN_CONT, expected_cols)]  # fixed: added CLIN_CONT

################################################################################
# APPLY MODELS AND COMPUTE PREDICTIONS
################################################################################
cat("\n========== APPLYING MODELS TO TEST SET ==========\n")

mrs_cutoffs  <- deployment_object$mrs_model$rs_cutoffs
clin_cutoffs <- deployment_object$clin_model$rs_cutoffs

for (t in EVAL_TIMES) {
  key <- paste0(t, "_year")
  
  risk_mrs  <- as.numeric(predictRisk(deployment_object$mrs_model$model,
                                      newdata = fgr_MRS_data_test,
                                      times = t, cause = 1)[, 1])
  risk_clin <- as.numeric(predictRisk(deployment_object$clin_model$model,
                                      newdata = fgr_CLIN_data_test,
                                      times = t, cause = 1)[, 1])
  
  fgr_data_test[[paste0("risk_mrs_",  t, "y")]]  <- risk_mrs
  fgr_data_test[[paste0("risk_clin_", t, "y")]]  <- risk_clin
  fgr_data_test[[paste0("group_mrs_",  t, "y")]] <- ifelse(risk_mrs  >= mrs_cutoffs[[key]],  "High risk", "Low risk")
  fgr_data_test[[paste0("group_clin_", t, "y")]] <- ifelse(risk_clin >= clin_cutoffs[[key]], "High risk", "Low risk")
  
  cat(sprintf("Year %d | MRS  high/low: %d/%d | Clin high/low: %d/%d\n", t,
              sum(fgr_data_test[[paste0("group_mrs_",  t, "y")]] == "High risk"),
              sum(fgr_data_test[[paste0("group_mrs_",  t, "y")]] == "Low risk"),
              sum(fgr_data_test[[paste0("group_clin_", t, "y")]] == "High risk"),
              sum(fgr_data_test[[paste0("group_clin_", t, "y")]] == "Low risk")))
}

################################################################################
# SAVE PREDICTIONS
################################################################################
save(fgr_data_test, clinical_test,
     file = file.path(current_output_dir, "testset_predictions.RData"))
cat("  - testset_predictions.RData\n")

################################################################################
# FINISH
################################################################################
total_runtime <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
cat(paste(rep("=", 80), collapse = ""), "\n")
cat(sprintf("PIPELINE COMPLETED\n"))
cat(sprintf("Finished:      %s\n", Sys.time()))
cat(sprintf("Total runtime: %.1f minutes\n", total_runtime))
cat(paste(rep("=", 80), collapse = ""), "\n")
