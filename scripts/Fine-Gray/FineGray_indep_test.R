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
# CROSS-TABULATION: MRS vs CLINICAL RISK GROUP AGREEMENT
################################################################################
cat("\n========== RISK GROUP CONCORDANCE ==========\n")
pdf(file.path(current_output_dir, "confusion_matrices.pdf"), width = 10, height = 5)
par(mfrow = c(1, 2))

for (t in c(5, 10)) {
  
  mrs_groups  <- factor(fgr_data_test[[paste0("group_mrs_",  t, "y")]], levels = c("High risk", "Low risk"))
  clin_groups <- factor(fgr_data_test[[paste0("group_clin_", t, "y")]], levels = c("High risk", "Low risk"))
  ctab        <- table(MRS = mrs_groups, Clinical = clin_groups)
  n_total     <- sum(ctab)
  n_agree     <- ctab["High risk", "High risk"] + ctab["Low risk", "Low risk"]
  p_chance    <- sum(rowSums(ctab)/n_total * colSums(ctab)/n_total)
  kappa       <- (n_agree/n_total - p_chance) / (1 - p_chance)
  pct_tab     <- prop.table(ctab) * 100
  
  cell_cols <- matrix(nrow = 2, ncol = 2)
  cell_cols[1,1] <- adjustcolor("steelblue", alpha.f = pct_tab[1,1]/100 + 0.15)
  cell_cols[2,2] <- adjustcolor("steelblue", alpha.f = pct_tab[2,2]/100 + 0.15)
  cell_cols[1,2] <- adjustcolor("tomato",    alpha.f = pct_tab[1,2]/100 + 0.15)
  cell_cols[2,1] <- adjustcolor("tomato",    alpha.f = pct_tab[2,1]/100 + 0.15)
  
  plot(NULL, xlim = c(0,1), ylim = c(0,1), xaxt = "n", yaxt = "n",
       xlab = "", ylab = "", bty = "n",
       main = sprintf("Risk group agreement | Year %d\nAgreement = %d/%d | Kappa = %.3f",
                      t, n_agree, n_total, kappa))
  
  for (row in 1:2) {
    for (col in 1:2) {
      x_lo <- if (col == 1) 0.15 else 0.50
      x_hi <- if (col == 1) 0.50 else 0.85
      y_lo <- if (row == 1) 0.50 else 0.15
      y_hi <- if (row == 1) 0.85 else 0.50
      rect(x_lo, y_lo, x_hi, y_hi, col = cell_cols[row, col], border = "white", lwd = 2)
      text((x_lo + x_hi)/2, (y_lo + y_hi)/2,
           labels = ctab[row, col], cex = 1.6, font = 2)
    }
  }
  
  text(c(0.325, 0.675), 0.92, c("High risk", "Low risk"), cex = 0.95, font = 2)
  text(0.5, 0.98, "Clinical model", cex = 1.0, font = 2)
  text(0.08, c(0.675, 0.325), c("High risk", "Low risk"), cex = 0.95, font = 2, srt = 90)
  text(0.02, 0.5, "MRS model", cex = 1.0, font = 2, srt = 90)
}

dev.off()

################################################################################
# EVALUATE MODEL PERFORMANCE ON TEST SET
################################################################################
cat("\n========== MODEL PERFORMANCE EVALUATION ==========\n")

# Score() needs the outcome and predictions on the test set.
# We pass the fitted model objects directly — riskRegression calls predictRisk
# internally using the newdata argument.
score_test <- Score(
  list("MRS"      = deployment_object$mrs_model$model,
       "Clinical" = deployment_object$clin_model$model),
  formula  = Hist(time_to_CompRisk_event, CompRisk_event_coded) ~ 1,
  data     = fgr_data_test,
  cause    = 1,
  times    = EVAL_TIMES,
  metrics  = c("auc", "brier"),
  summary  = "ibs",
  se.fit   = TRUE,
  conf.int = 0.95,
  null.model = FALSE,
  cens.model = "cox"
)

# Extract and print summary table
auc_mrs  <- score_test$AUC$score[model == "MRS"]
auc_clin <- score_test$AUC$score[model == "Clinical"]

brier_mrs  <- score_test$Brier$score[model == "MRS"]
brier_clin <- score_test$Brier$score[model == "Clinical"]

perf_table <- data.frame(
  time       = auc_mrs$times,
  auc_mrs    = round(auc_mrs$AUC,    3),
  auc_clin   = round(auc_clin$AUC,   3),
  brier_mrs  = round(brier_mrs$Brier,  3),
  brier_clin = round(brier_clin$Brier, 3)
)
cat("\nPerformance summary across timepoints:\n")
print(perf_table)

# IBS (integrated Brier score — single summary number per model)
ibs_mrs  <- brier_mrs[times  == max(times), IBS]
ibs_clin <- brier_clin[times == max(times), IBS]
cat(sprintf("\nIBS | MRS: %.3f | Clinical: %.3f\n", ibs_mrs, ibs_clin))

################################################################################
# PLOT: AUC(t) ACROSS TIMEPOINTS
################################################################################
cat("\n========== PLOTTING AUC(t) ==========\n")

# Pull 95% CI from Score output
auc_mrs_lo  <- auc_mrs$lower
auc_mrs_hi  <- auc_mrs$upper
auc_clin_lo <- auc_clin$lower
auc_clin_hi <- auc_clin$upper
times       <- auc_mrs$times

pdf(file.path(current_output_dir, "auc_timepoints.pdf"), width = 7, height = 5)

# Set up empty plot with axis ranges covering both models + CIs
plot(times, auc_mrs$AUC,
     type  = "n",
     xlim  = c(min(times), max(times)),
     ylim  = c(0.3, 1.0),
     xlab  = "Time (years)",
     ylab  = "AUC",
     main  = sprintf("Time-dependent AUC — %s test set", COHORT_NAME),
     xaxt  = "n")
axis(1, at = times)
abline(h = 0.5, lty = 3, col = "grey60")   # reference line at 0.5

# CI ribbons
polygon(c(times, rev(times)),
        c(auc_mrs_lo, rev(auc_mrs_hi)),
        col = adjustcolor("steelblue", alpha.f = 0.15), border = NA)
polygon(c(times, rev(times)),
        c(auc_clin_lo, rev(auc_clin_hi)),
        col = adjustcolor("tomato", alpha.f = 0.15), border = NA)

# AUC lines
lines(times, auc_mrs$AUC,  col = "steelblue", lwd = 2, lty = 1)
lines(times, auc_clin$AUC, col = "tomato",    lwd = 2, lty = 2)

# Points at each evaluated timepoint
points(times, auc_mrs$AUC,  col = "steelblue", pch = 16)
points(times, auc_clin$AUC, col = "tomato",    pch = 17)

legend("bottomleft",
       legend = c(sprintf("MRS (mean AUC=%.2f)",  mean(auc_mrs$AUC,  na.rm=TRUE)),
                  sprintf("Clinical (mean AUC=%.2f)", mean(auc_clin$AUC, na.rm=TRUE))),
       col    = c("steelblue", "tomato"),
       lty    = c(1, 2),
       pch    = c(16, 17),
       lwd    = 2,
       bty    = "n")

dev.off()
cat("  Saved: auc_timepoints.pdf\n")

################################################################################
# PLOT: CUMULATIVE INCIDENCE BY RISK GROUP (year 5 and year 10)
################################################################################
cat("\n========== PLOTTING CIF BY RISK GROUP ==========\n")

pdf(file.path(current_output_dir, "cif_risk_groups.pdf"), width = 10, height = 8)
par(mfrow = c(2, 2))  # 2 models x 2 timepoints
for (model_name in c("mrs", "clin")) {
  for (t in c(5, 10)) {
    
    group_col <- paste0("group_", model_name, "_", t, "y")
    
    # Skip if this timepoint wasn't evaluated (e.g. TNBC only goes to 5 years)
    if (!group_col %in% colnames(fgr_data_test)) {
      cat(sprintf("Skipping %s year %d (not in EVAL_TIMES)\n", model_name, t))
      next
    }
    
    counts <- table(factor(fgr_data_test[[group_col]],
                           levels = c("High risk", "Low risk")))
    
    ci <- cuminc(
      ftime   = fgr_data_test$time_to_CompRisk_event,
      fstatus = fgr_data_test$CompRisk_event_coded,
      group   = fgr_data_test[[group_col]]
    )
    
    # Rename curves to include n per group
    names(ci)[1:2] <- c(
      sprintf("High risk (n=%d)", counts["High risk"]),
      sprintf("Low risk  (n=%d)", counts["Low risk"])
    )
    
    plot(ci[1:2],
         col  = c("red", "steelblue"),
         lty  = 1:2,
         lwd  = 2,
         xlab = "Time (years)",
         ylab = "Cumulative incidence",
         main = sprintf("%s model | Risk groups at year %d",
                        ifelse(model_name == "mrs", "MRS", "Clinical"), t))
    abline(v = t, lty = 3, col = "grey50")  # reference line at cutoff year
  }
}

dev.off()
cat("  Saved: cif_risk_groups.pdf\n")

################################################################################
# PLOT: AUC(t) ACROSS TIMEPOINTS
################################################################################
cat("\n========== PLOTTING AUC(t) ==========\n")

times       <- auc_mrs$times
auc_mrs_lo  <- auc_mrs$lower
auc_mrs_hi  <- auc_mrs$upper
auc_clin_lo <- auc_clin$lower
auc_clin_hi <- auc_clin$upper

# Build a summary table mirroring aggregate_cv_performance_threefg$summary
# so this output is directly comparable to the CV results
auc_summary <- data.frame(
  time        = times,
  mrs_auc     = round(auc_mrs$AUC,  3),
  mrs_lower   = round(auc_mrs_lo,   3),
  mrs_upper   = round(auc_mrs_hi,   3),
  clin_auc    = round(auc_clin$AUC, 3),
  clin_lower  = round(auc_clin_lo,  3),
  clin_upper  = round(auc_clin_hi,  3)
)
cat("\nAUC(t) summary:\n")
print(auc_summary)

pdf(file.path(current_output_dir, "auc_timepoints.pdf"), width = 7, height = 5)

y_min <- min(c(auc_mrs_lo, auc_clin_lo), na.rm = TRUE)
y_min <- max(0.3, floor(y_min * 10) / 10)  # round down to nearest 0.1, floor at 0.3

plot(times, auc_mrs$AUC,
     type = "n",
     xlim = c(min(times), max(times)),
     ylim = c(y_min, 1.0),
     xlab = "Time (years)",
     ylab = "AUC",
     main = sprintf("Time-dependent AUC — %s test set", COHORT_NAME),
     xaxt = "n",
     bty  = "l")
axis(1, at = times)
abline(h = 0.5, lty = 3, col = "grey60")

# CI ribbons (same style as CV plot shading)
polygon(c(times, rev(times)),
        c(auc_mrs_lo, rev(auc_mrs_hi)),
        col = adjustcolor("steelblue", alpha.f = 0.15), border = NA)
polygon(c(times, rev(times)),
        c(auc_clin_lo, rev(auc_clin_hi)),
        col = adjustcolor("tomato", alpha.f = 0.15), border = NA)

# Lines
lines(times, auc_mrs$AUC,  col = "steelblue", lwd = 2, lty = 1)
lines(times, auc_clin$AUC, col = "tomato",    lwd = 2, lty = 2)

# Points
points(times, auc_mrs$AUC,  col = "steelblue", pch = 16, cex = 1.2)
points(times, auc_clin$AUC, col = "tomato",    pch = 17, cex = 1.2)

legend("bottomleft",
       legend = c(
         sprintf("MRS      (mean AUC = %.2f)", mean(auc_mrs$AUC,  na.rm = TRUE)),
         sprintf("Clinical (mean AUC = %.2f)", mean(auc_clin$AUC, na.rm = TRUE))
       ),
       col  = c("steelblue", "tomato"),
       lty  = c(1, 2),
       pch  = c(16, 17),
       lwd  = 2,
       bty  = "n",
       cex  = 0.9)

dev.off()
cat("  Saved: auc_timepoints.pdf\n")

################################################################################
# SAVE RESULTS
################################################################################
cat("\n========== SAVING ==========\n")

save(fgr_data_test, file = file.path(current_output_dir, "testset_predictions.RData"))
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