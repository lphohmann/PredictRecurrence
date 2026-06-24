#!/usr/bin/env Rscript
################################################################################
# Fine-Gray Model Test Set Analysis
# Author: Lennart Hohmann
################################################################################
setwd("~/PhD_Workspace/PredictRecurrence/")

library(cmprsk)
library(riskRegression)
library(prodlim)
library(dplyr)

source("./src/finegray_functions.R")

################################################################################
# INPUT / OUTPUT
################################################################################
SCRIPT_NAME  <- "finegray_TestSet_Analysis"
COHORT_NAME  <- "ERpHER2n"
EVAL_TIMES   <- 1:10

INFILE_PREDICTIONS <- "./output/FineGray/ERpHER2n/TestSet/testset_predictions.RData"
INFILE_DEPLOY      <- "./output/FineGray/ERpHER2n/deployment_object.RData"

current_output_dir <- "./output/FineGray/ERpHER2n/TestSet/"

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
# LOAD
################################################################################
load(INFILE_PREDICTIONS)  # fgr_data_test, clinical_test
load(INFILE_DEPLOY)       # deployment_object — needed for Score() model objects

# then all your downstream sections follow as-is:
# - cross tabulation
# - Score()
# - all plots


################################################################################
# ADD RISK GROUPS TO CLINICAL TEST DATA
################################################################################
cat("\n========== ADDING RISK GROUPS TO CLINICAL DATA ==========\n")

for (t in c(5, 10)) {
  clinical_test[, paste0("group_mrs_",  t, "y")] <- fgr_data_test[clinical_test$Sample, paste0("group_mrs_",  t, "y")]
  clinical_test[, paste0("group_clin_", t, "y")] <- fgr_data_test[clinical_test$Sample, paste0("group_clin_", t, "y")]
}

cat("Added columns:\n")
cat(paste(" ", grep("group_", colnames(clinical_test), value = TRUE), collapse = "\n"), "\n")
save(clinical_test, file = file.path(current_output_dir, "clinical_test_with_riskgroups.RData"))

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
  se.fit   = FALSE,
  null.model = TRUE,
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
# PLOT: KM BY COMBINED RISK GROUP OVERLAP
################################################################################

cat("\n========== PLOTTING KM BY COMBINED RISK GROUPS ==========\n")

library(survminer)

pdf(file.path(current_output_dir, "km_combined_riskgroups.pdf"), width = 8, height = 7, onefile = TRUE)

for (t in c(5, 10)) {
  
  clin_grp <- fgr_data_test[[paste0("group_clin_", t, "y")]]
  mrs_grp  <- fgr_data_test[[paste0("group_mrs_",  t, "y")]]
  
  # Clinical first, MRS second
  combined_group <- dplyr::case_when(
    clin_grp == "High risk" & mrs_grp == "High risk" ~ "High-High",
    clin_grp == "High risk" & mrs_grp == "Low risk"  ~ "High-Low",
    clin_grp == "Low risk"  & mrs_grp == "High risk" ~ "Low-High",
    clin_grp == "Low risk"  & mrs_grp == "Low risk"  ~ "Low-Low"
  )
  combined_group <- factor(combined_group,
                           levels = c("High-High", "High-Low", "Low-High", "Low-Low"))
  counts <- table(combined_group)
  
  km_df <- data.frame(
    os_time  = clinical_test[rownames(fgr_data_test), "OS_years"],
    os_event = clinical_test[rownames(fgr_data_test), "OS_event"],
    group    = combined_group
  )
  
  km_fit <- survfit(Surv(os_time, os_event) ~ group, data = km_df)
  
  p <- ggsurvplot(
    km_fit,
    data              = km_df,
    palette           = c("#b2182b", "#f4a582", "#92c5de", "#2166ac"),
    legend.labs       = sprintf("%s (n=%d)", names(counts), counts),
    legend.title      = "Clinical — MRS",
    title             = sprintf("KM — Clinical/MRS risk groups at year %d", t),
    xlab              = "Time (years)",
    ylab              = "Overall survival",
    conf.int          = FALSE,
    risk.table        = TRUE,
    risk.table.height = 0.25,
    ggtheme           = theme_classic()
  )
  
  print(p)
}

dev.off()
cat("  Saved: km_combined_riskgroups.pdf\n")

################################################################################
# COMPARE MODELS TO PROSIGNA ROR
################################################################################
cat("\n========== PROSIGNA ROR COMPARISON ==========\n")

ror.dat <- loadRData("./data/raw/SSP_ROR_data.RData")
ror.dat <- ror.dat[ror.dat$Follow.up.cohort == TRUE &
                     ror.dat$Sample %in% clinical_test$Sample,
                   c("Sample", "NCN.ROR.risk.cat",
                     "NCN.ROR.binary.risk.cat", "Size.mm",
                     "NCN.ROR.asT0", "NCN.ROR.asT1")]
ror.dat$NCN.ROR.comb <- ifelse(ror.dat$Size.mm <= 20,
                               ror.dat$NCN.ROR.asT0,
                               ror.dat$NCN.ROR.asT1)
ror.dat$ROR_group <- ifelse(ror.dat$NCN.ROR.binary.risk.cat == "High",
                            "High risk", "Low risk")
cat(sprintf("ROR samples available: %d\n", nrow(ror.dat)))

cat(sprintf("ROR high/low: %d/%d (NA: %d)\n",
            sum(ror.dat$ROR_group == "High risk", na.rm = TRUE),
            sum(ror.dat$ROR_group == "Low risk",  na.rm = TRUE),
            sum(is.na(ror.dat$ROR_group))))

pdf(file.path(current_output_dir, "confusion_matrices_ror.pdf"), width = 10, height = 5, onefile = TRUE)

for (t in c(5, 10)) {
  par(mfrow = c(1, 2))
  
  for (model_name in c("clin", "mrs")) {
    
    group_col  <- paste0("group_", model_name, "_", t, "y")
    
    # Align on shared samples
    shared <- intersect(ror.dat$Sample[!is.na(ror.dat$ROR_group)], rownames(fgr_data_test))
    cat(sprintf("\nYear %d | %s | shared samples with ROR: %d\n", t, model_name, length(shared)))
    
    model_groups <- factor(fgr_data_test[shared, group_col],
                           levels = c("High risk", "Low risk"))
    ror_groups   <- factor(ror.dat$ROR_group[match(shared, ror.dat$Sample)],
                           levels = c("High risk", "Low risk"))
    
    ctab     <- table(Model = model_groups, ROR = ror_groups)
    n_total  <- sum(ctab)
    n_agree  <- ctab["High risk", "High risk"] + ctab["Low risk", "Low risk"]
    
    p_chance <- sum(rowSums(ctab)/n_total * colSums(ctab)/n_total)
    kappa    <- (n_agree/n_total - p_chance) / (1 - p_chance)
    pct_tab  <- prop.table(ctab) * 100
    
    cell_cols <- matrix(nrow = 2, ncol = 2)
    cell_cols[1,1] <- adjustcolor("steelblue", alpha.f = pct_tab[1,1]/100 + 0.15)
    cell_cols[2,2] <- adjustcolor("steelblue", alpha.f = pct_tab[2,2]/100 + 0.15)
    cell_cols[1,2] <- adjustcolor("tomato",    alpha.f = pct_tab[1,2]/100 + 0.15)
    cell_cols[2,1] <- adjustcolor("tomato",    alpha.f = pct_tab[2,1]/100 + 0.15)
    
    plot(NULL, xlim = c(0,1), ylim = c(0,1), xaxt = "n", yaxt = "n",
         xlab = "", ylab = "", bty = "n",
         main = sprintf("%s vs ROR | Year %d\nAgreement = %d/%d | Kappa = %.3f",
                        ifelse(model_name == "mrs", "MRS", "Clinical"),
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
    text(0.5, 0.98, "ROR", cex = 1.0, font = 2)
    text(0.08, c(0.675, 0.325), c("High risk", "Low risk"), cex = 0.95, font = 2, srt = 90)
    text(0.02, 0.5, ifelse(model_name == "mrs", "MRS", "Clinical"),
         cex = 1.0, font = 2, srt = 90)
  }
}

dev.off()
cat("  Saved: confusion_matrices_ror.pdf\n")

################################################################################
# EVALUATE MODEL PERFORMANCE ON TEST SET
################################################################################
cat("\n========== MODEL PERFORMANCE EVALUATION ==========\n")

score_test <- Score(
  list("FGR_MRS"  = deployment_object$mrs_model$model,
       "FGR_CLIN" = deployment_object$clin_model$model),
  formula    = Hist(time_to_CompRisk_event, CompRisk_event_coded) ~ 1,
  data       = fgr_data_test,
  cause      = 1,
  times      = EVAL_TIMES,
  metrics    = c("auc", "brier"),
  summary    = "ibs",
  null.model = FALSE,
  cens.model = "cox"
)

# Build performance table in exact same structure as fgr_performances in nCV
# so results are directly comparable
testset_performances <- list()

for (fgmodel in c("FGR_CLIN", "FGR_MRS")) {
  
  auc_model   <- score_test$AUC$score[model == fgmodel]
  brier_model <- score_test$Brier$score[model == fgmodel]
  eval_times  <- auc_model$times
  
  auc_cols   <- setNames(auc_model$AUC,     paste0("auc_",   eval_times, "yr"))
  brier_cols <- setNames(brier_model$Brier, paste0("brier_", eval_times, "yr"))
  
  fgr_perf <- data.frame(
    model      = fgmodel,
    mean_auc   = mean(auc_model$AUC, na.rm = TRUE),
    final_ibs  = brier_model[times == max(times), IBS],
    as.list(auc_cols),
    as.list(brier_cols)
  )
  fgr_perf[, -1] <- round(fgr_perf[, -1], 3)
  testset_performances <- append(testset_performances, list(fgr_perf))
}

testset_performances <- do.call(rbind, testset_performances)

cat("\nTest set performance (mirrors nCV fgr_performances structure):\n")
print(testset_performances)

# Print in same style as aggregate_cv_performance_threefg console output
# so you can visually compare side by side
cat("\n--- Per-timepoint AUC ---\n")
auc_cols <- grep("^auc_[0-9]", colnames(testset_performances), value = TRUE)
for (fgmodel in c("FGR_CLIN", "FGR_MRS")) {
  row <- testset_performances[testset_performances$model == fgmodel, ]
  cat(sprintf("\n%s\n", fgmodel))
  cat(sprintf("  mean_auc:  %.3f\n", row$mean_auc))
  cat(sprintf("  final_ibs: %.3f\n", row$final_ibs))
  cat("  AUC per timepoint:\n")
  for (col in auc_cols) {
    t <- gsub("auc_|yr", "", col)
    cat(sprintf("    Year %2s: %.3f\n", t, row[[col]]))
  }
}

################################################################################
# COMPARE TEST SET vs CV PERFORMANCE
################################################################################
cat("\n========== TEST SET vs CV COMPARISON ==========\n")

# Load CV results for comparison
load(file.path("./output/FineGray", COHORT_NAME, "outer_fold_results.RData"))

for (fgmodel in c("FGR_CLIN", "FGR_MRS")) {
  
  cv_perf <- aggregate_cv_performance_threefg(
    nCV_results$outer_fold_results,
    model_name = fgmodel,
    verbose    = FALSE
  )
  
  test_row <- testset_performances[testset_performances$model == fgmodel, ]
  cv_mean_auc <- cv_perf$summary[cv_perf$summary$metric == "mean_auc", ]
  cv_ibs      <- cv_perf$summary[cv_perf$summary$metric == "final_ibs", ]
  
  cat(sprintf("\n%s\n", fgmodel))
  cat(sprintf("  %-12s  CV: %.3f (95%% CI: %.3f-%.3f)  |  Test: %.3f\n",
              "mean_auc",
              cv_mean_auc$mean, cv_mean_auc$ci_lower, cv_mean_auc$ci_upper,
              test_row$mean_auc))
  cat(sprintf("  %-12s  CV: %.3f (95%% CI: %.3f-%.3f)  |  Test: %.3f\n",
              "final_ibs",
              cv_ibs$mean, cv_ibs$ci_lower, cv_ibs$ci_upper,
              test_row$final_ibs))
}



################################################################################
# METAGENES FO BIOL ANNOTATION
################################################################################

### metagenes
mg.scores <- loadRData("./data/raw/Metagene_scores_All.RData")

