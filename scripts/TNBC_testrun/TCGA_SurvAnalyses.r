#!/usr/bin/env Rscript
# Script: Survival analyses in TCGA
# Author: Lennart Hohmann
# Date: 24.04.2025
#-------------------
# empty environment
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence/")
#-------------------
# packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  data.table, prodlim,
  survival, Publish, survival, survminer
)
source("./src/utils.R")
#-------------------
# set/create output directories
output_path <- "output/CoxNet_200k_simpleCV5/"
dir.create(output_path, showWarnings = FALSE)
#-------------------
# input paths
infile_1 = "./output/CoxNet_200k_simpleCV5/TCGA_risk_scores.csv" # replace with tnbc dat
#-------------------
# output paths
#-------------------

#######################################################################
# load data
#######################################################################

comb <- read.csv("output/CoxNet_200k_simpleCV5/TCGA_risk_scores.csv", stringsAsFactors = FALSE)
comb$X <- NULL

head(comb)
str(comb)

hist(comb$risk_score)
comb$risk_score_scaled <- scale(comb$risk_score)
hist(comb$risk_score_scaled)

risk_score_OS <- coxph(Surv(OS,OSbin)~risk_score_scaled,data=comb)
publish(risk_score_OS)

# risk_score_PFI <- coxph(Surv(PFI,PFIbin)~risk_score_scaled,data=comb)
# publish(risk_score_PFI)

pc_scaled <- 1.571565
pc <- 2.768252
med_scaled <- -0.3024001
med <- -0.5326662

cutoff <- med_scaled
rs_type <- "risk_score_scaled" #risk_score_scaled or risk_score
rs_type_binary <- "risk_score_scaled_binary" #risk_score_scaled_binary or risk_score_binary

table(comb[[rs_type]]>cutoff)

# binarisation of risk score
comb[[rs_type_binary]] <- ifelse(comb[[rs_type]] > cutoff, 1, 0)
comb[[rs_type_binary]] <- as.integer(comb[[rs_type_binary]])

table(comb[[rs_type_binary]])




comb$time <- as.numeric(comb$OS)
comb$status <- as.integer(comb$OSbin)

# Save the plot to a variable
fit <- survfit(Surv(time, status) ~ risk_score_scaled_binary, data = comb)

# Save full ggsurvplot object
km_plot <- ggsurvplot(fit,
    data = comb,
    risk.table = TRUE,
    risk.table.y.text.col = TRUE,
    risk.table.y.text = FALSE,
    risk.table.position = "right",
    font.main = c(18, "bold"),
    font.x = c(16, "plain"),
    font.y = c(16, "plain"),
    font.tickslab = c(14, "plain"),
    risk.table.fontsize = 5,
    legend.title = "Risk Group",
    legend.labs = c("Low risk", "High risk"),
    font.legend = c(14, "plain"),
    title = "TCGA: DNA-methylation risk stratification in TNBC",
    palette = c("steelblue", "tomato"),
    xlab = "OS (days)", # custom x-axis label
    ylab = "OS probability",
    pval = TRUE
) # custom y-axis label)

# Save to file — use arrange_ggsurvplots to combine main + table
ggsave("output/CoxNet_200k_simpleCV5/TNBC_KMcurves_Median.png",
    plot = arrange_ggsurvplots(list(km_plot)),
    width = 12, height = 6, dpi = 300
)
ggsave("output/CoxNet_200k_simpleCV5/TNBC_KMcurves_Median.pdf",
       plot = arrange_ggsurvplots(list(km_plot)),
       width = 12, height = 6)
