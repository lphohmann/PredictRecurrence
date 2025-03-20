#!/usr/bin/env Rscript
# Script: Project cohort - Global test on gene expression data
# Author: Lennart Hohmann
# Date: 20.03.2025
#-------------------
# empty environment
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence/")
#-------------------
# packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(data.table, globaltest)

#-------------------
# set/create output directories
output_path <- "./output/SCANB_GlobalTest/"
dir.create(output_path, showWarnings = FALSE)
#-------------------
# input paths
infile_1 <- "./data/standardized/SCANB_ProjectCohort/SCANB_PC_clinical.csv"
infile_2 <- "./data/standardized/SCANB_ProjectCohort/SCANB_PC_RNAseq_expression.csv"
#-------------------
# output paths
outfile_1 <- paste0(output_path, "GlobalTest_Results.txt")
#-------------------
# storing objects 
results <- list() # object to store results

#######################################################################
# load data
#######################################################################

clinical <- read.csv(infile_1)
gex <- data.table::fread(infile_2)
gex <- as.data.frame(gex)

# Ensure the first column is the gene ID
rownames(gex) <- gex$Gene
gex <- gex[, -1]

# Ensure clinical data has the same samples as gex
clinical <- clinical[clinical$Sample %in% colnames(gex), ]

#######################################################################
# Run Global Test
#######################################################################

# Define outcomes
outcomes <- c("RFi_event", "OS_event")

for (outcome in outcomes) {
  # Ensure the outcome is numeric
  clinical[[outcome]] <- as.numeric(as.character(clinical[[outcome]]))
  
  # Run the global test
  gt_result <- gt(clinical[[outcome]], gex)
  
  # Store the result
  results[[outcome]] <- summary(gt_result)
}

#######################################################################
# Save results to file
#######################################################################

sink(outfile_1)
for (outcome in outcomes) {
  cat("Global Test Results for", outcome, "\n")
  print(results[[outcome]])
  cat("\n\n")
}
sink()

