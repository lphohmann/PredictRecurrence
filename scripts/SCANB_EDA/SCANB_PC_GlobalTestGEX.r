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
outfile_1 <- paste0(output_path, "SCANB_PC_GlobalTestGEX.txt")
#-------------------
# storing objects 
results <- list() # object to store results

#######################################################################
# load data
#######################################################################

clinical <- read.csv(infile_1)
gex <- data.table::fread(infile_2)
gex <- as.data.frame(gex)

# Move gene names to rownames
rownames(gex) <- gex$Gene
gex <- gex[, -1]
#gex[1:5,1:5]

# Ensure clinical data has the same samples as gex
clinical <- clinical[clinical$Sample %in% colnames(gex), ]

# Ensure the samples in clinical data are in the same order as in gex
clinical <- clinical[match(colnames(gex), clinical$Sample), ]


#######################################################################
# Run Global Test
#######################################################################

# Define outcomes
outcomes <- c("RFi_event", "OS_event")
#outcome <- "RFi_event"
for (outcome in outcomes) {

    # select only samples with available outcome measure data
    sub_clinical <- clinical[!is.na(clinical[[outcome]]), ]
    sub_gex <- gex[, colnames(gex) %in% sub_clinical$Sample]
    #identical(colnames(sub_gex), sub_clinical$Sample) # TRUE

    # Ensure the outcome is numeric
    sub_clinical[[outcome]] <- as.numeric(as.character(sub_clinical[[outcome]]))
    
    # Run the global test
    gt_result <- gt(sub_clinical[[outcome]], t(sub_gex)) # Transpose gex to match the expected format
    
    
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

