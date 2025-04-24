#!/usr/bin/env Rscript
# Script: Predictiveness curve for risk score
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
pacman::p_load(data.table, globaltest)

#-------------------
# set/create output directories
output_path <- "./output/SCANB_GlobalTest/"
dir.create(output_path, showWarnings = FALSE)
#-------------------
# input paths
infile_1 <- "./data/raw/tnbc_anno.csv"
infile_2 <- "./data/raw/tnbc235.csv"
#-------------------
# output paths
outfile_1 <- paste0(output_path, "SCANB_TNBC_GlobalTestMethyl.txt")
#-------------------
# storing objects 
results <- list() # object to store results

#######################################################################
# load data
#######################################################################

clinical <- read.csv(infile_1)
dat <- data.table::fread(infile_2)
dat <- as.data.frame(dat)

# Move gene names to rownames
rownames(dat) <- dat$V1
dat <- dat[, -1]
#dat[1:5,1:5]

# Ensure clinical data has the same samples as dat
clinical <- clinical[clinical$PD_ID %in% colnames(dat), ]

# Ensure the samples in clinical data are in the same order as in dat
clinical <- clinical[match(colnames(dat), clinical$PD_ID), ]

head(clinical)
#identical(clinical$PD_ID,colnames(dat))

#######################################################################
# Run Global Test
#######################################################################

# Define outcomes
outcomes <- c("RFIbin", "OSbin")
#outcome <- "RFIbin"
for (outcome in outcomes) {

    # select only samples with available outcome measure data
    sub_clinical <- clinical[!is.na(clinical[[outcome]]), ]
    sub_dat <- dat[, colnames(dat) %in% sub_clinical$PD_ID]
    #identical(colnames(sub_dat), sub_clinical$PD_ID) # TRUE

    # Ensure the outcome is numeric
    sub_clinical[[outcome]] <- as.numeric(as.character(sub_clinical[[outcome]]))
    
    # Run the global test
    gt_result <- gt(sub_clinical[[outcome]], t(sub_dat)) # Transpose dat to match the expected format
    
    
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

