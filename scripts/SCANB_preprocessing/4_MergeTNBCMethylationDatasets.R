#!/usr/bin/env Rscript
# Script: Merging added TNBC methlytion datasets for next steps
# Author: Lennart Hohmann
# Date: 20.05.2025
#----------------------------------------------------------------------
# empty environment
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence/")
#----------------------------------------------------------------------
# packages
#source("./scripts/src/")
if (!require("pacman")) install.packages("pacman")
pacman::p_load(data.table)
source("./src/utils.R")
#----------------------------------------------------------------------
# input paths
infile.0 <- "./data/raw/Updated_merged_annotations_n235_WGS_MethylationCohort.RData"

# set 1
infile.1 <- "./data/raw/PurBeta_adjustedTumor_betaMatrix_V1_V2_reduced_717459commonCpGs_TNBCs_n136.RData"
infile.2 <- "./data/raw/GSE290981_ProcessedData_LUepic_V2_CpGnameChange.txt" # 
infile.3 <- "./data/raw/GSE290981_ProcessedData_LUepic_V1.txt" # 

# set2
infile.4 <- "./data/processed/TNBC_unadjusted.RData"
infile.5 <- "./data/processed/TNBC_adjusted.RData"

#----------------------------------------------------------------------
# output paths
output.path <- "./data/processed/"
dir.create(output.path, showWarnings = FALSE)
outfile.1 <- paste0(output.path, "TNBC_bothCohorts_unadjusted.RData")
outfile.2 <- paste0(output.path, "TNBC_bothCohorts_adjusted.RData")

#######################################################################
# Unadjusted
#######################################################################

# TNBC 136 cohort 2 batches
# batch 1
tnbc_136_unadjusted_1 <- as.data.frame(read.table(infile.2,header=TRUE))
tnbc_136_unadjusted_1 <- tnbc_136_unadjusted_1[, !grepl("Detection_Pval", colnames(tnbc_136_unadjusted_1))]
colnames(tnbc_136_unadjusted_1) <- gsub("\\.tsv$", "", colnames(tnbc_136_unadjusted_1))
colnames(tnbc_136_unadjusted_1) <- sub("\\..*", "", colnames(tnbc_136_unadjusted_1))
tnbc_136_unadjusted_1$ID_REF <- gsub("_.*$", "", tnbc_136_unadjusted_1$ID_REF) # correct cp ids
# batch 2
tnbc_136_unadjusted_2 <- as.data.frame(read.table(infile.3,header=TRUE))
tnbc_136_unadjusted_2 <- tnbc_136_unadjusted_2[, !grepl("Detection_Pval", colnames(tnbc_136_unadjusted_2))]
colnames(tnbc_136_unadjusted_2) <- gsub("\\.tsv$", "", colnames(tnbc_136_unadjusted_2))
colnames(tnbc_136_unadjusted_2) <- sub("\\..*", "", colnames(tnbc_136_unadjusted_2))
# merge into one object
tnbc_136_unadjusted <- merge(tnbc_136_unadjusted_1, tnbc_136_unadjusted_2, by = "ID_REF")
rm(tnbc_136_unadjusted_1)
rm(tnbc_136_unadjusted_2)

# TNBC 235
tnbc_235_unadjusted <- as.data.frame(loadRData(infile.4))
# correct ids
tnbc.idkey <- loadRData(infile.0)
colnames(tnbc_235_unadjusted) <- tnbc.idkey$External_ID_sample[match(
    colnames(tnbc_235_unadjusted),
    tnbc.idkey$PD_ID
)]
tnbc_235_unadjusted$ID_REF <- rownames(tnbc_235_unadjusted)

# make both tnbc cohorts into one
#dim(tnbc_136_unadjusted)
#tnbc_136_unadjusted[1:5,1:5]
#dim(tnbc_235_unadjusted)
#tnbc_235_unadjusted[1:5,1:5]

merged_tnbc_unadjusted <- merge(tnbc_136_unadjusted, tnbc_235_unadjusted, by = "ID_REF")
dim(merged_tnbc_unadjusted)

#######################################################################
# Ajusted
#######################################################################

# TNBC extra
tnbc_136_adjusted <- as.data.frame(loadRData(infile.1))
colnames(tnbc_136_adjusted) <- sub("\\..*", "", colnames(tnbc_136_adjusted))
tnbc_136_adjusted$ID_REF <- rownames(tnbc_136_adjusted)
tnbc_136_adjusted[1:5,1:5]

# set 2
tnbc_235_adjusted <- as.data.frame(loadRData(infile.5))
colnames(tnbc_235_adjusted) <- tnbc.idkey$External_ID_sample[match(
    colnames(tnbc_235_adjusted),
    tnbc.idkey$PD_ID
)]
tnbc_235_adjusted$ID_REF <- rownames(tnbc_235_adjusted)

merged_tnbc_adjusted <- merge(tnbc_136_adjusted, tnbc_235_adjusted, by = "ID_REF")
dim(merged_tnbc_adjusted)

# filter down the unadjusted to same set of cpgs
merged_tnbc_unadjusted <- merged_tnbc_unadjusted[merged_tnbc_unadjusted$ID_REF %in% merged_tnbc_adjusted$ID_REF, ]
dim(merged_tnbc_unadjusted)

#######################################################################
# save output files
#######################################################################
merged_tnbc_unadjusted <- merged_tnbc_unadjusted[c("ID_REF", sort(setdiff(names(merged_tnbc_unadjusted), "ID_REF")))]
merged_tnbc_adjusted <- merged_tnbc_adjusted[c("ID_REF", sort(setdiff(names(merged_tnbc_adjusted), "ID_REF")))]

save(merged_tnbc_unadjusted, file=outfile.1)
save(merged_tnbc_adjusted, file=outfile.2) 