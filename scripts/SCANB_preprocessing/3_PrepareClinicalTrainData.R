#!/usr/bin/env Rscript
# Script: Standardizing methylation data for SCANB, creating 1 file with all training data
# Author: Lennart Hohmann
# Date: 24.03.2025
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
source("./src/untils.R")
#----------------------------------------------------------------------
# input paths
infile.0 <- "./data/raw/Updated_merged_annotations_n235_WGS_MethylationCohort.RData"
infile.1 <- "./data/raw/GSE278586_ProcessedData_LUepic_n499.txt"
infile.2 <- "./data/raw/PurBeta_adjustedTumor_betaMatrix_V1_V2_reduced_717459commonCpGs_TNBCs_n136.RData"
infile.3 <- "./data/raw/TNBC_adjustedBetaTumor_workspace_235samples_trimmedCpGs_updatedAnno.RData"

#infile.2 <- "./data/raw/" # NatMed TNBC cohort
#infile.3 <- "./data/raw/" # MO cohort data
#----------------------------------------------------------------------
# output paths
output.path <- "./data/standardized/"
dir.create(output.path, showWarnings = FALSE)
outfile.1 <- paste0(output.path,"SCANB_DNAmethylation.csv")
#######################################################################
# HER2E paper data (n=499)
#######################################################################
DNA_methyl.dat <- data.table::fread(infile.1)
DNA_methyl.dat <- as.data.frame(DNA_methyl.dat)
DNA_methyl.dat <- DNA_methyl.dat[!grepl("Detection_Pval", names(DNA_methyl.dat))]
names(DNA_methyl.dat)[-1] <- sub("\\..*", "", names(DNA_methyl.dat[-1]))
#######################################################################
# MO data (n=) 
#######################################################################
#mo.train <- loadRData(infile.5)
#mo.train.ids <- mo.train$SpecimenName
#mo.test <- loadRData(infile.6)
#mo.test.ids <- mo.test$SpecimenName
#######################################################################
# NatMed TNBC paper data (n=)
#######################################################################

#tnbc.dat.1 <- loadRData(infile.2)
#tnbc.dat.1 <- as.data.frame(tnbc.dat.1)

#head(tnbc.dat.2)

tnbc.dat.2 <- loadRData(infile.3)
tnbc.dat.2 <- as.data.frame(tnbc.dat.2)

tnbc.anno <- loadRData(infile.0)
tnbc.anno <- as.data.frame(tnbc.anno)[c("PD_ID","OS","OSbin","RFIbin","RFI")]
#dim(tnbc.anno)
write.csv(tnbc.dat.2, file = "./data/raw/tnbc235.csv", row.names = TRUE)
write.csv(tnbc.anno, file = "./data/raw/tnbc_anno.csv", row.names = FALSE)
#View(tnbc.anno)
#######################################################################
# Overlap, defining final dataset
#######################################################################
#length(intersect(names(DNA_methyl.dat)[-1], c(mo.test.ids,mo.train.ids))) # 310 overlap

# filter to include the same CpG set for all samples



#######################################################################
# save output files
#######################################################################
write.csv(allMethyl.dat, file=outfile.1, row.names = FALSE)
