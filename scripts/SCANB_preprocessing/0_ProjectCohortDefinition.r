#!/usr/bin/env Rscript
# Script: Defining the Project training cohort, by merging DNA methylation cohorts and excluding samples in MO test set
# Author: Lennart Hohmann
# Date: 14.05.2025
#----------------------------------------------------------------------
# empty environment
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence/")
#----------------------------------------------------------------------
# packages
#source("./scripts/src/")
if (!require("pacman")) install.packages("pacman")
pacman::p_load(data.table,caret)
source("./src/untils.R")
#----------------------------------------------------------------------
# input paths
infile.0 <- "./data/raw/id_multiomics_TEST_cohort_EPICassay.Rdata"
infile.1 <- "./data/processed/MO_train_unadjusted.RData"
infile.2 <- "./data/processed/ERpHER2n_WGSset_unadjusted.RData"
infile.3 <- "./data/processed/TNBC_unadjusted.RData"
infile.4 <- "./data/standardized/SCANB_FullFU/SCANB_clinical.csv"
infile.5 <- "./data/raw/Updated_merged_annotations_n235_WGS_MethylationCohort.RData"
infile.6 <- "./data/raw/PurBeta_adjustedTumor_betaMatrix_V1_V2_reduced_717459commonCpGs_TNBCs_n136.RData"

#----------------------------------------------------------------------
# output paths
outfile.1 <- "./data/set_definitions/PC_train.csv"
outfile.2 <- "./data/set_definitions/PC_test.csv" # depends on if MO test has sufficient events or not (if not add extra samples to it)

###################
# LOAD INPUT FILES
###################

mo.test <- loadRData(infile.0)
mo.train <- loadRData(infile.1)
erp <- loadRData(infile.2)
tnbc <- loadRData(infile.3)
tnbc.idkey <- loadRData(infile.5)
anno <- read.csv(infile.4, header = TRUE, sep = ",")

##########################
# TEST SET EVENT NUM CHECK
##########################

mo.test$Sample <- mo.test$SpecimenName
mo.test.anno <- merge(mo.test, anno, by = "Sample")

table(mo.test.anno$clinGroup, mo.test.anno$OS_event)
table(mo.test.anno$clinGroup, mo.test.anno$RFi_event)

# need more events in test set

#######################################################
# STANDARDIZE SAMPLE IDS
#######################################################

#mo.test$Sample # is in right format
# standardize sample ids
colnames(mo.train) <- gsub("\\..*$", "", colnames(mo.train))
colnames(erp) <- gsub("\\..*$", "", colnames(erp))
# convert ids for tnbc
colnames(tnbc) <- tnbc.idkey$External_ID_sample[match(
    colnames(tnbc),
    tnbc.idkey$PD_ID
)]

#######################################################
# IDENTIFY EXTRA SAMPLES (NOT IN TRAIN OR TEST ALREADY)
#######################################################
mo.all <- c(mo.test$Sample, colnames(mo.train))

length(setdiff(colnames(erp), mo.all)) # 189 extra samples
# length(setdiff(colnames(tnbc), mo.all)) # 1 extra sample, drop

#######################################################
# SPLIT INTO TRAIN AND TEST TO ADD TO THE EXISTING SETS
# TAKING OUTCOME INTO CONSIDERATION
#######################################################

# Step 1: Select erp samples not already in mo.all
extra.samples <- setdiff(colnames(erp), mo.all)
extra.samples.anno <- data.frame("Sample"=extra.samples) 
extra.samples.anno$RFI_event <- anno$RFi_event[match(extra.samples, anno$Sample)]
sum(is.na(extra.samples.anno$RFI_event)) # 0
# Step 2: Create stratified split based on RFI events
set.seed(22)  # For reproducibility
samp <- createDataPartition(extra.samples.anno$RFI_event, p = 0.75, list = FALSE)
# Step 3: Create train and test sets
added.samples.train <- extra.samples.anno[samp, ]
added.samples.test  <- extra.samples.anno[-samp, ]

# check
#table(added.samples.train$RFI_event)
# table(added.samples.test$RFI_event)

##############################
# DEFINE FINAL TEST TRAIN SETS
##############################

train.ids <- c(colnames(mo.train),added.samples.train$Sample)
test.ids <- c(mo.test$Sample,added.samples.test$Sample)
length(train.ids) # 1567
length(test.ids) # 759

###################
# SAVE OUTPUT FILES
###################

write.table(train.ids, file = outfile.1, row.names = FALSE, col.names = FALSE)
write.table(test.ids, file = outfile.2, row.names = FALSE, col.names = FALSE)
