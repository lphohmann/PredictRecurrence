#!/usr/bin/env Rscript
# Script: Defining the training and test sets
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
source("./src/utils.R")
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
outfile.1 <- "./data/set_definitions/train_ids.csv"
outfile.2 <- "./data/set_definitions/test_ids.csv" # depends on if MO test has sufficient events or not (if not add extra samples to it)

###################
# LOAD INPUT FILES
###################

anno <- read.csv(infile.4, header = TRUE, sep = ",")

mo.test <- loadRData(infile.0)
mo.train <- loadRData(infile.1)

erp <- loadRData(infile.2)

tnbc <- loadRData(infile.3)
tnbc.idkey <- loadRData(infile.5)
tnbc_136 <- loadRData(infile.6)
colnames(tnbc_136) <- gsub("\\..*$", "", colnames(tnbc_136))

##########################
# TEST SET EVENT NUM CHECK
##########################

mo.test$Sample <- mo.test$SpecimenName
mo.test.anno <- merge(mo.test, anno, by = "Sample")

table(mo.test.anno$clinGroup, mo.test.anno$OS_event)
table(mo.test.anno$clinGroup, mo.test.anno$RFi_event)

#########################
# STANDARDIZE SAMPLE IDS
#########################

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
length(mo.all)
length(setdiff(colnames(erp), mo.all)) # 189 extra samples, add them
length(setdiff(colnames(tnbc), mo.all)) # 129 extra samples, add them 
length(setdiff(colnames(tnbc_136), mo.all)) # 21 extra samples, add them 

#tnbc_extra.df <- data.frame("Sample"=c(colnames(tnbc_136),colnames(tnbc)))
#sum(!(tnbc_extra.df$Sample %in% anno$Sample)) # 7 not in anno
#tnbc_extra.df <- merge(tnbc_extra.df, anno, by = "Sample")
#tnbc_extra.df <- tnbc_extra.df[!(tnbc_extra.df$Sample %in% mo.all),]

#table(tnbc_extra.df$OS_event)
#table(tnbc_extra.df$RFi_event)

#######################################################
# SPLIT INTO TRAIN AND TEST TO ADD TO THE EXISTING SETS
# TAKING OUTCOME INTO CONSIDERATION
#######################################################

# extra ERpHER2n samples
# Step 1: Select erp samples not already in mo.all
extra.samples <- setdiff(colnames(erp), mo.all)
extra.samples.anno <- data.frame("Sample"=extra.samples) 
extra.samples.anno$RFI_event <- anno$RFi_event[match(extra.samples, anno$Sample)]
# Step 2: Create stratified split based on RFI events
set.seed(22)  # For reproducibility
samp <- createDataPartition(extra.samples.anno$RFI_event, p = 0.75, list = FALSE)
# Step 3: Create train and test sets
added.erp.samples.train <- extra.samples.anno[samp, ]
added.erp.samples.test  <- extra.samples.anno[-samp, ]

# check
table(added.erp.samples.train$RFI_event)
table(added.erp.samples.test$RFI_event)

# ------------------------------------------------

# extra TNBC samples
# Step 1: Select  samples not already in mo.all
extra.samples.anno <- data.frame("Sample"=c(colnames(tnbc_136),colnames(tnbc)))
sum(!(extra.samples.anno$Sample %in% anno$Sample)) # 7 not in anno
extra.samples.anno$RFI_event <- anno$RFi_event[match(extra.samples.anno$Sample, anno$Sample)]
extra.samples.anno <- extra.samples.anno[!(extra.samples.anno$Sample %in% mo.all),]
dim(extra.samples.anno)
table(is.na(extra.samples.anno$RFI_event)) # 16 have no RFI
extra.samples.anno <- extra.samples.anno[!is.na(extra.samples.anno$RFI_event),]
dim(extra.samples.anno)

# Step 2: Create stratified split based on RFI events
samp <- createDataPartition(extra.samples.anno$RFI_event, p = 0.75, list = FALSE)
# Step 3: Create train and test sets
added.tnbc.samples.train <- extra.samples.anno[samp, ]
added.tnbc.samples.test  <- extra.samples.anno[-samp, ]

# check
table(added.tnbc.samples.train$RFI_event)
table(added.tnbc.samples.test$RFI_event)

##############################
# DEFINE TEST TRAIN SETS
##############################

train.ids <- c(colnames(mo.train), added.erp.samples.train$Sample, added.tnbc.samples.train$Sample)
test.ids <- c(mo.test$Sample, added.erp.samples.test$Sample, added.tnbc.samples.test$Sample)
length(train.ids) # 1668
length(test.ids) # 792

####################################
# FILT CASES W/O AVAIL RFI OUTCOME DATA?
####################################

table(is.na(anno[anno$Sample %in% test.ids, "RFi_event"]))
table(is.na(anno[anno$Sample %in% train.ids, "RFi_event"]))

table(is.na(anno[anno$Sample %in% test.ids, "DRFi_event"]))
table(is.na(anno[anno$Sample %in% train.ids, "DRFi_event"]))

table(is.na(anno[anno$Sample %in% test.ids, "OS_event"]))
table(is.na(anno[anno$Sample %in% train.ids, "OS_event"]))

table(anno[anno$Sample %in% test.ids, c("RFi_event","DRFi_event")])
table(anno[anno$Sample %in% train.ids, c("RFi_event","DRFi_event")])


train.ids <- anno$Sample[anno$Sample %in% train.ids & !is.na(anno$RFi_event)]
test.ids <- anno$Sample[anno$Sample %in% test.ids & !is.na(anno$RFi_event)]
length(train.ids) # 1347
length(test.ids) # 636

##############################
# EVENT NUMS IN SETS
##############################

table(anno[anno$Sample %in% train.ids, c("RFi_event","Group")])
table(anno[anno$Sample %in% test.ids, c("RFi_event","Group")])

###################
# SAVE OUTPUT FILES
###################

write.table(train.ids, file = outfile.1, row.names = FALSE, col.names = FALSE)
write.table(test.ids, file = outfile.2, row.names = FALSE, col.names = FALSE)
