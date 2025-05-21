#!/usr/bin/env Rscript
# Script: Standardizing uncorrected methylation data for SCANB
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
infile.0 <- "./data/set_definitions/train_ids.csv"
# MO
infile.1 <- "./data/processed/MO_train_unadjusted.RData"
# TNBC extra
#infile.2 <- "./data/raw/GSE290981_ProcessedData_LUepic_V2_CpGnameChange.txt" # 
#infile.3 <- "./data/raw/GSE290981_ProcessedData_LUepic_V1.txt" # 
infile.2 <- "./data/processed/TNBC_bothCohorts_unadjusted.RData"
# ER+HER2- extra
infile.4 <- "./data/processed/ERpHER2n_WGSset_unadjusted.RData"

infile.5 <- "./data/set_definitions/CpG_set.csv"
#----------------------------------------------------------------------
# output paths
output.path <- "./data/train/"
dir.create(output.path, showWarnings = FALSE)
outfile.1 <- paste0(output.path, "train_methylation_unadjusted.csv")
#######################################################################
# load data
#######################################################################

train.ids <- read.table(infile.0)[[1]]

# MO
MO_train_unadjusted <- as.data.frame(loadRData(infile.1))
colnames(MO_train_unadjusted) <- sub("\\..*", "", colnames(MO_train_unadjusted))

# TNBC extra
tnbc <- as.data.frame(loadRData(infile.2))
dim(tnbc)

# ER+HER2- extra
ERpHER2n_unadjusted <- as.data.frame(loadRData(infile.4))
colnames(ERpHER2n_unadjusted) <- sub("\\..*", "", colnames(ERpHER2n_unadjusted))

#######################################################################
# prepare methylation datasets for training
#######################################################################

extra_samples <- setdiff(train.ids, colnames(MO_train_unadjusted))

# merge on cpg_id
MO_train_unadjusted$ID_REF <- rownames(MO_train_unadjusted)

tnbc_toadd <- tnbc[colnames(tnbc) %in% c("ID_REF",extra_samples)]
dim(tnbc_toadd)
rm(tnbc)

ERpHER2n_unadjusted$ID_REF <- rownames(ERpHER2n_unadjusted)
ERpHER2n_unadjusted_toadd <- ERpHER2n_unadjusted[colnames(ERpHER2n_unadjusted) %in% c("ID_REF",extra_samples)]
rm(ERpHER2n_unadjusted)

#dim(ERpHER2n_unadjusted_toadd)
#train_unadjusted <- merge(MO_train_unadjusted, tnbc_136_unadjusted_toadd, by = "ID_REF") # not enough memory
#train_unadjusted <- merge(train_unadjusted, ERpHER2n_unadjusted_toadd, by = "ID_REF")

shared_cpgs <- read.table(infile.5)[[1]] # take set of cpgs from adjusted data (has a few less)

# Helper function
subset_and_order <- function(df, shared_cpgs) {
  df_subset <- df[df$ID_REF %in% shared_cpgs, ]
  df_subset[order(df_subset$ID_REF), ]
}

MO_train_unadjusted <- subset_and_order(MO_train_unadjusted, shared_cpgs)
tnbc_toadd <- subset_and_order(tnbc_toadd, shared_cpgs)
ERpHER2n_unadjusted_toadd <- subset_and_order(ERpHER2n_unadjusted_toadd, shared_cpgs)

all.equal(MO_train_unadjusted$ID_REF, tnbc_toadd$ID_REF) #true
all.equal(MO_train_unadjusted$ID_REF, ERpHER2n_unadjusted_toadd$ID_REF) #true

# Final combined data
train_unadjusted <- cbind(
    MO_train_unadjusted, 
    tnbc_toadd[, -which(names(tnbc_toadd) == "ID_REF")], 
    ERpHER2n_unadjusted_toadd[, -which(names(ERpHER2n_unadjusted_toadd) == "ID_REF")])

dim(train_unadjusted)
#train_unadjusted <- train_unadjusted[, c("ID_REF", setdiff(names(train_unadjusted), "ID_REF"))]
train_unadjusted <- train_unadjusted[, c("ID_REF", train.ids)]

#######################################################################
# save output files
#######################################################################

setDT(train_unadjusted)
setcolorder(train_unadjusted, c("ID_REF", sort(setdiff(names(train_unadjusted), "ID_REF"))))
fwrite(train_unadjusted, file=outfile.1, na = "NA")
