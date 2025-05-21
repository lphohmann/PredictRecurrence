#!/usr/bin/env Rscript
# Script: Standardizing TCC adjusted methylation data for SCANB
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
infile.1 <- "./data/processed/MO_train_adjusted.RData"
# TNBC extra
infile.2 <- "./data/processed/TNBC_bothCohorts_adjusted.RData"
#infile.2 <- "./data/raw/PurBeta_adjustedTumor_betaMatrix_V1_V2_reduced_717459commonCpGs_TNBCs_n136.RData"
# ER+HER2- extra
infile.3 <- "./data/processed/ERpHER2n_WGSset_adjusted.RData"
#----------------------------------------------------------------------
# output paths
output.path <- "./data/train/"
dir.create(output.path, showWarnings = FALSE)
outfile.1 <- paste0(output.path, "train_methylation_adjusted.csv")
outfile.2 <- "./data/set_definitions/CpG_set.csv"
#######################################################################
# load data
#######################################################################

train.ids <- read.table(infile.0)[[1]]

# MO
MO_train <- as.data.frame(loadRData(infile.1))
colnames(MO_train) <- sub("\\..*", "", colnames(MO_train))
dim(MO_train)

# TNBC extra
tnbc <- as.data.frame(loadRData(infile.2))
#colnames(tnbc_136) <- sub("\\..*", "", colnames(tnbc_136))
dim(tnbc)

# ER+HER2- extra
ERpHER2n <- as.data.frame(loadRData(infile.3))
colnames(ERpHER2n) <- sub("\\..*", "", colnames(ERpHER2n))
dim(ERpHER2n)

#######################################################################
# prepare methylation datasets for training
#######################################################################

extra_samples <- setdiff(train.ids, colnames(MO_train))

# merge on cpg_id
MO_train$ID_REF <- rownames(MO_train)

tnbc_toadd <- tnbc[colnames(tnbc) %in% c("ID_REF",extra_samples)]
dim(tnbc_toadd)
rm(tnbc)

ERpHER2n$ID_REF <- rownames(ERpHER2n)
ERpHER2n_toadd <- ERpHER2n[colnames(ERpHER2n) %in% c("ID_REF",extra_samples)]
rm(ERpHER2n)

shared_cpgs <- Reduce(intersect, list(
  MO_train$ID_REF,
  tnbc_toadd$ID_REF,
  ERpHER2n_toadd$ID_REF
))

write.table(shared_cpgs, file = outfile.2, row.names = FALSE, col.names = FALSE)

# Helper function
subset_and_order <- function(df, shared_cpgs) {
  df_subset <- df[df$ID_REF %in% shared_cpgs, ]
  df_subset[order(df_subset$ID_REF), ]
}

MO_train <- subset_and_order(MO_train, shared_cpgs)
tnbc_toadd <- subset_and_order(tnbc_toadd, shared_cpgs)
ERpHER2n_toadd <- subset_and_order(ERpHER2n_toadd, shared_cpgs)

dim(ERpHER2n_toadd)
dim(tnbc_toadd)
dim(MO_train)

all.equal(MO_train$ID_REF, tnbc_toadd$ID_REF) #true
all.equal(MO_train$ID_REF, ERpHER2n_toadd$ID_REF) #true

# Final combined data
train_adjusted <- cbind(
    MO_train, 
    tnbc_toadd[, -which(names(tnbc_toadd) == "ID_REF")], 
    ERpHER2n_toadd[, -which(names(ERpHER2n_toadd) == "ID_REF")])

dim(train_adjusted)
#train_adjusted <- train_adjusted[, c("ID_REF", setdiff(names(train_adjusted), "ID_REF"))]

train_adjusted <- train_adjusted[, c("ID_REF", train.ids)]

#######################################################################
# save output files
#######################################################################

setDT(train_adjusted)
setcolorder(train_adjusted, c("ID_REF", sort(setdiff(names(train_adjusted), "ID_REF"))))
fwrite(train_adjusted, file=outfile.1, na = "NA")

#x <- fread("./data/train/train_methylation_adjusted.csv")
#dim(x)
#y <- fread("./data/train/train_methylation_unadjusted.csv")
#dim(y)

#identical(colnames(x), colnames(y))
#x[1:5,1:5]
#y[1:5,1:5]
