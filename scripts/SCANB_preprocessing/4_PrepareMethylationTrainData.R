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
infile.0 <- "./data/set_definitions/train_ids.csv"
infile.1 <- "./data/processed/MO_train_adjusted.RData"
infile.2 <- "./data/processed/MO_train_unadjusted.RData"
# 499 erp 
infile.3 <- "./data/raw/PurBeta_adjustedTumor_betaMatrix_V1_V2_reduced_717459commonCpGs_TNBCs_n136.RData"
infile.4 <- "./data/raw/GSE290981_ProcessedData_LUepic_V2_CpGnameChange.txt" # 
infile.5 <- "./data/raw/GSE290981_ProcessedData_LUepic_V1.txt" # 

#infile.2 <- "./data/raw/" # NatMed TNBC cohort
#infile.3 <- "./data/raw/" # MO cohort data
#----------------------------------------------------------------------
# output paths
output.path <- "./data/standardized/"
dir.create(output.path, showWarnings = FALSE)
outfile.1 <- paste0(output.path, "SCANB_DNAmethylation.csv")
#######################################################################
# check unadj. 136 data samples that are in both to see if correct
#######################################################################

train.ids <- read.table(infile.0)[[1]]
head(train.ids)
mo.train.unadj <- loadRData(infile.2)

x <- read.table(infile.4,header=TRUE)
x <- x[, !grepl("Detection_Pval", colnames(x))]
head(x)
dim(x)
colnames(x) <- gsub("\\.tsv$", "", colnames(x))
y <- read.table(infile.5,header=TRUE)
y <- y[, !grepl("Detection_Pval", colnames(y))]
dim(y)
head(y)
colnames(y) <- gsub("\\.tsv$", "", colnames(y))

intersect(colnames(mo.train.unadj), colnames(x))
intersect(colnames(mo.train.unadj), colnames(y))
mo.train.unadj <- as.data.frame(mo.train.unadj)
mo.train.unadj$ID_REF <- rownames(mo.train.unadj)




y[1:5, c("ID_REF", "S003585.l.d.mth")] # y is correct, it matches the mo cohort

mo.train.unadj[1:5, c("ID_REF","S003585.l.d.mth")]

x$ID_REF <- gsub("_.*$", "", x$ID_REF)
x[1:5, c("ID_REF", "S004240.l2.d.mth")] # x is also correct except for the weird copy ids remove _ stuff
mo.train.unadj[1:5, c("ID_REF","S004240.l2.d.mth")]




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
# write.csv(allMethyl.dat, file=outfile.1, row.names = FALSE)
