#!/usr/bin/env Rscript
# Script: Standardizing patient identifiers and structure across SCAN-B datasets
# Author: Lennart Hohmann
# Date: 10.03.2025
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
#----------------------------------------------------------------------
# input paths
infile.1 <- "./data/standardized/SCANB_FullFU/SCANB_sample_modalities.csv"
infile.2 <- "./data/standardized/SCANB_FullFU/SCANB_clinical.csv"
infile.3 <- "./data/standardized/SCANB_FullFU/SCANB_RNAseq_expression.csv"
#infile.4 <- "./data/standardized/SCANB_FullFU/SCANB_RNAseq_mutations.csv" 
infile.5 <- "./data/standardized/SCANB_FullFU/SCANB_DNAmethylation.csv"
#----------------------------------------------------------------------
# output paths
output.path <- "./data/standardized/SCANB_ProjectCohort/"
dir.create(output.path)
outfile.1 <- paste0(output.path,"SCANB_PC_sample_modalities.csv")
outfile.2 <- paste0(output.path,"SCANB_PC_clinical.csv")
outfile.3 <- paste0(output.path,"SCANB_PC_RNAseq_expression.csv")
#outfile.4 <- paste0(output.path,"SCANB_PC_RNAseq_mutations.csv")
outfile.5 <- paste0(output.path,"SCANB_PC_DNAmethylation.csv")

#######################################################################
# load data
#######################################################################

sample_modalities <- read.csv(infile.1)
clinical <- read.csv(infile.2)
RNAseq_expr <- data.table::fread(infile.3)
RNAseq_expr <- as.data.frame(RNAseq_expr)
#RNAseq_mut <- data.table::fread(infile.4)
#RNAseq_mut <- as.data.frame(RNAseq_mut)
DNAmethyl <- data.table::fread(infile.5)
DNAmethyl <- as.data.frame(DNAmethyl)

#######################################################################
# exclude non project samples
#######################################################################
sample_modalities <- sample_modalities[sample_modalities$DNAmethylation == 1, ]
#table(sample_modalities$TRAIN,sample_modalities$TEST)
clinical <- clinical[clinical$Sample %in% sample_modalities$Sample,]
RNAseq_expr <- RNAseq_expr[, c("Gene", intersect(colnames(RNAseq_expr), sample_modalities$Sample))]
#RNAseq_mut <- RNAseq_mut[, c("Gene", intersect(colnames(RNAseq_mut), sample_modalities$Sample))]
DNAmethyl <- DNAmethyl[c("ID_REF", intersect(colnames(DNAmethyl), sample_modalities$Sample))]

#######################################################################
# save output files
#######################################################################

write.csv(sample_modalities, file=outfile.1, row.names = FALSE)
write.csv(clinical, file=outfile.2, row.names = FALSE)
write.csv(RNAseq_expr, file=outfile.3, row.names = FALSE)
write.csv(DNAmethyl, file=outfile.5, row.names = FALSE)
