#!/usr/bin/env Rscript
# Script: Defining test and train datasets
# Author: Lennart Hohmann
# Date: 24.03.2025
# test change
#----------------------------------------------------------------------
# empty environment
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence/")
#----------------------------------------------------------------------
# packages
#source("./scripts/src/")
if (!require("pacman")) install.packages("pacman")
pacman::p_load(caret)
#----------------------------------------------------------------------
# input paths
#infile.1 <- "./data/standardized/" # clinical data for all samples
# if predefined split is used
infile.2 <- "./data/raw/multiomics_cohort_EPIC_TRAIN.RData" 
infile.3 <- "./data/raw/id_multiomics_TEST_cohort_EPICassay.Rdata"
#----------------------------------------------------------------------
# output paths
output.path <- "./data/set_definitions/"
dir.create(output.path, showWarnings = FALSE)
outfile.1 <- paste0(output.path, "SCANB_TrainTestSets.csv")
#######################################################################
# function: loads RData data file and allows to assign it directly to variable
loadRData <- function(file.path){
  load(file.path)
  get(ls()[ls() != "file.path"])
}
#######################################################################
# load data
#######################################################################

# compare clin dat in terms of event, delete later
# clin.dat <- read.csv("./data/standardized/SCANB_FullFU/SCANB_clinical.csv")
# head(clin.dat)
# mo.train <- loadRData(infile.2)
# x <- mo.train[mo.train$SpecimenName %in% clin.dat$Sample, c("SpecimenName", "oas_event", "rfi_event")]
# names(x)[1] <- c("Sample")
# x[2:3] <- sapply(x[2:3], as.numeric)
# y <- clin.dat[clin.dat$Sample %in% mo.train$SpecimenName, c("Sample", "OS_event", "RFi_event")]
# z <- merge(x, y, by = "Sample")
# View(z)
# table(z$oas_event,z$OS_event)
# table(z$rfi_event, z$RFi_event)
#######################################################################
# predefined MO test and training cohort
#######################################################################
mo.train <- loadRData(infile.2)
mo.train.ids <- mo.train$SpecimenName
mo.test <- loadRData(infile.3)
mo.test.ids <- mo.test$SpecimenName
#######################################################################
# Check which samples to add from the methlyation HERE cohort
#######################################################################

# assign the ones not overlapping to train/test sets
#added.methyl.samples <- clin.dat[clin.dat$Sample %in% setdiff(names(DNA_methyl.dat)[-1], c(mo.test.ids,mo.train.ids)),]
# consider TreatGroup as this should take the important ClinPath vars into consideration (they are all ER+HER2-)
#samp <- createDataPartition(as.factor(added.methyl.samples$TreatGroup), p = 0.75, list = F)
#added.methyl.samples.train <- added.methyl.samples[samp,]
#added.methyl.samples.test <- added.methyl.samples[-samp,]


#######################################################################
# save output file
#######################################################################
write.csv(TrainTest.df, file=outfile.1, row.names = FALSE)
