#!/usr/bin/env Rscript
# Script: Standardizing clinical training data for SCANB; define sample subsets
# Author: Lennart Hohmann
# Date: 21.05.2025
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
infile.1 <- "./data/raw/Summarized_SCAN_B_rel4_NPJbreastCancer_with_ExternalReview_Bosch_data.RData"
#----------------------------------------------------------------------
# output paths
output.path <- "./data/train/"
dir.create(output.path, showWarnings = FALSE)
outfile.1 <- paste0(output.path, "train_clinical.csv")
dir.create("./data/set_definitions/train_subcohorts/", showWarnings = FALSE)
outfile.2 <- "./data/set_definitions/train_subcohorts/ERpHER2n_train_ids.csv"
outfile.3 <- "./data/set_definitions/train_subcohorts/TNBC_train_ids.csv"

#######################################################################
# clinical data
#######################################################################

train.ids <- read.table(infile.0)[[1]]

clin.dat <- loadRData(infile.1)
clin.dat <- clin.dat[clin.dat$Follow.up.cohort == TRUE,]
clin.dat <- clin.dat[clin.dat$Sample %in% train.ids,]

clin.dat$LN <- ifelse(clin.dat$LN > 0, "N+", "N0")
clin.dat$PR[clin.dat$PR==""] <- NA

# select clinical variables to include
clin.dat <- clin.dat[c("Sample","ER","PR","HER2","LN",
     "NHG","Size.mm","TreatGroup","InvCa.type",
     "Age","NCN.PAM50",
     "OS_days","OS_event",
     "RFi_days","RFi_event")]

# convert outcome to years
clin.dat$OS_years <- clin.dat$OS_days / 365
clin.dat$RFi_years <- clin.dat$RFi_days / 365
clin.dat$OS_days <- NULL
clin.dat$RFi_days <- NULL

# split treatment
clin.dat$TreatGroup[clin.dat$TreatGroup == ""] <- NA
clin.dat$TreatGroup[is.na(clin.dat$TreatGroup)] <- "Missing"

# create column for clinical groups (ER and HER2 status)
clin.dat$Group <- ifelse(
    clin.dat$ER == "Positive" & clin.dat$HER2 == "Negative",
    "ER+HER2-",
        ifelse(
            clin.dat$HER2 == "Negative" & clin.dat$ER == "Negative" & clin.dat$PR == "Negative", 
            "TNBC",
                "Other"))

clin.dat[clin.dat == ""] <- NA

#######################################################################
# save
#######################################################################

fwrite(clin.dat, file=outfile.1, na = "NA")

ERpHER2n_train_ids <- clin.dat$Sample[clin.dat$Group=="ER+HER2-"]
TNBC_train_ids <- clin.dat$Sample[clin.dat$Group=="TNBC"]

#table(clin.dat$Group)

write.table(ERpHER2n_train_ids, file = outfile.2, row.names = FALSE, col.names = FALSE) #erp
write.table(TNBC_train_ids, file = outfile.3, row.names = FALSE, col.names = FALSE) #tnbc