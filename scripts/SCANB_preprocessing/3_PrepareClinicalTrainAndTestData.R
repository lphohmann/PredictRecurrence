#!/usr/bin/env Rscript
# Script: Standardizing clinical train and test data for SCANB; define sample subsets
# Author: Lennart Hohmann
# Date: 21.05.2025
#----------------------------------------------------------------------
# empty environment
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence/")
#----------------------------------------------------------------------
# packages
# Install if needed
if (!require("mice")) install.packages("mice")
library(mice)
library(data.table)

source("./src/utils.R")
#----------------------------------------------------------------------
# input paths
infile.0 <- "./data/set_definitions/train_ids.csv"
infile.1 <- "./data/raw/Summarized_SCAN_B_rel4_NPJbreastCancer_with_ExternalReview_Bosch_data.RData"
infile.2 <- "./data/set_definitions/test_ids.csv"
#----------------------------------------------------------------------
# output paths
output.path <- "./data/train/"
dir.create(output.path, showWarnings = FALSE)
outfile.0 <- paste0(output.path, "test_clinical.csv")
outfile.1 <- paste0(output.path, "train_clinical.csv")
dir.create("./data/train/train_subcohorts/", showWarnings = FALSE)
#outfile.2 <- "./data/train/train_subcohorts/ERpHER2n_train_ids.csv"
#outfile.3 <- "./data/train/train_subcohorts/TNBC_train_ids.csv"
#outfile.4 <- "./data/train/train_subcohorts/All_train_ids.csv"

#######################################################################
# clinical data
#######################################################################
test.ids <- read.table(infile.2)[[1]]
train.ids <- read.table(infile.0)[[1]]

clin.dat <- loadRData(infile.1)
clin.dat <- clin.dat[clin.dat$Follow.up.cohort == TRUE,]
#clin.dat <- clin.dat[clin.dat$Sample %in% train.ids,]
clin.dat <- clin.dat[clin.dat$Sample %in% c(train.ids,test.ids),]
#dim(clin.dat)
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


#----------------------------------------------------------------------
# train
# impute missing data for Age, Size.mm NHG LN

# check missingness
nrow(clin.dat)
print(colSums(is.na(clin.dat[c("Age", "Size.mm", "NHG", "LN")])))
# minimal impute (drop this block where you had your comment)
mode_value <- function(x) {
  ux <- na.omit(unique(x))
  ux[which.max(tabulate(match(x, ux)))]
}

# Size.mm (numeric)
clin.dat$Size.mm[is.na(clin.dat$Size.mm)] <- median(clin.dat$Size.mm, na.rm = TRUE)

# LN (categorical)
clin.dat$LN[is.na(clin.dat$LN)] <- mode_value(clin.dat$LN)

# NHG (ordinal) -> mode + indicator
clin.dat$NHG_missing <- as.integer(is.na(clin.dat$NHG))
clin.dat$NHG[is.na(clin.dat$NHG)] <- mode_value(clin.dat$NHG)

# check missingness again
print(colSums(is.na(clin.dat[c("Age", "Size.mm", "NHG", "LN")])))

# correct data taypeds for modeeling:
#clin.dat$LN <- ifelse(clin.dat$LN == "N+", 1,
#                          ifelse(clin.dat$LN == "N0", 0, NA))
#clin.dat[c("Age","Size.mm","NHG","LN")] <- lapply(clin.dat[c("Age","Size.mm","NHG","LN")], as.numeric)

#----------------------------------------------------------------------
clin.dat.train <- clin.dat[clin.dat$Sample %in% train.ids,]
clin.dat.test <- clin.dat[clin.dat$Sample %in% test.ids,]
dim(clin.dat.test)

#######################################################################
# save
#######################################################################
head(clin.dat.train)
colnames(clin.dat.train)
fwrite(clin.dat.train, file=outfile.1, na = "NA")
fwrite(clin.dat.test, file=outfile.0, na = "NA")

ERpHER2n_train_ids <- clin.dat.train$Sample[clin.dat.train$Group=="ER+HER2-"]
TNBC_train_ids <- clin.dat.train$Sample[clin.dat.train$Group=="TNBC"]

#table(clin.dat.train$Group)

#write.table(ERpHER2n_train_ids, file = outfile.2, row.names = FALSE, col.names = FALSE) #erp
#write.table(TNBC_train_ids, file = outfile.3, row.names = FALSE, col.names = FALSE) #tnbc
#write.table(train.ids, file = outfile.4, row.names = FALSE, col.names = FALSE) #all