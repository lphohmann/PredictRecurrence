#!/usr/bin/env Rscript
# Script: tables in subsets train test, tnbc
# Author: Lennart Hohmann
# Date: 11.05.2025
#-------------------
# empty environment
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence/")
#-------------------
# packages
#source("./scripts/src/")
for(pkg in c("table1", "htmltools", "finalfit", "Publish", "htmlTable", "webshot")){
    if (!requireNamespace(pkg, quietly = TRUE)) {
        install.packages(pkg)
    }
    library(pkg, character.only = TRUE)
}

#-------------------
# set/create output directories
output_path <- "./output/PC_EDA/"
dir.create(output_path)
#-------------------
# input paths
infile_0 <- "./data/set_definitions/train_subcohorts/TNBC_train_ids.csv"
infile_1 <- "./data/train/train_clinical.csv"
infile_2 <- "./data/train/test_clinical.csv"
#-------------------
# output paths
outfile_1 <- paste0(output_path, "TNBC_Train_table1.html")
outfile_2 <- paste0(output_path, "TNBC_Test_table1.html")
outfile_3 <- paste0(output_path, "TNBC_TestVsTrain_table1.html")
#-------------------
# storing objects 
table_list <- list()

#######################################################################
# training set
#######################################################################

train_ids <- read.csv(infile_0,header=FALSE)[[1]]
clinical.train <- read.csv(infile_1)
clinical.train <- clinical.train[clinical.train$Sample %in% train_ids,]
clinical.train$NHG <- as.character(clinical.train$NHG)

# Assign labels for better readability
#label(clinical.train$ER) <- "ER Status"
#label(clinical.train$PR) <- "PR Status"
#label(clinical.train$HER2) <- "HER2 Status"
label(clinical.train$LN) <- "Lymph Node Status"
label(clinical.train$NHG) <- "Nottingham Histologic Grade (NHG)"
label(clinical.train$Size.mm) <- "Tumor Size (mm)"
#label(clinical.train$TreatGroup) <- "Treatment Group"
#label(clinical.train$InvCa.type) <- "Inv. Cancer type"
label(clinical.train$Age) <- "Age (years)"
#label(clinical.train$NCN.PAM50) <- "PAM50 subtype"
#label(clinical.train$OS_event) <- "OS Event"
label(clinical.train$RFi_event) <- "RFI Event"
#label(clinical.train$OS_years) <- "Overall Survival (years)"
#label(clinical.train$RFi_years) <- "Recurrence-Free Interval (years)"
#label(clinical.train$Group) <- "Subgroup"

# Convert categorical variables to factors for proper formatting
#clinical.train$ER <- factor(clinical.train$ER, levels = c("Negative", "Positive"))
#clinical.train$HER2 <- factor(clinical.train$HER2, levels = c("Negative", "Positive"))
#clinical.train$PR <- factor(clinical.train$PR, levels = c("Negative", "Positive"))
clinical.train$LN <- factor(clinical.train$LN)
#clinical.train$Group <- factor(clinical.train$Group, levels = c("ER+HER2-", "TNBC", "Other"))
clinical.train$RFi_event <- factor(clinical.train$RFi_event, levels = c("0", "1"))
#clinical.train$OS_event <- factor(clinical.train$OS_event, levels = c("0","1"))

# Apply custom renders
my.render.cont <- function(x) {
    with(stats.apply.rounding(stats.default(x), digits=2), c("",
        "Mean (SD)"=sprintf("%s (&plusmn; %s)", MEAN, SD)))
}
my.render.cat <- function(x) {
    c("", sapply(stats.default(x), function(y) {
        with(
            y,
            sprintf("%d (%0.0f %%)", FREQ, PCT)
        )
    }))
}

table_1 <- table1(~ LN + NHG + Age + Size.mm + RFi_event,
    data = clinical.train,
    render.continuous = my.render.cont,
    render.categorical = my.render.cat,
topclass="Rtable1-zebra")

save_html(table_1, file = outfile_1)

webshot::webshot(
  url = outfile_1,
  file = sub("\\.html$", ".pdf", outfile_1),
  zoom = 1,
  vwidth = 1200,
  vheight = 800
)

#######################################################################
# test set
#######################################################################

#test_ids <- read.csv(infile_0,header=FALSE)[[1]]
#clinical.test <- read.csv(infile_2)
#clinical.test <- clinical.test[clinical.test$Sample %in% test_ids,]

#clinical.test$NHG <- as.character(clinical.test$NHG)

# Assign labels for better readability

#label(clinical.test$LN) <- "Lymph Node Status"
#label(clinical.test$NHG) <- "Nottingham Histologic Grade (NHG)"
#label(clinical.test$Size.mm) <- "Tumor Size (mm)"

#label(clinical.test$Age) <- "Age (years)"

#label(clinical.test$RFi_event) <- "RFI Event"

#clinical.test$LN <- factor(clinical.test$LN)

#clinical.test$RFi_event <- factor(clinical.test$RFi_event, levels = c("0", "1"))


# Apply custom renders
#my.render.cont <- function(x) {
#    with(stats.apply.rounding(stats.default(x), digits=2), c("",
#        "Mean (SD)"=sprintf("%s (&plusmn; %s)", MEAN, SD)))
#}
#my.render.cat <- function(x) {
#    c("", sapply(stats.default(x), function(y) {
#        with(
#            y,
#            sprintf("%d (%0.0f %%)", FREQ, PCT)
#        )
#    }))
#}

#table_2 <- table1(~ LN + NHG + Age + Size.mm + RFi_event, data = clinical.test,
#                render.continuous=my.render.cont, 
#                render.categorical=my.render.cat,
#topclass="Rtable1-zebra")

#save_html(table_2, file = outfile_2)

#webshot::webshot(
#    url = outfile_2,
#    file = sub("\\.html$", ".pdf", outfile_2),
#    zoom = 1,
#    vwidth = 1200,
#    vheight = 800
#)

#######################################################################
# test vs train set
#######################################################################

#clinical <- rbind(clinical.train, clinical.test)

#clinical$Set <- ifelse(clinical$Sample %in% clinical.train$Sample, "Train", "Test")
#
#clinical$NHG <- as.character(clinical$NHG)

# Assign labels for better readability
#label(clinical$Set) <- "Set"

#label(clinical$LN) <- "Lymph Node Status"
#label(clinical$NHG) <- "Nottingham Histologic Grade (NHG)"
#label(clinical$Size.mm) <- "Tumor Size (mm)"
#label(clinical$Age) <- "Age (years)"
#label(clinical$RFi_event) <- "RFI Event"

# Convert categorical variables to factors for proper formatting

#clinical$LN <- factor(clinical$LN)

#clinical$RFi_event <- factor(clinical$RFi_event, levels = c("0", "1"))

#clinical$Set <- factor(clinical$Set, levels = c("Train", "Test"))

# Apply custom renders
#my.render.cont <- function(x) {
#    with(stats.apply.rounding(stats.default(x), digits=2), c("",
#        "Mean (SD)"=sprintf("%s (&plusmn; %s)", MEAN, SD)))
#}
#my.render.cat <- function(x) {
#    c("", sapply(stats.default(x), function(y) {
#        with(
#            y,
#            sprintf("%d (%0.0f %%)", FREQ, PCT)
#        )
#    }))
#}

#table_3 <- table1(~  LN + NHG + Age + Size.mm + RFi_event | Set, data = clinical,
#                overall=FALSE,render.continuous=my.render.cont, 
#                render.categorical=my.render.cat,
#topclass="Rtable1-zebra")

#save_html(table_3, file = outfile_3)

#webshot::webshot(
#    url = outfile_3,
#    file = sub("\\.html$", ".pdf", outfile_3),
#    zoom = 1,
#    vwidth = 1200,
#    vheight = 800
#)