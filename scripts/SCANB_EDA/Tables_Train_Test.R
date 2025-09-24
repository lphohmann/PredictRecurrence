#!/usr/bin/env Rscript
# Script: tables in subsets train test
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
if (!require("pacman")) install.packages("pacman")
pacman::p_load(table1,htmltools,finalfit,Publish,htmlTable,webshot)
#-------------------
# set/create output directories
output_path <- "./output/PC_EDA/"
dir.create(output_path)
#-------------------
# input paths
infile_1 <- "./data/train/train_clinical.csv"
infile_2 <- "./data/train/test_clinical.csv"
#-------------------
# output paths
outfile_1 <- paste0(output_path, "Train_table1.html")
outfile_2 <- paste0(output_path, "Test_table1.html")
outfile_3 <- paste0(output_path, "TestVsTrain_table1.html")
#-------------------
# storing objects 
table_list <- list()

#######################################################################
# training set
#######################################################################

clinical.train <- read.csv(infile_1)
clinical.train$NHG <- as.character(clinical.train$NHG)

# Assign labels for better readability
label(clinical.train$ER) <- "ER Status"
label(clinical.train$PR) <- "PR Status"
label(clinical.train$HER2) <- "HER2 Status"
label(clinical.train$LN) <- "Lymph Node Status"
label(clinical.train$NHG) <- "Nottingham Histologic Grade (NHG)"
label(clinical.train$Size.mm) <- "Tumor Size (mm)"
label(clinical.train$TreatGroup) <- "Treatment Group"
label(clinical.train$InvCa.type) <- "Inv. Cancer type"
label(clinical.train$Age) <- "Age (years)"
label(clinical.train$NCN.PAM50) <- "PAM50 subtype"
label(clinical.train$OS_event) <- "OS Event"
label(clinical.train$RFi_event) <- "RFI Event"
label(clinical.train$OS_years) <- "Overall Survival (years)"
label(clinical.train$RFi_years) <- "Recurrence-Free Interval (years)"
label(clinical.train$Group) <- "Subgroup"

# Convert categorical variables to factors for proper formatting
clinical.train$ER <- factor(clinical.train$ER, levels = c("Negative", "Positive"))
clinical.train$HER2 <- factor(clinical.train$HER2, levels = c("Negative", "Positive"))
clinical.train$PR <- factor(clinical.train$PR, levels = c("Negative", "Positive"))
clinical.train$LN <- factor(clinical.train$LN)
clinical.train$Group <- factor(clinical.train$Group, levels = c("ER+HER2-", "TNBC", "Other"))
clinical.train$RFi_event <- factor(clinical.train$RFi_event, levels = c("0", "1"))
clinical.train$OS_event <- factor(clinical.train$OS_event, levels = c("0","1"))

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

table_1 <- table1(~ ER + HER2 + PR + LN + NHG + Age + Size.mm + OS_event + OS_years + RFi_event + RFi_years | Group,
    data = clinical.train,
    overall = c(left = "Total"), render.continuous = my.render.cont,
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

#-----------------------------------------------
# RFI Events at 1y / 5y / 10y by Subgroup (Train)
#-----------------------------------------------
# Helper function to compute event counts at a specific time
count_rfi_events <- function(data, time_cutoff) {
  data$event_at_time <- with(data, as.numeric(as.character(RFi_event)) == 1 & RFi_years <= time_cutoff)
  counts <- aggregate(event_at_time ~ Group, data = data, sum)
  total <- data.frame(Group = "All", 
                      event_at_time = sum(data$event_at_time, na.rm = TRUE))
  rbind(counts, total)
}
# Count RFI events at 1, 5, 10 years
rfi_1y <- count_rfi_events(clinical.train, 1)
rfi_5y <- count_rfi_events(clinical.train, 5)
rfi_10y <- count_rfi_events(clinical.train, 10)
# Merge by Group
rfi_table <- Reduce(function(x, y) merge(x, y, by = "Group", all = TRUE),
                    list(rfi_1y, rfi_5y, rfi_10y))
colnames(rfi_table) <- c("Group", "RFI_1y", "RFI_5y", "RFI_10y")
rfi_table <- rfi_table[order(factor(rfi_table$Group, levels = c("All", levels(clinical.train$Group)))), ]
print(rfi_table)

#######################################################################
# test set
#######################################################################

clinical.test <- read.csv(infile_2)
clinical.test$NHG <- as.character(clinical.test$NHG)

# Assign labels for better readability
label(clinical.test$ER) <- "ER Status"
label(clinical.test$PR) <- "PR Status"
label(clinical.test$HER2) <- "HER2 Status"
label(clinical.test$LN) <- "Lymph Node Status"
label(clinical.test$NHG) <- "Nottingham Histologic Grade (NHG)"
label(clinical.test$Size.mm) <- "Tumor Size (mm)"
label(clinical.test$TreatGroup) <- "Treatment Group"
label(clinical.test$InvCa.type) <- "Inv. Cancer type"
label(clinical.test$Age) <- "Age (years)"
label(clinical.test$NCN.PAM50) <- "PAM50 subtype"
label(clinical.test$OS_event) <- "OS Event"
label(clinical.test$RFi_event) <- "RFI Event"
label(clinical.test$OS_years) <- "Overall Survival (years)"
label(clinical.test$RFi_years) <- "Recurrence-Free Interval (years)"
label(clinical.test$Group) <- "Subgroup"

# Convert categorical variables to factors for proper formatting
clinical.test$ER <- factor(clinical.test$ER, levels = c("Negative", "Positive"))
clinical.test$HER2 <- factor(clinical.test$HER2, levels = c("Negative", "Positive"))
clinical.test$PR <- factor(clinical.test$PR, levels = c("Negative", "Positive"))
clinical.test$LN <- factor(clinical.test$LN)
clinical.test$Group <- factor(clinical.test$Group, levels = c("ER+HER2-", "TNBC", "Other"))
clinical.test$RFi_event <- factor(clinical.test$RFi_event, levels = c("0", "1"))
clinical.test$OS_event <- factor(clinical.test$OS_event, levels = c("0","1"))

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

table_2 <- table1(~ ER + HER2 + PR + LN + NHG + Age + Size.mm + OS_event + OS_years + RFi_event + RFi_years | Group, data = clinical.test,
                overall=c(left="Total"),render.continuous=my.render.cont, 
                render.categorical=my.render.cat,
topclass="Rtable1-zebra")

save_html(table_2, file = outfile_2)

webshot::webshot(
    url = outfile_2,
    file = sub("\\.html$", ".pdf", outfile_2),
    zoom = 1,
    vwidth = 1200,
    vheight = 800
)
#-----------------------------------------------
# RFI Events at 1y / 5y / 10y by Subgroup 
#-----------------------------------------------
# Helper function to compute event counts at a specific time
count_rfi_events <- function(data, time_cutoff) {
  data$event_at_time <- with(data, as.numeric(as.character(RFi_event)) == 1 & RFi_years <= time_cutoff)
  counts <- aggregate(event_at_time ~ Group, data = data, sum)
  total <- data.frame(Group = "All", 
                      event_at_time = sum(data$event_at_time, na.rm = TRUE))
  rbind(counts, total)
}
# Count RFI events at 1, 5, 10 years
rfi_1y <- count_rfi_events(clinical.test, 1)
rfi_5y <- count_rfi_events(clinical.test, 5)
rfi_10y <- count_rfi_events(clinical.test, 10)
# Merge by Group
rfi_table <- Reduce(function(x, y) merge(x, y, by = "Group", all = TRUE),
                    list(rfi_1y, rfi_5y, rfi_10y))
colnames(rfi_table) <- c("Group", "RFI_1y", "RFI_5y", "RFI_10y")
rfi_table <- rfi_table[order(factor(rfi_table$Group, levels = c("All", levels(clinical.test$Group)))), ]
print(rfi_table)

#######################################################################
# test vs train set
#######################################################################

clinical <- rbind(clinical.train, clinical.test)

clinical$Set <- ifelse(clinical$Sample %in% clinical.train$Sample, "Train", "Test")

clinical$NHG <- as.character(clinical$NHG)

# Assign labels for better readability
label(clinical$Set) <- "Set"
label(clinical$ER) <- "ER Status"
label(clinical$PR) <- "PR Status"
label(clinical$HER2) <- "HER2 Status"
label(clinical$LN) <- "Lymph Node Status"
label(clinical$NHG) <- "Nottingham Histologic Grade (NHG)"
label(clinical$Size.mm) <- "Tumor Size (mm)"
label(clinical$TreatGroup) <- "Treatment Group"
label(clinical$InvCa.type) <- "Inv. Cancer type"
label(clinical$Age) <- "Age (years)"
label(clinical$NCN.PAM50) <- "PAM50 subtype"
label(clinical$OS_event) <- "OS Event"
label(clinical$RFi_event) <- "RFI Event"
label(clinical$OS_years) <- "Overall Survival (years)"
label(clinical$RFi_years) <- "Recurrence-Free Interval (years)"
label(clinical$Group) <- "Subgroup"

# Convert categorical variables to factors for proper formatting
clinical$ER <- factor(clinical$ER, levels = c("Negative", "Positive"))
clinical$HER2 <- factor(clinical$HER2, levels = c("Negative", "Positive"))
clinical$PR <- factor(clinical$PR, levels = c("Negative", "Positive"))
clinical$LN <- factor(clinical$LN)
clinical$Group <- factor(clinical$Group, levels = c("ER+HER2-", "TNBC", "Other"))
clinical$RFi_event <- factor(clinical$RFi_event, levels = c("0", "1"))
clinical$OS_event <- factor(clinical$OS_event, levels = c("0", "1"))
clinical$Set <- factor(clinical$Set, levels = c("Train", "Test"))

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

table_3 <- table1(~ Group + LN + NHG + Age + Size.mm + OS_event + OS_years + RFi_event + RFi_years | Set, data = clinical,
                overall=FALSE,render.continuous=my.render.cont, 
                render.categorical=my.render.cat,
topclass="Rtable1-zebra")

save_html(table_3, file = outfile_3)

webshot::webshot(
    url = outfile_3,
    file = sub("\\.html$", ".pdf", outfile_3),
    zoom = 1,
    vwidth = 1200,
    vheight = 800
)


#-------------------------------
# Median follow-up for patients with no events
#-------------------------------

# Overall Survival (OS) - patients with no OS event
median_os_followup <- median(clinical$OS_years[clinical$OS_event == "0"], na.rm = TRUE)
cat("Median OS follow-up (no event):", median_os_followup, "years\n")

# Recurrence-Free Interval (RFI) - patients with no RFI event
median_rfi_followup <- median(clinical$RFi_years[clinical$RFi_event == "0"], na.rm = TRUE)
cat("Median RFI follow-up (no event):", median_rfi_followup, "years\n")
