#!/usr/bin/env Rscript
# Script: tables in subsets SCAN-B FU methlyation cohrot
# Author: Lennart Hohmann
# Date: 11.03.2025
#-------------------
# empty environment
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence/")
#-------------------
# packages
#source("./scripts/src/")
if (!require("pacman")) install.packages("pacman")
pacman::p_load(table1,htmltools,finalfit,Publish,htmlTable)
#-------------------
# set/create output directories
output_path <- "./output/SCANB_EDA/"
dir.create(output_path)
#-------------------
# input paths
infile_1 <- "./data/standardized/SCANB_FullFU/SCANB_sample_modalities.csv"
infile_2 <- "./data/standardized/SCANB_FullFU/SCANB_clinical.csv"
#-------------------
# which clin group to run for
clin_group <- "All" # "ER+HER2-"
#clin_group <- "ER+HER2-"

#-------------------
# output paths
outfile_1 <- paste0(output_path, "SCANB_FUmethyl_SubgroupTable_", clin_group, ".html")
outfile_2 <- paste0(output_path,"SCANB_FUmethyl_TreatmentTable_",clin_group,".html")

#-------------------
# storing objects 
table_list <- list()

#######################################################################
# load data
#######################################################################

sample_modalities <- read.csv(infile_1)
clinical <- read.csv(infile_2)
clinical$NHG <- as.character(clinical$NHG)
sample_modalities <- sample_modalities[sample_modalities$DNAmethylation==1,]
clinical <- clinical[clinical$Sample %in% sample_modalities$Sample, ]

#str(clinical)
# Subgroup data
if (clin_group == "All") {
  sub_sample_modalities <- sample_modalities
  sub_clinical <- clinical
  } else {
    sub_clinical <- subset(clinical, Group == clin_group)
    sub_sample_modalities <- subset(sample_modalities, Sample %in% sub_clinical$Sample)
}
#names(sub_sample_modalities)

# sub cohort
#sub_clinical$Subset <- sub_sample_modalities$DNAmethylation[match(sub_clinical$Sample,sub_sample_modalities$Sample)]
sub_clinical$TreatGroup_ERpHER2n <- ifelse(!(sub_clinical$TreatGroup %in% c("ChemoEndo","Endo","None")),"Other",sub_clinical$TreatGroup)

#######################################################################
# make Table 1
#######################################################################

# Assign labels for better readability
label(sub_clinical$PR) <- "PR Status"
label(sub_clinical$LN) <- "Lymph Node Status"
label(sub_clinical$NHG) <- "Nottingham Histologic Grade (NHG)"
label(sub_clinical$Size.mm) <- "Tumor Size (mm)"
label(sub_clinical$TreatGroup) <- "Treatment Group"
label(sub_clinical$Age) <- "Age (years)"
label(sub_clinical$DRFi_event) <- "Distant Recurrence Event"
label(sub_clinical$OS_event) <- "Overall Survival Event"
label(sub_clinical$RFi_event) <- "Recurrence-Free Event"
label(sub_clinical$OS_years) <- "Overall Survival (years)"
label(sub_clinical$RFi_years) <- "Recurrence-Free Interval (years)"
label(sub_clinical$DRFi_years) <- "Distant Recurrence-Free Interval (years)"
label(sub_clinical$TreatGroup_ERpHER2n) <- "Treatment regimen"
label(sub_clinical$Group) <- "Subgroup"

# Convert categorical variables to factors for proper formatting
sub_clinical$PR <- factor(sub_clinical$PR, levels = c("Negative", "Positive"))
sub_clinical$LN <- factor(sub_clinical$LN)
sub_clinical$TreatGroup_ERpHER2n <- factor(sub_clinical$TreatGroup_ERpHER2n)
sub_clinical$Group <- factor(sub_clinical$Group, levels = c("ER+HER2-", "ER+HER2+", "ER-HER2+", "TNBC", "Other"))

# Apply custom renders
my.render.cont <- function(x) {
    with(stats.apply.rounding(stats.default(x), digits=2), c("",
        "Mean (SD)"=sprintf("%s (&plusmn; %s)", MEAN, SD)))
}
my.render.cat <- function(x) {
    c("", sapply(stats.default(x), function(y) with(y,
        sprintf("%d (%0.0f %%)", FREQ, PCT))))
}
table_1 <- table1(~ Age + Size.mm + PR + LN + NHG | Group, data = sub_clinical,
                overall=c(left="Total"),render.continuous=my.render.cont, 
                render.categorical=my.render.cat)

table_list <- append(as.data.frame(table_1), table_list)

table_2 <- table1(~ Age + Size.mm + PR + LN + NHG | TreatGroup, 
data = sub_clinical,
                overall=c(left="Total"),render.continuous=my.render.cont, 
                render.categorical=my.render.cat)

table_list <- append(as.data.frame(table_2), table_list)

#######################################################################
# save
#######################################################################
save_html(table_1, file = outfile_1)
save_html(table_2, file = outfile_2)

# Loop through the list and save each table as sheet in excel file
# for (table in table_list) {
#   # Convert each table to HTML and append it to the file
#   cat("<h3>Table</h3>", file = html_file)  # Optional heading for each table
#   cat(htmlTable::htmlTable(table), file = html_file)
#   cat("<br><br>", file = html_file)  # Space between tables
# }

