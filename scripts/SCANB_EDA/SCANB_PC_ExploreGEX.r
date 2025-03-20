#!/usr/bin/env Rscript
# Script: Project cohort - Exploring gene expression data
# Author: Lennart Hohmann
# Date: 20.03.2025
#-------------------
# empty environment
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence/")
#-------------------
# packages
#source("./scripts/src/")
if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, ggVennDiagram, data.table, gridExtra, naniar)
#-------------------
# set/create output directories
output_path <- "./output/SCANB_EDA/"
dir.create(output_path)
#-------------------
# input paths
infile_1 <- "./data/standardized/SCANB_ProjectCohort/SCANB_PC_clinical.csv"
infile_2 <- "./data/standardized/SCANB_ProjectCohort/SCANB_PC_RNAseq_expression.csv"
#-------------------
# output paths
outfile_1 <- paste0(output_path,"SCANB_PC_ExploreGEX.pdf")
#-------------------
# storing objects 
#plot.list <- list() # object to store plots
txt.out <- c() # object to store text output, ggf. use capture.output()

#######################################################################
# load data
#######################################################################
# which clin groups to run for
clin_groups <- c("All", "ER+", "ER+HER2-", "TNBC")
#clin_group <- "ER+HER2-" # All ER+HER2-

clinical <- read.csv(infile_1)
gex <- data.table::fread(infile_2)
gex <- as.data.frame(gex)

# for(clin_group in clin_groups) {}
#clin_group = "All" #"ER+HER2-"
clin_group = "ER+HER2-" #"ER+HER2-" tests

# Subgroup data
if (clin_group == "All") {
    sub_gex <- gex
} else {
    sub_gex <- gex[, c(
        "Gene",
        intersect(colnames(gex), clinical$Sample[clinical$Group == clin_group])
    )]
}

#######################################################################
# f
#######################################################################

#######################################################################
# Missing data
#######################################################################

# Plot Nullity Matrix
sub_clinical$TreatGroup[sub_clinical$TreatGroup == "Missing"] <- NA
plot_4 <- vis_miss(sub_clinical[c("ER", "HER2", "PR", "Age", "PR", "LN", "NHG", "Size.mm", "TreatGroup", "RFi_event", "RFi_years", "OS_event", "OS_years", "DRFi_event", "DRFi_years")]) +
    scale_fill_manual(values = c("black", "red")) +
    theme(
    axis.text.x = element_text(size = 14, angle = 45, hjust = 0.5),
    axis.title = element_text(size = 16),
    axis.text.y = element_text(size = 14))  # Angle and adjust )

#######################################################################
# save plots to pdf
#######################################################################

# Set up the PDF output
pdf(file = outfile_1, onefile = TRUE, width = 8.27, height = 11.69)

grid.arrange(plot_1, plot_2, ncol = 2, nrow = 3)
grid.arrange(plot_3, plot_4, ncol = 1, nrow = 2)

dev.off()