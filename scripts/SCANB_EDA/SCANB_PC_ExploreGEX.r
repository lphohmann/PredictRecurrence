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
if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, ggVennDiagram, data.table, gridExtra, naniar, reshape2)
#-------------------
# set/create output directories
output_path <- "./output/SCANB_EDA/"
dir.create(output_path, showWarnings = FALSE)
#-------------------
# input paths
infile_1 <- "./data/standardized/SCANB_ProjectCohort/SCANB_PC_clinical.csv"
infile_2 <- "./data/standardized/SCANB_ProjectCohort/SCANB_PC_RNAseq_expression.csv"
#-------------------
# output paths
outfile_1 <- paste0(output_path, "SCANB_PC_ExploreGEX.pdf")
#-------------------
# storing objects 
plot.list <- list() # object to store plots
txt.out <- c() # object to store text output, ggf. use capture.output()

#######################################################################
# functions
#######################################################################

# Function to calculate missing data summary
calculate_missing_data <- function(data) {
  missing_per_sample <- colSums(is.na(data))
  missing_per_gene <- rowSums(is.na(data))
  list(missing_per_sample = missing_per_sample, missing_per_gene = missing_per_gene)
}

# Function to plot missing data summary
plot_missing_data_summary <- function(missing_data, title) {
  missing_data_df <- data.frame(
    Sample = names(missing_data$missing_per_sample),
    MissingValues = missing_data$missing_per_sample
  )
  
  plot <- ggplot(missing_data_df, aes(x = Sample, y = MissingValues)) +
    geom_bar(stat = "identity") +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
    labs(title = title, x = "Samples", y = "Number of Missing Values")
  
  return(plot)
}

#######################################################################
# load data
#######################################################################

# which clin groups to run for
clin_groups <- c("All", "ER+", "ER+HER2-", "TNBC")

clinical <- read.csv(infile_1)
gex <- data.table::fread(infile_2)
gex <- as.data.frame(gex)

#######################################################################
# Analyses in general PC
#######################################################################

# Calculate and plot missing data summary for the entire dataset
missing_data <- calculate_missing_data(gex[, -1]) # exclude the gene col
plot <- plot_missing_data_summary(missing_data, "Missing Data Summary for All Samples")
plot.list <- append(plot.list, list(plot))

#sum(is.na(gex[, -1])) # no missing values but 0 

#######################################################################
# Analyses to run in all subgroups
#######################################################################
for (clin_group in clin_groups) {
  # Subgroup data
  if (clin_group == "All") {
    sub_gex <- gex
  } else {
    sub_samples <- clinical$Sample[grepl(
      clin_group,
      clinical$Group,
      fixed = TRUE # don't treat like regex, otherwise + issues
    )]
    sub_gex <- gex[, c("Gene", intersect(colnames(gex), sub_samples))]
  }

  # Calculate and plot missing data summary for the subgroup
  missing_data <- calculate_missing_data(sub_gex[, -1])
  plot <- plot_missing_data_summary(missing_data, paste0("Missing Data Summary for ", clin_group))
  plot.list <- append(plot.list, list(plot))
}

#######################################################################
# save plots to pdf
#######################################################################

# Set up the PDF output
pdf(file = outfile_1, onefile = TRUE, width = 8.27, height = 11.69)

for (i in 1:length(plot.list)) {
  print(plot.list[[i]])
  #grid.arrange(plot.list[[i]], ncol = 1, nrow = 1)
}

dev.off()
