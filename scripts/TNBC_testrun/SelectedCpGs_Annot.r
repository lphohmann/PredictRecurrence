#!/usr/bin/env Rscript
# Script: annotating selected CpG sites
# Author: Lennart Hohmann
# Date: 28.04.2025
#-------------------
# empty environment
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence/")
#-------------------
# packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  data.table, prodlim,
  survival, Publish
)
source("./src/untils.R")
#-------------------
# set/create output directories
output_path <- "./output/CoxNet_manual/"
dir.create(output_path, showWarnings = FALSE)
#-------------------
# input paths
infile_1 <- "./data/raw/tnbc_anno.csv"
infile_2 <- "./data/raw/tnbc235.csv"
infile_3 <- "./output/CoxNet_manual/manual_non_zero_coefs.csv"
infile_4 <- "./data/raw/EPIC_probeAnnoObj.RData"
infile_5 <- "./data/raw/TCGA_TNBC_MergedAnnotations.RData"
infile_6 <- "./data/raw/TCGA_TNBC_betaAdj.RData"

#-------------------
# output paths
outfile_1 <- paste0(output_path, "SelectedCpGs_Annot.pdf")
pdf(onefile = TRUE, file = outfile_1, height = 10, width = 15)
#-------------------
# storing objects 
#results <- list() # object to store results

#######################################################################
# load data
#######################################################################

clinical <- read.csv(infile_1)
coeffs <- read.csv(infile_3)
names(coeffs)[1] <- "CpG_ID"

# molec dat
dat <- data.table::fread(infile_2)
dat <- as.data.frame(dat)
dat <- dat[dat$V1 %in% coeffs$CpG_ID, ]
dim(dat)
# Move cpg names to rownames
rownames(dat) <- dat$V1
dat <- dat[, -1]
dat[1:5,1:5]
# Ensure clinical data has the same samples as dat
clinical <- clinical[clinical$PD_ID %in% colnames(dat), ]
# Ensure the samples in clinical data are in the same order as in dat
clinical <- clinical[match(colnames(dat), clinical$PD_ID), ]
identical(clinical$PD_ID,colnames(dat))

cpg_anno <- loadRData(infile_4)
cpg_anno <- cpg_anno[cpg_anno$illuminaID %in% coeffs$CpG_ID, ]
head(cpg_anno)
names(cpg_anno)[1] <- "CpG_ID"
cpg_anno <- merge(coeffs,cpg_anno, by = "CpG_ID")

#######################################################################
# load data
#######################################################################
library(pheatmap)
cpg_anno[1:5,1:2]
dat[1:5, 1:5]


# 1. Your data is already in 'dat' (rows = CpGs, columns = patients)

# 2. First, transpose the data so CpGs become columns and patients become rows
dat_t <- t(dat)
#head(dat_t)
# 3. Compute the correlation matrix
cor_matrix <- cor(dat_t, method = "spearman")  # or pearson

# 4. Visualize the correlation matrix
# pheatmap(cor_matrix,
#     clustering_distance_rows = "correlation",
#     clustering_distance_cols = "correlation",
#     color = colorRampPalette(c("blue", "white", "red"))(100),
#     main = "CpG Correlation Matrix"
# )

# Generate and save the heatmap plot directly
pheatmap(cor_matrix, 
         clustering_distance_rows = "correlation",
         clustering_distance_cols = "correlation",
         color = colorRampPalette(c("blue", "white", "red"))(100),
         main = "CpG Correlation Matrix",
         filename = paste0(output_path, "CpG_Correlation_Matrix.png"))  # You can use .pdf, .png, or .jpg



# now barplot location

names(cpg_anno)
str(cpg_anno)
cpg_anno$featureClass

# 1. Count how many CpGs per featureClass
feature_counts <- table(cpg_anno$featureClass)

# 2. Barplot
barplot(feature_counts,
        main = "Number of CpGs per Feature Class",
        xlab = "Feature Class",
        ylab = "Count",
        col = rainbow(length(feature_counts)),
        las = 2,             # rotate x labels vertical
        cex.names = 0.8)     # shrink x-axis labels if needed
#View(cpg_anno)

# check egenes fgor promoter
# 1. Subset cpg_anno for rows where featureClass is "promotor"
promoter_cpgs <- cpg_anno[cpg_anno$featureClass == "promoter", ]

# 2. Check which genes overlap with these promoter CpGs
genes_in_promoter <- promoter_cpgs$nameUCSCknownGeneOverlap

# 3. Print the gene names (you can see a list)
print(genes_in_promoter) #"COX6A2"            "CYP1B1_CYP1B1-AS1" ""  





# shoing beta distributions


dev.off()



