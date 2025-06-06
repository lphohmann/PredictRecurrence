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
source("./src/utils.R")
#-------------------
# set/create output directories
output_path <- "./output/CoxNet_manual/"
dir.create(output_path, showWarnings = FALSE)
#-------------------
# input paths
infile_1 <- "./data/train/train_clinical.csv" 
infile_2 <- "./data/train/train_methylation_adjusted.csv"
infile_3 <- "./output/CoxNet_200k_simpleCV5/non_zero_coefs.csv"
infile_4 <- "./data/raw/EPIC_probeAnnoObj.RData"

#-------------------
# output paths
outfile_1 <- paste0(output_path, "SelectedCpGs_Annot.pdf")
#pdf(onefile = TRUE, file = outfile_1, height = 10, width = 15)
#-------------------
# storing objects 
#results <- list() # object to store results

#######################################################################
# load data
#######################################################################

clinical <- read.csv(infile_1)
coeffs <- read.csv(infile_3)

# molec dat
dat <- data.table::fread(infile_2)
dat <- as.data.frame(dat)
dat <- dat[dat$ID_REF %in% coeffs$ID_REF, ]
dim(dat)
# Move cpg names to rownames
rownames(dat) <- dat$ID_REF
dat <- dat[, -1]
dat[1:5,1:5]
# Ensure clinical data has the same samples as dat
clinical <- clinical[clinical$Sample %in% colnames(dat), ]
clinical <- clinical[match(colnames(dat), clinical$Sample), ]
identical(clinical$Sample,colnames(dat))

cpg_anno <- loadRData(infile_4)
cpg_anno <- cpg_anno[cpg_anno$illuminaID %in% coeffs$ID_REF, ]
head(cpg_anno)
names(cpg_anno)[1] <- "ID_REF"
cpg_anno <- merge(coeffs,cpg_anno, by = "ID_REF")

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
        cex.names = 1.2,     # increase x-axis label size
        cex.axis = 1.2,      # increase tick label size
        cex.lab = 1.4,       # increase axis title size
        cex.main = 1.6)      # increase main title size
    # shrink x-axis labels if needed
    # View(cpg_anno)

    # check egenes fgor promoter
    # 1. Subset cpg_anno for rows where featureClass is "promotor"
#promoter_cpgs <- cpg_anno[cpg_anno$featureClass %in% c("promoter", "proximal dn", "proximal up"), ]
promoter_cpgs <- cpg_anno[cpg_anno$featureClass %in% c("promoter"), ]
dim(promoter_cpgs)
#promoter_cpgs <- cpg_anno

# 2. Check which genes overlap with these promoter CpGs
genes_in_promoter <- promoter_cpgs$namePromOverlap
#genes_in_promoter <- genes_in_promoter[genes_in_promoter != ""]
# 3. Print the gene names (you can see a list)
print(genes_in_promoter)

# pw emnrich
# Load required libraries
library(clusterProfiler)
library(org.Hs.eg.db)
library(dplyr)

# Clean gene list
# Split on underscores and hyphens, flatten, and clean
#gene_parts <- unlist(strsplit(genes_in_promoter, split = "|"))
# gene_parts <- gene_parts[gene_parts != ""]             # remove empty strings
#gene_parts <- sapply(strsplit(genes_in_promoter, split = "[-_]"), `[`, 1)
#gene_parts <- gene_parts[!is.na(gene_parts)]
gene_parts <- sub("^[0-9]+\\|", "", gene_parts)
gene_parts <- unique(gene_parts)

# Convert gene symbols to Entrez IDs
entrez_ids <- bitr(gene_parts, fromType = "SYMBOL", 
                   toType = "ENTREZID", 
                   OrgDb = org.Hs.eg.db)

# Perform GO enrichment (Biological Process category)
go_results <- enrichGO(gene = entrez_ids$ENTREZID,
                       OrgDb = org.Hs.eg.db,
                       keyType = "ENTREZID",
                       ont = "BP",
                       pAdjustMethod = "BH",
                       qvalueCutoff = 0.05,
                       readable = TRUE)

# View top results
head(go_results)
go_results
# Plot top 10 enriched terms
barplot(go_results, showCategory = 10, title = "Top GO BP Terms")

# showing beta distributions
dev.off()



