# Script: Getting adjusted methlyation data from PureBeta output
# Author: Lennart Hohmann
# Date: 14.05.2025
#-------------------
# empty environment
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence/")
#-------------------
# packages
#source("./scripts/src/")
#if (!require("pacman")) install.packages("pacman")
#pacman::p_load()
source("./src/utils.R")
#-------------------
# input paths
infile_1 <- "./data/raw/PureBeta_results/TNBC_PureBetaResult.RData"
infile_2 <- "./data/raw/PureBeta_results/ERpHER2n_WGSset_PureBetaResult.RData"
#-------------------
# output paths
outfile_1 <- "./data/processed/TNBC_adjusted.RData"
outfile_2 <- "./data/processed/ERpHER2n_WGSset_adjusted.RData"

#######################################################################

# load data
tnbc_res <- loadRData(infile_1)
erp_res <- loadRData(infile_2)

# save tnbc
tnbc_adj <- tnbc_res$Corrected_tumour
save(tnbc_adj, file = outfile_1)

# save erp
erp_adj <- erp_res$Corrected_tumour
save(erp_adj, file = outfile_2)