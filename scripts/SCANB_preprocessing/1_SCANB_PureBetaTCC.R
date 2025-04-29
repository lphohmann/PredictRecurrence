#!/usr/bin/env Rscript
# Script: PureBeta TCC for selected dna methylation datasets
# Author: Lennart Hohmann
# Date: 29.04.2025
#-------------------
# empty environment
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence/")
#-------------------
# packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
    devtools
)
#install_github("StaafLab/PureBeta")
library(PureBeta)
source("./src/untils.R")
#-------------------
# set/create output directories
output_path <- "./data/processed/"
dir.create(output_path, showWarnings = FALSE)
#-------------------
# input paths
infile_1 <- "./data/processed/ERpHER2n_WGSset_unadjusted.RData"
infile_2 <- "./data/processed/TNBC_unadjusted.RData"
infile_3 <- "./data/Lennart/referenceModels_PureBeta/TNBC_reference_regressions/tnbc_reference_regressions_JS_n235.RData"
#-------------------
# output paths
outfile_tnbc <- paste0(output_path, "TNBC_unadjusted_estPurity.RData")
outfile_erp <- paste0(output_path, "ERpHER2n_WGSset_unadjusted_estPurity.RData")

#######################################################################
# load data
#######################################################################

reference_regressions <- loadRData(infile_3)
tnbc_dat <- loadRData(infile_2)
tnbc_dat[1:5,1:5]
erp_dat <- loadRData(infile_1)
erp_dat[1:5,1:5]

#######################################################################
# estimate purity
#######################################################################

dat_list <- list(tnbc_dat, erp_dat) # same other as outfile
outfile_vec <- c(outfile_tnbc, outfile_erp)

for(i in 1:length(dat_list)) {
    dat <- dat_list[[i]]
    # filter to only include common cpgs in reference and data
    cpgfilter <- rownames(dat)
    # Filter the list based on object type
    filtered_reference <- lapply(reference_regressions, function(x) {
        if (is.matrix(x) || is.data.frame(x)) {
            x[rownames(x) %in% cpgfilter, , drop = FALSE]
        } else if (is.vector(x)) {
            x[names(x) %in% cpgfilter]
        } else {
            x # leave unchanged if neither
        }
    })
    # filter to only include common cpgs in data
    dat_filt <- dat[rownames(dat) %in% names(filtered_reference$cpg.variance), ]
    # estimate purity
    dat_purity <- purity_estimation(
        filtered_reference,
        dat_filt, 
        cores = 9
    )
    save(dat_purity, file = outfile_vec[i])
}