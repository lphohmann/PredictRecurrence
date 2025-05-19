#!/usr/bin/env Rscript
# Script: PureBeta TCC for selected dna methylation datasets
# Author: Lennart Hohmann
# Date: 29.04.2025
#-------------------
# empty environment
rm(list=ls())
options(repos = c(CRAN = "https://cloud.r-project.org"))
# set working directory to the project directory
#setwd("~/PhD_Workspace/PredictRecurrence/")
#-------------------
# packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
    devtools
)
#install_github("StaafLab/PureBeta")
library(PureBeta)
source("./untils.R")
#-------------------
# set/create output directories
output_path <- "./output/"
dir.create(output_path, showWarnings = FALSE)
#-------------------
# input paths
infile_1 <- "./data/ERpHER2n_WGSset_unadjusted.RData"
infile_2 <- "./data/TNBC_unadjusted.RData"
infile_3 <- "./data/tnbc_reference_regressions_JS_n235.RData"
#-------------------
# output paths
outfile_tnbc <- paste0(output_path, "TNBC_unadjusted_estPurity.RData")
outfile_erp <- paste0(output_path, "ERpHER2n_WGSset_unadjusted_estPurity.RData")
outfile_tnbc_PBres <- paste0(output_path, "TNBC_PureBetaResult.RData")
outfile_erp_PBres <- paste0(output_path, "ERpHER2n_WGSset_PureBetaResult.RData")

#######################################################################
# load data
#######################################################################

reference_regressions <- loadRData(infile_3)
tnbc_dat <- loadRData(infile_2)
#tnbc_dat[1:5,1:5]
erp_dat <- loadRData(infile_1)
#erp_dat[1:5,1:5]
#apply(erp_dat,2,sum(is.na))

#######################################################################
# estimate purity
#######################################################################

dat_list <- list(tnbc_dat, erp_dat) # same order as outfile
purity_outfile_vec <- c(outfile_tnbc, outfile_erp)
PBres_outfile_vec <- c(outfile_tnbc_PBres, outfile_erp_PBres)

# in batches
batch_size <- 20

# test
#batch_size <- 2
#dat_list <- list(tnbc_dat, erp_dat[,1:6])

for (i in 2) { # only run for erpher2n set in this case
    dat_full <- dat_list[[i]]  
    total_samples <- ncol(dat_full)  
    sample_indices <- seq(1, total_samples, by = batch_size)  # Create start indices for batches of size batch_size

    all_purities <- list()   
    all_corrected_tumour <- list()
    all_corrected_micro <- list()

    for (b in seq_along(sample_indices)) {  # Loop over batches
        start_idx <- sample_indices[b]  # Start column index for this batch
        end_idx <- min(start_idx + batch_size - 1, total_samples)  # End column index (don't exceed total columns)
        dat <- dat_full[, start_idx:end_idx]  # Subset the data matrix for the current batch

        print(paste0("Dataset ", i, ", Batch ", b, ": Estimating purity."))

        # Run purity estimation on current batch
        dat_purity <- purity_estimation(
            reference_regressions,
            dat,
            cores = 31  # Use 31 cores
        )
        all_purities[[b]] <- dat_purity  # Store result 

        print(paste0("Dataset ", i, ", Batch ", b, ": Beta correction."))

        # Run reference-based beta correction on the batch
        dat_adjusted <- reference_based_beta_correction(
            betas_to_correct = dat,
            purities_samples_to_correct = dat_purity,
            only_certain_CpGs = FALSE,
            refitting = FALSE,
            reference_regressions = reference_regressions,
            cores = 31
        )

        all_corrected_tumour[[b]] <- dat_adjusted$Corrected_tumour
        all_corrected_micro[[b]] <- dat_adjusted$Corrected_microenvironment
    }

    # After all batches are processed, combine purity results and adjusted data
    final_purity <- do.call(cbind, all_purities)  # Combine all purity estimates column-wise
    final_corrected_tumour <- do.call(cbind, all_corrected_tumour)
    final_corrected_micro <- do.call(cbind, all_corrected_micro)

    # Save combined results to output files
    save(final_purity, file = purity_outfile_vec[i])
    final_adjusted <- list(
        Corrected_tumour = final_corrected_tumour,
        Corrected_microenvironment = final_corrected_micro
        )
    
    save(final_adjusted, file = PBres_outfile_vec[i])
    print(paste0("Dataset ", i, ": Final outputs saved."))  
}
