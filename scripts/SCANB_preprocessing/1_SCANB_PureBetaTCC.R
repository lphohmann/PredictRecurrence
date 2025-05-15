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
#outfile_tnbc_adj <- paste0(output_path, "TNBC_adjusted.RData")
#outfile_erp_adj <- paste0(output_path, "ERpHER2n_WGSset_adjusted.RData")
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

dat_list <- list(tnbc_dat, erp_dat) # same other as outfile
purity_outfile_vec <- c(outfile_tnbc, outfile_erp)
#betas_outfile_vec <- c(outfile_tnbc_adj, outfile_erp_adj)
PBres_outfile_vec <- c(outfile_tnbc_PBres, outfile_erp_PBres)

# for(i in 2){# 1:length(dat_list)) { # run only for erp dat
#     dat <- dat_list[[i]]

#     # test run delete later
#     #dat <- dat[,30:60]

#     print(paste0("Estimating purity and saving to file: ", purity_outfile_vec[i]))
    
#     # estimate purity
#     dat_purity <- purity_estimation(
#         reference_regressions,
#         dat,
#         cores = 31#parallel::detectCores()-3
#     )
#     # save purity
#     save(dat_purity, file = purity_outfile_vec[i])
#     print("EstPurity saved")

#     # TCC of beta values
#     print(paste0("PureBeta complete output saving to file: ", PBres_outfile_vec[i]))
#     #print(paste0("TCC of beta values and saving to file: ", betas_outfile_vec[i]))

#     dat_adjusted <- reference_based_beta_correction(
#         betas_to_correct = dat,
#         purities_samples_to_correct = dat_purity,
#         only_certain_CpGs = FALSE,
#         refitting = FALSE,
#         reference_regressions = reference_regressions,
#         cores = 31#parallel::detectCores()-3
#     )
#     # save purity
#     save(dat_adjusted, file = PBres_outfile_vec[i])
#     #save(dat_adjusted$Corrected_tumour, file = betas_outfile_vec[i])
#     print("TCC beta values saved")
# }


# in batches
batch_size <- 20

for (i in 2) { 
    dat_full <- dat_list[[i]]  
    total_samples <- ncol(dat_full)  
    sample_indices <- seq(1, total_samples, by = batch_size)  # Create start indices for batches of size batch_size

    all_purities <- list()   
    all_adjusted <- list()   

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
        all_adjusted[[b]] <- dat_adjusted  # Store result in list
    }

    # After all batches are processed, combine purity results and adjusted data
    final_purity <- do.call(cbind, all_purities)  # Combine all purity estimates column-wise
    final_adjusted <- do.call(cbind, all_adjusted)  # Combine all adjusted data column-wise

    # Save combined results to output files
    save(final_purity, file = purity_outfile_vec[i])
    save(final_adjusted, file = PBres_outfile_vec[i])

    print(paste0("Dataset ", i, ": Final outputs saved."))  
}
