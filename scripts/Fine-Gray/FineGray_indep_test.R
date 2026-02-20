#!/usr/bin/env Rscript

################################################################################
# Fine-Gray Competing Risks with Dual Final Models
# Author: Lennart Hohmann

################################################################################
# LIBRARIES
################################################################################

library(readr)        # Fast CSV reading
library(dplyr)        # Data manipulation
library(survival)     # Survival analysis (Surv objects, concordance)
library(glmnet)       # Elastic net Cox regression
library(cmprsk)
library(caret)        # Cross-validation fold creation
library(data.table)   # Fast data reading with fread
library(riskRegression) # Score() function for competing risks metrics
library(prodlim) #prodlim::Hist()
library(devtools)
library(fastcmprsk)

setwd("~/PhD_Workspace/PredictRecurrence/")
source("./src/finegray_functions.R")

################################################################################
# INPUT SETTINGS
################################################################################

COHORT_NAME <- "ERpHER2n"
OUTPUT_BASE_DIR <- "./output/FineGray/ERpHER2n/DualMode/Indep/"

# Input files
INFILE_METHYLATION <- "./data/train/train_methylation_unadjusted.csv"
COHORT_TRAIN_IDS_PATHS <- list(
  TNBC = "./data/train/train_subcohorts/TNBC_train_ids.csv",
  ERpHER2n = "./data/train/train_subcohorts/ERpHER2n_train_ids.csv",
  All = "./data/train/train_subcohorts/All_train_ids.csv"
)
INFILE_CLINICAL <- "./data/train/train_clinical.csv"

INFILE_DEPLOY <- "./output/FineGray/ERpHER2n/DualMode/Unadjusted/deployment_object.RData"

INFILE_FINAL_RESULTS <- "./output/FineGray/ERpHER2n/DualMode/Unadjusted/final_fg_results.RData"

# Administrative censoring
ADMIN_CENSORING_CUTOFF <- if (COHORT_NAME == "TNBC") 5.01 else NULL

################################################################################
# load training data
################################################################################

# Load sample IDs and data
train_ids <- read_csv(
  COHORT_TRAIN_IDS_PATHS[[COHORT_NAME]], 
  col_names = FALSE, 
  show_col_types = FALSE
)[[1]]

data_list <- load_training_data(train_ids, INFILE_METHYLATION, INFILE_CLINICAL)
beta_matrix <- data_list$beta_matrix
clinical_data <- data_list$clinical_data

################################################################################
# load training results
################################################################################

load(INFILE_DEPLOY)
load(INFILE_FINAL_RESULTS)
#str(deployment_object)
#str(final_results)
final_results$variable_importance$fgr_COMBINED
#vimp_fg

################################################################################
# methylation matrix 
################################################################################

rs_cpgs <- deployment_object$mrs_params$required_cpgs
beta_cpgs <- beta_matrix[, rs_cpgs, drop = FALSE]
# Convert beta values to M-values
mvals_cpgs <- beta_to_m(beta_cpgs, beta_threshold = 0.001)

cat(sprintf("CV feature matrix: %d samples × %d CpGs\n", 
            nrow(mvals_cpgs), ncol(mvals_cpgs)))

################################################################################
# clinical matrix 
################################################################################

# Apply administrative censoring if needed
if (!is.null(ADMIN_CENSORING_CUTOFF)) {
  clinical_data <- apply_admin_censoring(
    clinical_data, "RFi_years", "RFi_event", ADMIN_CENSORING_CUTOFF
  )
  clinical_data <- apply_admin_censoring(
    clinical_data, "OS_years", "OS_event", ADMIN_CENSORING_CUTOFF
  )
}

CLIN_CONT <- deployment_object$clinical_params$continuous_vars
CLIN_CATEGORICAL <- deployment_object$clinical_params$categorical_vars

clinical_data$LN <- gsub("N\\+", "Np", clinical_data$LN)
clin <- clinical_data[c(CLIN_CONT, CLIN_CATEGORICAL)]
clin <- clin[rownames(mvals_cpgs), , drop = FALSE]
encoded_result <- onehot_encode_clinical(clin, CLIN_CATEGORICAL)




################################################################################
# load test data
################################################################################
load("./data/train/TestSetSamples_Singlets_betaMatrix_V1_V2_reduced_717459commonCpGs_n2601.RData")

clinical_test <- read.csv("./data/train/test_clinical.csv", stringsAsFactors = FALSE)

colnames(beta.matrix.singlet) <- sub("\\..*$", "", colnames(beta.matrix.singlet))
beta.matrix.singlet <- t(beta.matrix.singlet)
beta_test_cpgs <- as.data.frame(beta.matrix.singlet)
beta_test_cpgs <- beta.matrix.singlet[rownames(beta_test_cpgs) %in% clinical_test$Sample,rs_cpgs]
clinical_test <- clinical_test[clinical_test$Sample %in% rownames(beta_test_cpgs),]
clinical_test <- clinical_test[clinical_test$Group == "ER+HER2-",]
beta_test_cpgs <- beta_test_cpgs[clinical_test$Sample,]
test_data <- cbind(clinical_test,beta_test_cpgs)

################################################################################
# apply risk prediction
################################################################################


#
deployment_object <- list(
  
  # For MRS calculation
  mrs_params = list(
    required_cpgs = pFG_selected_cpgs,
    cpg_coefs = pFG_cpg_coefs,
    cpg_scaling = pFG_cpg_scaling,
    mrs_scaling = mrs_scaling
  ),
  
  # For clinical variable preparation
  clinical_params = list(
    continuous_vars = CLIN_CONT,
    categorical_vars = CLIN_CATEGORICAL,
    encoded_cols = encoded_result$encoded_cols,
    clinical_scaling = clin_scaling
  ),
  
  # The three models
  models = list(
    fgr_MRS = fgr_MRS,
    fgr_CLIN = fgr_CLIN,
    fgr_COMBINED = fgr_COMBINED
  )
)
