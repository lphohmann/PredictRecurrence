#!/usr/bin/env Rscript
# Script: Standardizing clinical train and test data for SCANB with MICE imputation
# Author: Lennart Hohmann (modified with MICE)
# Date: 21.05.2025
#----------------------------------------------------------------------
# empty environment
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence/")
#----------------------------------------------------------------------
# packages
if (!require("mice")) install.packages("mice")
library(mice)
library(data.table)
library(ggplot2)
library(gridExtra)
source("./src/utils.R")
#----------------------------------------------------------------------
# input paths
infile.0 <- "./data/set_definitions/train_ids.csv"
infile.1 <- "./data/raw/Summarized_SCAN_B_rel4_NPJbreastCancer_with_ExternalReview_Bosch_data.RData"
infile.2 <- "./data/set_definitions/test_ids.csv"
#----------------------------------------------------------------------
# output paths
output.path <- "./data/train/"
dir.create(output.path, showWarnings = FALSE)
outfile.0 <- paste0(output.path, "test_clinical.csv")
outfile.1 <- paste0(output.path, "train_clinical.csv")
outfile.2 <- paste0(output.path, "mice_model.rds")  # Save MICE model for test data
outfile.3 <- paste0(output.path, "imputation_diagnostics.pdf")  # Diagnostic plots
dir.create("./data/train/train_subcohorts/", showWarnings = FALSE)

#######################################################################
# Load and prepare clinical data
#######################################################################
test.ids <- read.table(infile.2)[[1]]
train.ids <- read.table(infile.0)[[1]]
clin.dat <- loadRData(infile.1)
clin.dat <- clin.dat[clin.dat$Follow.up.cohort == TRUE,]
clin.dat <- clin.dat[clin.dat$Sample %in% c(train.ids,test.ids),]
clin.dat$LN <- ifelse(clin.dat$LN > 0, "N+", "N0")
clin.dat$PR[clin.dat$PR==""] <- NA

# select clinical variables to include
clin.dat <- clin.dat[c("Sample","ER","PR","HER2","LN",
                       "NHG","Size.mm",#"TreatGroup",
                       "Age","NCN.PAM50",
                       "OS_days","OS_event",
                       "RFi_days","RFi_event")]

# convert outcome to years
clin.dat$OS_years <- clin.dat$OS_days / 365
clin.dat$RFi_years <- clin.dat$RFi_days / 365
clin.dat$OS_days <- NULL
clin.dat$RFi_days <- NULL

# split treatment
#clin.dat$TreatGroup[clin.dat$TreatGroup == ""] <- NA
#clin.dat$TreatGroup[is.na(clin.dat$TreatGroup)] <- "Missing"

# create column for clinical groups (ER and HER2 status)
clin.dat$Group <- ifelse(
    clin.dat$ER == "Positive" & clin.dat$HER2 == "Negative",
    "ER+HER2-",
    ifelse(
        clin.dat$HER2 == "Negative" & clin.dat$ER == "Negative" & clin.dat$PR == "Negative", 
        "TNBC",
        "Other"))

clin.dat[clin.dat == ""] <- NA

# Convert to appropriate types
clin.dat$LN <- as.factor(clin.dat$LN)
clin.dat$ER <- as.factor(clin.dat$ER)
clin.dat$HER2 <- as.factor(clin.dat$HER2)
clin.dat$PR <- as.factor(clin.dat$PR)
clin.dat$NHG <- factor(clin.dat$NHG, levels = c("1", "2", "3"), ordered = TRUE)
clin.dat$Age <- as.numeric(clin.dat$Age)
clin.dat$Size.mm <- as.numeric(clin.dat$Size.mm)
#clin.dat$TreatGroup <- as.factor(clin.dat$TreatGroup)
clin.dat$NCN.PAM50 <- as.factor(clin.dat$NCN.PAM50)
clin.dat$Group <- as.factor(clin.dat$Group)
table(clin.dat$Group)

#######################################################################
# Split train and test
#######################################################################
clin.dat.train <- clin.dat[clin.dat$Sample %in% train.ids,]
clin.dat.test <- clin.dat[clin.dat$Sample %in% test.ids,]

cat("\n=== Missing Data Pattern (Training Set) ===\n")
# Check missing data pattern
cat("Number of missing values per variable:\n")
print(colSums(is.na(clin.dat.train)))
cat("\nPercentage of missing data per variable:\n")
print(round(colMeans(is.na(clin.dat.train)) * 100, 2))
print(round(colMeans(is.na(clin.dat.test)) * 100, 2))

#######################################################################
# MICE Imputation Configuration
#######################################################################

# Variables to EXCLUDE from imputation (outcomes and ID)
vars_to_exclude <- c("Sample", "Group", "OS_years", "OS_event", "RFi_years", "RFi_event")

# Create predictor matrix
# This controls which variables are used to predict each variable
init <- mice(clin.dat.train, maxit = 0)  # Initialize to get default settings
pred_matrix <- init$predictorMatrix

# Set rows for variables we don't want to impute to 0
pred_matrix[vars_to_exclude, ] <- 0
# Set columns for variables we don't want to use as predictors to 0
pred_matrix[, vars_to_exclude] <- 0

cat("\n=== Predictor Matrix ===\n")
cat("Variables used to predict each imputed variable:\n")
print(pred_matrix)

# Method specification
# The mice package automatically chooses methods based on variable type:
# - "pmm" (predictive mean matching) for numeric variables
# - "logreg" for binary factors
# - "polyreg" for unordered categorical (>2 levels)
# - "polr" for ordered categorical variables
method <- init$method
method[vars_to_exclude] <- ""  # Don't impute these variables

cat("\n=== Imputation Methods ===\n")
cat("Method used for each variable:\n")
print(method[method != ""]) # when no missing data then dropeed (not needed to impute)

#######################################################################
# Run MICE on Training Data
#######################################################################

cat("\n=== Running MICE Imputation (Training Data) ===\n")
cat("Parameters:\n")
cat("- m = 5 (number of imputed datasets)\n")
cat("- maxit = 10 (number of iterations per imputation)\n")
cat("- seed = 123 (for reproducibility)\n\n")

# m = number of imputed datasets (typically 5-10)
# maxit = number of iterations (typically 5-20, increase if not converged)
# seed = for reproducibility
# printFlag = show progress

set.seed(123)
mice_model_train <- mice(
    clin.dat.train,
    m = 5,                    # Create 5 imputed datasets
    maxit = 10,               # 10 iterations per dataset
    predictorMatrix = pred_matrix,
    method = method,
    seed = 123,
    printFlag = TRUE
)

cat("\n=== MICE Imputation Complete ===\n")

#######################################################################
# Convergence Diagnostics
#######################################################################

cat("\n=== Checking Convergence ===\n")
cat("Creating diagnostic plots...\n")

pdf(outfile.3, width = 12, height = 8)

# Plot 1: Trace plots to check convergence
# Each line represents one imputed dataset
# Lines should be intermingled and show no trend = good convergence
plot(mice_model_train, layout = c(3, 3))
#title(main = "Convergence Trace Plots - Training Data", outer = TRUE, line = -1)

# Plot 2: Density plots comparing observed vs imputed values
# Imputed values (red) should have similar distribution to observed (blue)
densityplot(mice_model_train)

# Plot 3: Stripplot showing distribution of observed and imputed values
stripplot(mice_model_train, pch = 20, cex = 1.2)

dev.off()
cat("Diagnostic plots saved to:", outfile.3, "\n")

#######################################################################
# Extract Completed Dataset (Training)
#######################################################################

# Option 1: Use the first imputed dataset
# This is common for building a single model
clin.dat.train.imputed <- complete(mice_model_train, 1) #

# Verify no missing data in variables that were imputed
cat("\n=== Post-Imputation Check (Training) ===\n")
cat("Missing values after imputation:\n")
print(colSums(is.na(clin.dat.train.imputed)))

#######################################################################
# Simple imputation stability check - ONE plot per variable
#######################################################################

cat("\n=== Checking Imputation Stability Across 5 Datasets ===\n")

# Get all 5 imputed datasets
imp1 <- complete(mice_model_train, 1)
imp2 <- complete(mice_model_train, 2)
imp3 <- complete(mice_model_train, 3)
imp4 <- complete(mice_model_train, 4)
imp5 <- complete(mice_model_train, 5)

# Identify which observations were originally missing for each variable
originally_missing <- lapply(clin.dat.train, is.na)

# Variables that were imputed
imputed_vars <- names(method[method != ""])

cat("Variables being checked:", paste(imputed_vars, collapse = ", "), "\n\n")

# Create PDF with one plot per variable
pdf("./data/train/imputation_stability_check.pdf", width = 10, height = 7)

#----------------------------------------------------------------------
# CONTINUOUS VARIABLES - Boxplot
#----------------------------------------------------------------------

continuous_vars <- c("Size.mm")
continuous_vars <- continuous_vars[continuous_vars %in% imputed_vars]

for(var in continuous_vars) {
  missing_idx <- originally_missing[[var]]
  n_missing <- sum(missing_idx)
  
  if(n_missing > 0) {
    # Boxplot comparing 5 imputations
    boxplot(
      imp1[[var]][missing_idx],
      imp2[[var]][missing_idx],
      imp3[[var]][missing_idx],
      imp4[[var]][missing_idx],
      imp5[[var]][missing_idx],
      names = c("Imp1", "Imp2", "Imp3", "Imp4", "Imp5"),
      main = paste0(var, ": Stability Across 5 Imputations\n(n=", n_missing, " imputed values)"),
      ylab = var,
      col = "lightblue",
      las = 1
    )
    
    # Add means
    means <- c(
      mean(imp1[[var]][missing_idx]),
      mean(imp2[[var]][missing_idx]),
      mean(imp3[[var]][missing_idx]),
      mean(imp4[[var]][missing_idx]),
      mean(imp5[[var]][missing_idx])
    )
    points(1:5, means, pch = 18, col = "red", cex = 1.5)
    legend("topright", legend = "Mean", pch = 18, col = "red", pt.cex = 1.5)
    
    cat(var, "- Mean values across imputations:", round(means, 2), "\n")
    cat("     CV (coefficient of variation):", round(sd(means)/mean(means)*100, 2), "%\n")
  }
}

#----------------------------------------------------------------------
# CATEGORICAL VARIABLES - Overall concordance plot
#----------------------------------------------------------------------

categorical_vars <- c("ER", "PR", "HER2", "LN", "NHG", "NCN.PAM50")
categorical_vars <- categorical_vars[categorical_vars %in% imputed_vars]

for(var in categorical_vars) {
  missing_idx <- originally_missing[[var]]
  n_missing <- sum(missing_idx)
  
  if(n_missing > 0) {
    # Get all 5 imputations for missing cases
    imputed_categories <- data.frame(
      Imp1 = imp1[[var]][missing_idx],
      Imp2 = imp2[[var]][missing_idx],
      Imp3 = imp3[[var]][missing_idx],
      Imp4 = imp4[[var]][missing_idx],
      Imp5 = imp5[[var]][missing_idx]
    )
    
    # For each case, count how many of the 5 imputations agree
    agreement_count <- apply(imputed_categories, 1, function(row) {
      max(table(row))
    })
    
    # Simple table: just count 5/5, 4/5, 3/5
    agreement_table <- table(factor(agreement_count, levels = 5:3))
    names(agreement_table) <- c("5/5", "4/5", "3/5")
    
    # Simple bar plot
    bp <- barplot(
      agreement_table,
      main = paste0(var, ": Agreement Across 5 Imputations\n(n=", n_missing, " imputed values)"),
      ylab = "Number of cases",
      col = c("darkgreen", "gold", "coral"),
      ylim = c(0, max(agreement_table) * 1.2),
      las = 1
    )
    
    # Add counts on bars
    text(bp, agreement_table + max(agreement_table) * 0.05, 
         labels = agreement_table, 
         cex = 1.5, font = 2)
    
    # Summary
    perfect_pct <- agreement_table["5/5"] / n_missing * 100
    cat(sprintf("%s - Perfect agreement: %.1f%%\n", var, perfect_pct))
  }
}

dev.off()

cat("\n=== Stability Check Complete ===\n")
cat("Results saved to: ./data/train/imputation_stability_check.pdf\n")
cat("\nINTERPRETATION:\n")
cat("  Continuous: Boxplots should overlap (similar distributions)\n")
cat("  Categorical: Confusion matrices should be dark green on diagonal (high agreement)\n")
cat("               Agreement >80% is excellent, >70% is good\n")

#######################################################################
# Apply MICE Model to Test Data
#######################################################################

cat("\n=== Imputing Test Data ===\n")
cat("Using training data distributions to impute test data...\n")

# Use mice.mids to impute test data based on training model
# This ensures test data is imputed using the same relationships learned from training
mice_model_test <- mice.mids(
    mice_model_train,
    newdata = clin.dat.test,
    maxit = 5,  # Fewer iterations needed for applying existing model
    printFlag = TRUE
)

# Extract completed test dataset (first imputation)
clin.dat.test.imputed <- complete(mice_model_test, 1)

cat("\n=== Post-Imputation Check (Test) ===\n")
cat("Missing values after imputation:\n")
print(colSums(is.na(clin.dat.test.imputed)))

#######################################################################
# Save Results
#######################################################################

cat("\n=== Saving Results ===\n")

# Save imputed datasets
write.csv(clin.dat.train.imputed, outfile.1, row.names = FALSE)
cat("Training data saved to:", outfile.1, "\n")

write.csv(clin.dat.test.imputed, outfile.0, row.names = FALSE)
cat("Test data saved to:", outfile.0, "\n")

# Save MICE models for reproducibility/documentation
saveRDS(mice_model_train, outfile.2)
cat("MICE model saved to:", outfile.2, "\n")

#######################################################################
# Summary Statistics
#######################################################################

cat("\n=== Imputation Summary ===\n")
cat("\nTraining set dimensions:", dim(clin.dat.train.imputed), "\n")
cat("Test set dimensions:", dim(clin.dat.test.imputed), "\n")

cat("\n=== Script Complete ===\n")
cat("1. Review diagnostic plots in", outfile.3, "\n")
