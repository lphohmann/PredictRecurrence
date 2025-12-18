#!/usr/bin/env Rscript
# Script: Standardizing clinical train and test data for SCANB; define sample subsets
# Author: Lennart Hohmann
# Date: 21.05.2025
#----------------------------------------------------------------------
# empty environment
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence/")
#----------------------------------------------------------------------
# packages
# Install if needed
if (!require("mice")) install.packages("mice")
library(mice)
library(data.table)
library(ggplot2)
library(gridExtra)  # for arranging multiple plots

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
dir.create("./data/train/train_subcohorts/", showWarnings = FALSE)

#######################################################################
# clinical data
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
     "NHG","Size.mm","TreatGroup","InvCa.type",
     "Age","NCN.PAM50",
     "OS_days","OS_event",
     "RFi_days","RFi_event")]

# convert outcome to years
clin.dat$OS_years <- clin.dat$OS_days / 365
clin.dat$RFi_years <- clin.dat$RFi_days / 365
clin.dat$OS_days <- NULL
clin.dat$RFi_days <- NULL

# split treatment
clin.dat$TreatGroup[clin.dat$TreatGroup == ""] <- NA
clin.dat$TreatGroup[is.na(clin.dat$TreatGroup)] <- "Missing"

# create column for clinical groups (ER and HER2 status)
clin.dat$Group <- ifelse(
    clin.dat$ER == "Positive" & clin.dat$HER2 == "Negative",
    "ER+HER2-",
        ifelse(
            clin.dat$HER2 == "Negative" & clin.dat$ER == "Negative" & clin.dat$PR == "Negative", 
            "TNBC",
                "Other"))

clin.dat[clin.dat == ""] <- NA

# Categorical variables - convert to factors
clin.dat$LN <- as.factor(clin.dat$LN)
table(clin.dat$LN)
clin.dat$ER <- as.factor(clin.dat$ER)
clin.dat$HER2 <- as.factor(clin.dat$HER2)
clin.dat$PR <- as.factor(clin.dat$PR)

# Ordered categorical - convert to ordered factors
clin.dat$NHG <- factor(clin.dat$NHG, 
                             levels = c("1", "2", "3"),  # or c(1, 2, 3) if numeric
                             ordered = TRUE)

# Continuous - ensure numeric (probably already are)
clin.dat$Age <- as.numeric(clin.dat$Age)
clin.dat$Size.mm <- as.numeric(clin.dat$Size.mm)


#######################################################################
# split
#######################################################################


clin.dat.train <- clin.dat[clin.dat$Sample %in% train.ids,]
clin.dat.test <- clin.dat[clin.dat$Sample %in% test.ids,]

#######################################################################
# impute missing data for Age, Size.mm NHG LN
#######################################################################

print(colSums(is.na(clin.dat.train[c("Age", "Size.mm", "NHG", "LN")])))
table(clin.dat.train$NCN.PAM50, useNA = "always")

# Save original data with missing values for comparison
clin.dat.train_original <- clin.dat.train  # before imputation
clin.dat.test_original <- clin.dat.test  # before imputation

clin.dat.train <- impute_clinical_data(
  clin.dat.train,
  vars_to_impute = c("Age", "Size.mm", "NHG", "LN"),
  complete_vars = c("ER", "HER2", "PR", "NCN.PAM50"),
  binary_vars = c("LN"),
  ordered_vars = c("NHG"))

clin.dat.test <- impute_clinical_data(
  clin.dat.test,
  vars_to_impute = c("Age", "Size.mm", "NHG", "LN"),
  complete_vars = c("ER", "HER2", "PR", "NCN.PAM50"),
  binary_vars = c("LN"),
  ordered_vars = c("NHG"))

#######################################################################
# save
#######################################################################
head(clin.dat.train)
colnames(clin.dat.train)
fwrite(clin.dat.train, file=outfile.1, na = "NA")
fwrite(clin.dat.test, file=outfile.0, na = "NA")

#######################################################################
# VISUALIZATION AND ERROR CHECKING AFTER IMPUTATION
#######################################################################
ds_pairs <- list(
  train = list(clin.dat.train, clin.dat.train_original),
  test = list(clin.dat.test, clin.dat.test_original))

for (name in names(ds_pairs)) {
    print(name)
    pair <- ds_pairs[[name]]
    ds <- pair[[1]]
    original <- pair[[2]]

    # After imputation, compare distributions
    # Create indicator for which values were imputed
    ds$Size.mm_imputed <- is.na(original$Size.mm)
    ds$NHG_imputed <- is.na(original$NHG)
    ds$LN_imputed <- is.na(original$LN)

    #----------------------------------------------------------------------
    # 1. Compare distributions: Observed vs Imputed
    #----------------------------------------------------------------------

    # Size.mm (continuous)
    p1 <- ggplot(ds, aes(x = Size.mm, fill = Size.mm_imputed)) +
      geom_histogram(position = "identity", alpha = 0.6, bins = 30) +
      scale_fill_manual(values = c("FALSE" = "blue", "TRUE" = "red"),
                        labels = c("Observed", "Imputed")) +
      labs(title = paste0(name," - Size.mm: Observed vs Imputed"),
          x = "Tumor Size (mm)", y = "Count", fill = "") +
      theme_minimal()

    # NHG (categorical)
    nhg_summary <- as.data.frame(table(ds$NHG, ds$NHG_imputed))
    colnames(nhg_summary) <- c("NHG", "Imputed", "Count")

    nhg_summary$Imputed <- factor(
      nhg_summary$Imputed,
      levels = c(FALSE, TRUE),
      labels = c("Observed", "Imputed")
    )

    p2 <- ggplot(nhg_summary, aes(x = NHG, y = Count, fill = Imputed)) +
      geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
      scale_fill_manual(values = c("Observed" = "blue", "Imputed" = "red")) +
      labs(title = paste0(name," - NHG: Observed vs Imputed"),
          x = "Tumor Grade", y = "Count") +
      theme_minimal()

    # LN (binary)
    ln_summary <- as.data.frame(table(ds$LN, ds$LN_imputed))
    colnames(ln_summary) <- c("LN", "Imputed", "Count")
    ln_summary$Imputed <- factor(
      ln_summary$Imputed,
      levels = c(FALSE, TRUE),
      labels = c("Observed", "Imputed")
    )

    p3 <- ggplot(ln_summary, aes(x = LN, y = Count, fill = Imputed)) +
      geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
      scale_fill_manual(values = c("Observed" = "blue", "Imputed" = "red")) +
      labs(title = paste0(name," - LN: Observed vs Imputed"),
          x = "Lymph Node Status", y = "Count") +
      theme_minimal()

    # Arrange plots
    grid.arrange(p1, p2, p3, ncol = 2)

    #----------------------------------------------------------------------
    # 2. Check for extreme/unrealistic imputed values
    #----------------------------------------------------------------------

    # Size.mm range check
    cat("\n=== Size.mm Range Check ===\n")
    cat("Observed values:\n")
    print(summary(ds$Size.mm[!ds$Size.mm_imputed]))
    cat("\nImputed values:\n")
    print(summary(ds$Size.mm[ds$Size.mm_imputed]))

    #----------------------------------------------------------------------
    # 3. Check relationships between imputed and predictor variables
    #----------------------------------------------------------------------

    # Size.mm by ER status
    p4 <- ggplot(ds, aes(x = ER, y = Size.mm, fill = Size.mm_imputed)) +
      geom_boxplot(alpha = 0.7) +
      scale_fill_manual(values = c("FALSE" = "blue", "TRUE" = "red"),
                        labels = c("Observed", "Imputed")) +
      labs(title = paste0(name," - Size.mm by ER Status"),
          subtitle = "Check if imputed values follow same pattern",
          x = "ER Status", y = "Tumor Size (mm)", fill = "") +
      theme_minimal()

    # NHG by ER status
    p5 <- ggplot(ds, aes(x = ER, fill = NHG)) +
      geom_bar(position = "fill") +
      facet_wrap(~NHG_imputed, labeller = labeller(NHG_imputed = c("FALSE" = "Observed", "TRUE" = "Imputed"))) +
      labs(title = paste0(name," - NHG distribution by ER Status"),
          subtitle = "Compare observed vs imputed patterns",
          x = "ER Status", y = "Proportion") +
      theme_minimal()

    grid.arrange(p4,p5, ncol = 2)

    #----------------------------------------------------------------------
    # 5. Count summary
    #----------------------------------------------------------------------

    cat("\n=== Imputation Summary ===\n")
    cat("Size.mm - Imputed:", sum(ds$Size.mm_imputed), 
        "out of", nrow(ds), 
        sprintf("(%.1f%%)\n", 100 * mean(ds$Size.mm_imputed)))

    cat("NHG - Imputed:", sum(ds$NHG_imputed), 
        "out of", nrow(ds), 
        sprintf("(%.1f%%)\n", 100 * mean(ds$NHG_imputed)))

    cat("LN - Imputed:", sum(ds$LN_imputed), 
        "out of", nrow(ds), 
        sprintf("(%.1f%%)\n", 100 * mean(ds$LN_imputed)))

}
