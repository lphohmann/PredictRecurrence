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
# func
#######################################################################
impute_clinical_data <- function(data, 
                                  vars_to_impute,
                                  complete_vars,
                                  binary_vars = NULL,
                                  ordered_vars = NULL,
                                  maxit = 10,
                                  seed = 13) {
  
  all_vars <- c(vars_to_impute, complete_vars)

  # Print missingness before
  cat("\n=== Missingness BEFORE imputation ===\n")
  missing_counts <- colSums(is.na(data[vars_to_impute]))
  print(missing_counts)
  cat("Total missing:", sum(missing_counts), "\n")
  
  # Prepare data for MICE
  data_for_mice <- data[, all_vars, drop = FALSE]
  
  # Set up imputation methods
  # Default is PMM for continuous variables
  methods <- make.method(data_for_mice)
  
  # Override methods for categorical variables
  if (!is.null(binary_vars)) {
    for (var in binary_vars) {
      if (var %in% names(methods)) {
        methods[var] <- "logreg"
      }
    }
  }
  
  if (!is.null(ordered_vars)) {
    for (var in ordered_vars) {
      if (var %in% names(methods)) {
        methods[var] <- "polr"
      }
    }
  }
  
  # Print methods being used
  cat("\nImputation methods:\n")
  impute_methods <- methods[vars_to_impute]
  impute_methods <- impute_methods[impute_methods != ""]  # Remove empty ones
  print(impute_methods)
  
  # Run MICE
  cat("\nRunning MICE imputation (maxit =", maxit, ")...\n")
  
  mice_result <- mice(data_for_mice, 
                      m = 1,
                      maxit = maxit,
                      method = methods,
                      seed = seed,
                      printFlag = FALSE)
  
  # Check for logged events (problems during imputation)
  if (!is.null(mice_result$loggedEvents)) {
    cat("\nWARNING: MICE logged", nrow(mice_result$loggedEvents), "events:\n")
    print(mice_result$loggedEvents)
  }
  
  # Get imputed data
  imputed_data <- complete(mice_result, action = 1)
  
  # Replace missing values in original data
  result <- data
  for (var in vars_to_impute) {
    result[[var]][is.na(data[[var]])] <- imputed_data[[var]][is.na(data[[var]])]
  }
  
  # Print missingness after
  cat("\n=== Missingness AFTER imputation ===\n")
  missing_counts_after <- colSums(is.na(result[vars_to_impute]))
  print(missing_counts_after)
  cat("Total missing:", sum(missing_counts_after), "\n")
  
  # Warn if any variables still have missing values
  still_missing <- vars_to_impute[missing_counts_after > 0]
  if (length(still_missing) > 0) {
    cat("\nWARNING: These variables still have missing values:\n")
    print(missing_counts_after[still_missing])
    cat("Consider adding more predictors or using fallback imputation.\n")
  }
  
  cat("\n")
  return(result)
}

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



# just chekcign

#######################################################################
# VISUALIZATION AND ERROR CHECKING AFTER IMPUTATION
#######################################################################

library(ggplot2)
library(gridExtra)  # for arranging multiple plots


# After imputation, compare distributions
# Create indicator for which values were imputed
clin.dat.train$Size.mm_imputed <- is.na(clin.dat.train_original$Size.mm)
clin.dat.train$NHG_imputed <- is.na(clin.dat.train_original$NHG)
clin.dat.train$LN_imputed <- is.na(clin.dat.train_original$LN)

#----------------------------------------------------------------------
# 1. Compare distributions: Observed vs Imputed
#----------------------------------------------------------------------

# Size.mm (continuous)
p1 <- ggplot(clin.dat.train, aes(x = Size.mm, fill = Size.mm_imputed)) +
  geom_histogram(position = "identity", alpha = 0.6, bins = 30) +
  scale_fill_manual(values = c("FALSE" = "blue", "TRUE" = "red"),
                    labels = c("Observed", "Imputed")) +
  labs(title = "Size.mm: Observed vs Imputed",
       x = "Tumor Size (mm)", y = "Count", fill = "") +
  theme_minimal()

# Age (continuous) - if it had missing values
# p_age <- ggplot(clin.dat.train, aes(x = Age, fill = Age_imputed)) +
#   geom_histogram(position = "identity", alpha = 0.6, bins = 30) +
#   scale_fill_manual(values = c("FALSE" = "blue", "TRUE" = "red")) +
#   labs(title = "Age: Observed vs Imputed") +
#   theme_minimal()

# NHG (categorical)
nhg_summary <- as.data.frame(table(clin.dat.train$NHG, clin.dat.train$NHG_imputed))
colnames(nhg_summary) <- c("NHG", "Imputed", "Count")

nhg_summary$Imputed <- factor(
  nhg_summary$Imputed,
  levels = c(FALSE, TRUE),
  labels = c("Observed", "Imputed")
)

p2 <- ggplot(nhg_summary, aes(x = NHG, y = Count, fill = Imputed)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  scale_fill_manual(values = c("Observed" = "blue", "Imputed" = "red")) +
  labs(title = "NHG: Observed vs Imputed",
       x = "Tumor Grade", y = "Count") +
  theme_minimal()

# LN (binary)
ln_summary <- as.data.frame(table(clin.dat.train$LN, clin.dat.train$LN_imputed))
colnames(ln_summary) <- c("LN", "Imputed", "Count")
ln_summary$Imputed <- factor(
  ln_summary$Imputed,
  levels = c(FALSE, TRUE),
  labels = c("Observed", "Imputed")
)

p3 <- ggplot(ln_summary, aes(x = LN, y = Count, fill = Imputed)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  scale_fill_manual(values = c("Observed" = "blue", "Imputed" = "red")) +
  labs(title = "LN: Observed vs Imputed",
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
print(summary(clin.dat.train$Size.mm[!clin.dat.train$Size.mm_imputed]))
cat("\nImputed values:\n")
print(summary(clin.dat.train$Size.mm[clin.dat.train$Size.mm_imputed]))

# Flag potential issues
size_imputed <- clin.dat.train$Size.mm[clin.dat.train$Size.mm_imputed]
if (any(size_imputed < 1 | size_imputed > 200)) {
  cat("\nWARNING: Some imputed Size.mm values seem unrealistic!\n")
  print(size_imputed[size_imputed < 1 | size_imputed > 200])
}

#----------------------------------------------------------------------
# 3. Check relationships between imputed and predictor variables
#----------------------------------------------------------------------

# Example: Size.mm by ER status
p4 <- ggplot(clin.dat.train, aes(x = ER, y = Size.mm, fill = Size.mm_imputed)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c("FALSE" = "blue", "TRUE" = "red"),
                    labels = c("Observed", "Imputed")) +
  labs(title = "Size.mm by ER Status",
       subtitle = "Check if imputed values follow same pattern",
       x = "ER Status", y = "Tumor Size (mm)", fill = "") +
  theme_minimal()

# NHG by HER2 status
p5 <- ggplot(clin.dat.train, aes(x = HER2, fill = NHG)) +
  geom_bar(position = "fill") +
  facet_wrap(~NHG_imputed, labeller = labeller(NHG_imputed = c("FALSE" = "Observed", "TRUE" = "Imputed"))) +
  labs(title = "NHG distribution by HER2 Status",
       subtitle = "Compare observed vs imputed patterns",
       x = "HER2 Status", y = "Proportion") +
  theme_minimal()

grid.arrange(p4, p5, ncol = 2)

#----------------------------------------------------------------------
# 4. Correlation check (for continuous variables)
#----------------------------------------------------------------------

# Check if imputation preserved correlations
cat("\n=== Correlation: Size.mm vs Age ===\n")
cat("Observed data only:\n")
observed_only <- clin.dat.train[!clin.dat.train$Size.mm_imputed, ]
print(cor(observed_only$Size.mm, observed_only$Age, use = "complete.obs"))

cat("\nAll data (observed + imputed):\n")
print(cor(clin.dat.train$Size.mm, clin.dat.train$Age, use = "complete.obs"))

#----------------------------------------------------------------------
# 5. Count summary
#----------------------------------------------------------------------

cat("\n=== Imputation Summary ===\n")
cat("Size.mm - Imputed:", sum(clin.dat.train$Size.mm_imputed), 
    "out of", nrow(clin.dat.train), 
    sprintf("(%.1f%%)\n", 100 * mean(clin.dat.train$Size.mm_imputed)))

cat("NHG - Imputed:", sum(clin.dat.train$NHG_imputed), 
    "out of", nrow(clin.dat.train), 
    sprintf("(%.1f%%)\n", 100 * mean(clin.dat.train$NHG_imputed)))

cat("LN - Imputed:", sum(clin.dat.train$LN_imputed), 
    "out of", nrow(clin.dat.train), 
    sprintf("(%.1f%%)\n", 100 * mean(clin.dat.train$LN_imputed)))
