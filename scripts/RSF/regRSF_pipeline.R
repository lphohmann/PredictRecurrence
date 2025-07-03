#!/usr/bin/env Rscript

# ================================
# RSF Pipeline for Methylation Data
# ================================

# Load utility scripts
source("utils.R")
source("rsf_functions.R")

# Load required libraries
suppressPackageStartupMessages({
  library(data.table)
  library(tidyverse)
  library(caret)
  library(survival)
  library(randomForestSRC)
})

# ----------------
# Parameters
# ----------------
top_n_cpgs      <- 10000
inner_cv_folds  <- 3
outer_cv_folds  <- 5
eval_time_grid  <- seq(1, 10, by = 0.5)
output_dir      <- "output/RSF_adjusted"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ----------------
# Logging
# ----------------
sink(file.path(output_dir, "pipeline_run.log"), append = FALSE, split = TRUE)
log_msg("RSF pipeline started")

# ----------------
# Load & Preprocess Data
# ----------------
train_ids <- fread("./data/train/train_subcohorts/TNBC_train_ids.csv", header = FALSE)[[1]]
data      <- load_training_data(
  train_ids,
  "./data/train/train_methylation_adjusted.csv",
  "./data/train/train_clinical.csv"
)
X <- preprocess_data(data$beta, top_n_cpgs = top_n_cpgs)

# Survival outcome
y <- data$clinical %>%
  select(RFi_years, RFi_event) %>%
  rename(time = RFi_years, status = RFi_event)

# ----------------
# Nested CV
# ----------------
param_grid <- define_param_grid()
models     <- run_nested_cv(X, y, param_grid, outer_cv_folds, inner_cv_folds)

# ----------------
# Evaluation
# ----------------
perf <- evaluate_models(models, X, y, eval_time_grid)

# ----------------
# Save & Select Best
# ----------------
saveRDS(models, file.path(output_dir, "outer_cv_models.rds"))
saveRDS(perf,   file.path(output_dir, "outer_cv_performance.rds"))

best_idx <- select_best_model(perf, metric = "auc")
saveRDS(models[[best_idx]], file.path(output_dir, "best_model.rds"))

log_msg(sprintf("Best model: Fold %d", best_idx))
log_msg("Pipeline completed.")
sink()
