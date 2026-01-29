#!/usr/bin/env Rscript

################################################################################
# Stability Analysis: Feature Selection Across Nested CV
# Author: Lennart Hohmann

################################################################################

################################################################################
# LOAD LIBRARIES
################################################################################

library(ggplot2)
library(reshape2)
library(pheatmap)
library(RColorBrewer)
library(dplyr)

################################################################################
# PARAMETER SETTINGS - CHANGE THESE TO SWITCH BETWEEN ANALYSES
################################################################################

# Set your cohort and data type here
COHORT <- "ERpHER2n"           # Options: "TNBC", "ERpHER2n", "All"
DATA_TYPE <- "Methylation"     # Options: "Methylation", "Combined"

# Construct the path to results based on parameters
setwd("~/PhD_Workspace/PredictRecurrence/")

results_path <- file.path(
  "./output/FineGray",
  COHORT,
  DATA_TYPE,
  "Unadjusted/outer_fold_results.RData"
)

# Print settings
cat(sprintf("\n========================================\n"))
cat(sprintf("STABILITY ANALYSIS PARAMETERS\n"))
cat(sprintf("========================================\n"))
cat(sprintf("Cohort:      %s\n", COHORT))
cat(sprintf("Data type:   %s\n", DATA_TYPE))
cat(sprintf("Results:     %s\n", results_path))
cat(sprintf("========================================\n\n"))

################################################################################
# LOAD DATA
################################################################################

# Check if file exists
if (!file.exists(results_path)) {
  stop(sprintf("Results file not found: %s\n", results_path))
}

cat("Loading data from:", results_path, "\n")

# Load the outer fold results
load(results_path)

# Check what's loaded
if (!exists("outer_fold_results")) {
  stop("outer_fold_results not found in the loaded file")
}

# Check fold completion status
completed_fold_indices <- which(!sapply(outer_fold_results, is.null))
n_completed_folds <- length(completed_fold_indices)
N_OUTER_FOLDS <- length(outer_fold_results)

cat(sprintf("CV Folds: %d/%d completed successfully\n", 
            n_completed_folds, N_OUTER_FOLDS))

if (n_completed_folds == 0) {
  stop("No completed folds found in results")
}

if (n_completed_folds < N_OUTER_FOLDS) {
  skipped_folds <- setdiff(1:N_OUTER_FOLDS, completed_fold_indices)
  cat(sprintf("  WARNING: Folds %s were skipped\n",
              paste(skipped_folds, collapse = ", ")))
}
cat("\n")

################################################################################
# EXTRACT STABILITY INFORMATION
################################################################################

cat("Extracting stability information from nested CV results...\n")

# Initialize storage for stability data
stability_data_rfi <- list()
stability_data_death <- list()

for (fold_idx in completed_fold_indices) {
  fold_result <- outer_fold_results[[fold_idx]]
  
  # Extract RFI stability info (if it exists)
  if (!is.null(fold_result$coxnet_rfi_complete_results$best_result$stability_info)) {
    stability_data_rfi[[fold_idx]] <- list(
      fold = fold_idx,
      stability_info = fold_result$coxnet_rfi_complete_results$best_result$stability_info,
      stability_matrix = fold_result$coxnet_rfi_complete_results$best_result$stability_matrix,
      sign_matrix = fold_result$coxnet_rfi_complete_results$best_result$sign_matrix,
      features_selected = fold_result$features_rfi,
      best_alpha = fold_result$coxnet_rfi_complete_results$best_alpha,
      best_lambda = fold_result$coxnet_rfi_complete_results$best_lambda
    )
  }
  
  # Extract Death stability info (if it exists)
  if (!is.null(fold_result$coxnet_death_complete_results$best_result$stability_info)) {
    stability_data_death[[fold_idx]] <- list(
      fold = fold_idx,
      stability_info = fold_result$coxnet_death_complete_results$best_result$stability_info,
      stability_matrix = fold_result$coxnet_death_complete_results$best_result$stability_matrix,
      sign_matrix = fold_result$coxnet_death_complete_results$best_result$sign_matrix,
      features_selected = fold_result$features_death,
      best_alpha = fold_result$coxnet_death_complete_results$best_alpha,
      best_lambda = fold_result$coxnet_death_complete_results$best_lambda
    )
  }
}

# Remove NULL entries
stability_data_rfi <- stability_data_rfi[!sapply(stability_data_rfi, is.null)]
stability_data_death <- stability_data_death[!sapply(stability_data_death, is.null)]

cat(sprintf("  RFI stability data: %d outer folds\n", length(stability_data_rfi)))
cat(sprintf("  Death stability data: %d outer folds\n", length(stability_data_death)))

if (length(stability_data_rfi) == 0 && length(stability_data_death) == 0) {
  stop("No stability information found. Was compute_stability=TRUE in your analysis?")
}

cat("\n")

################################################################################
# AGGREGATE STABILITY ACROSS OUTER FOLDS
################################################################################

cat("Aggregating feature selection across outer folds...\n")

# Function to aggregate features across outer folds
aggregate_feature_selection <- function(stability_data, model_name) {
  
  all_features <- unique(unlist(lapply(stability_data, function(x) x$features_selected)))
  
  if (length(all_features) == 0) {
    cat(sprintf("  WARNING: No features selected in %s model\n", model_name))
    return(NULL)
  }
  
  # Create matrix: rows = features, columns = outer folds
  n_outer_folds <- length(stability_data)
  outer_selection_matrix <- matrix(
    0, 
    nrow = length(all_features), 
    ncol = n_outer_folds,
    dimnames = list(all_features, paste0("OuterFold", sapply(stability_data, `[[`, "fold")))
  )
  
  # Fill in which features were selected in each outer fold
  for (i in 1:n_outer_folds) {
    selected <- stability_data[[i]]$features_selected
    outer_selection_matrix[selected, i] <- 1
  }
  
  # Calculate outer fold selection frequency
  outer_freq <- rowSums(outer_selection_matrix) / n_outer_folds
  
  # Get average inner fold stability for each feature
  inner_stability <- sapply(all_features, function(feat) {
    # Get stability scores for this feature across outer folds where it appears
    stabilities <- sapply(stability_data, function(fold_data) {
      info <- fold_data$stability_info
      if (feat %in% info$feature) {
        return(info$selection_freq[info$feature == feat])
      } else {
        return(NA)
      }
    })
    return(mean(stabilities, na.rm = TRUE))
  })
  
  # Combine into dataframe
  feature_summary <- data.frame(
    feature = all_features,
    outer_fold_freq = outer_freq,
    mean_inner_stability = inner_stability,
    n_outer_folds_selected = rowSums(outer_selection_matrix),
    stringsAsFactors = FALSE
  )
  
  feature_summary <- feature_summary[order(feature_summary$outer_fold_freq, decreasing = TRUE), ]
  rownames(feature_summary) <- NULL
  
  return(list(
    summary = feature_summary,
    outer_selection_matrix = outer_selection_matrix
  ))
}

# Aggregate for both models
if (length(stability_data_rfi) > 0) {
  rfi_aggregated <- aggregate_feature_selection(stability_data_rfi, "RFI")
  cat(sprintf("  RFI: %d unique features across %d outer folds\n", 
              nrow(rfi_aggregated$summary), 
              ncol(rfi_aggregated$outer_selection_matrix)))
}

if (length(stability_data_death) > 0) {
  death_aggregated <- aggregate_feature_selection(stability_data_death, "Death")
  cat(sprintf("  Death: %d unique features across %d outer folds\n", 
              nrow(death_aggregated$summary), 
              ncol(death_aggregated$outer_selection_matrix)))
}

cat("\n")

################################################################################
# SUMMARY STATISTICS
################################################################################

cat("========================================\n")
cat("STABILITY SUMMARY STATISTICS\n")
cat("========================================\n\n")

# Helper function to print summary for one model
print_model_summary <- function(aggregated_data, model_name) {
  if (is.null(aggregated_data)) return()
  
  summary_df <- aggregated_data$summary
  
  cat(sprintf("--- %s MODEL ---\n", model_name))
  cat(sprintf("Total features: %d\n", nrow(summary_df)))
  cat(sprintf("Features selected in all outer folds: %d\n", 
              sum(summary_df$outer_fold_freq == 1)))
  cat(sprintf("Features selected in ≥80%% outer folds: %d\n", 
              sum(summary_df$outer_fold_freq >= 0.8)))
  cat(sprintf("Features selected in ≥60%% outer folds: %d\n", 
              sum(summary_df$outer_fold_freq >= 0.6)))
  cat(sprintf("Mean outer fold frequency: %.2f\n", 
              mean(summary_df$outer_fold_freq)))
  cat(sprintf("Mean inner fold stability: %.2f\n", 
              mean(summary_df$mean_inner_stability, na.rm = TRUE)))
  cat("\n")
  
  # Show top 10 most stable features
  cat("Top 10 most consistently selected features:\n")
  print(head(summary_df[, c("feature", "outer_fold_freq", "mean_inner_stability")], 10))
  cat("\n")
}

if (length(stability_data_rfi) > 0) {
  print_model_summary(rfi_aggregated, "RFI")
}

if (length(stability_data_death) > 0) {
  print_model_summary(death_aggregated, "DEATH")
}

################################################################################
# CREATE OUTPUT DIRECTORY FOR PLOTS
################################################################################

plot_dir <- file.path(dirname(results_path), "stability_figures")
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

cat(sprintf("Plots will be saved to: %s\n\n", plot_dir))

################################################################################
# SET PLOTTING THEME
################################################################################

theme_pub <- theme_bw(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "white"),
    legend.position = "right"
  )

################################################################################
# READY FOR PLOTTING
################################################################################

cat("========================================\n")
cat("Data loaded and processed successfully!\n")
cat("You can now create stability plots.\n")
cat("========================================\n\n")

# Available objects for plotting:
# - stability_data_rfi: List with stability info for each outer fold (RFI)
# - stability_data_death: List with stability info for each outer fold (Death)
# - rfi_aggregated: Aggregated feature selection across outer folds (RFI)
# - death_aggregated: Aggregated feature selection across outer folds (Death)
# - outer_fold_results: Original results with all models and predictions







stability_results$selection_matrix
str(stability_results$stability_metrics)
plotdata <- fg_stability[c()]
final_fg_input_features <- fg_stability[fg_stability$n_selected >= 4 & fg_stability$direction_consistent == TRUE,]

str(stability_results)

features_to_plot <- fg_stability[fg_stability$n_selected >= 3 & fg_stability$direction_consistent == TRUE,]

features_to_plot <- fg_stability$feature[fg_stability$n_selected >= 3 & fg_stability$direction_consistent == TRUE]

fg_matrix_filtered <- fg_matrix[features_to_plot, , drop = FALSE]

View(stability_results$selection_matrix)

# plot
# Load required library
library(pheatmap)

# Extract and prepare the selection matrix
selection_data <- stability_results$selection_matrix
rownames(selection_data) <- selection_data$feature
selection_data <- selection_data[, -1]  # Remove the 'feature' column
selection_matrix <- as.matrix(selection_data)

# Create the output filepath
output_file <- file.path(current_output_dir, "feature_selection_heatmap.png")

# Save the heatmap to file
png(filename = output_file,
    width = 800,      # Width in pixels
    height = 1000,    # Height in pixels (adjust based on number of features)
    res = 150)        # Resolution (higher = better quality)

pheatmap(selection_matrix,
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         color = c("white", "darkblue"),
         legend = TRUE,
         main = "Feature Selection Across CV Folds",
         fontsize = 10,
         border_color = "grey60")

dev.off()  # Close the graphics device to finalize the file

# Print confirmation
cat("Heatmap saved to:", output_file, "\n")