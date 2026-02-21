#!/usr/bin/env Rscript

################################################################################
# SIMPLE AUC COMPARISON PLOT
# Compares Clinical vs Methylation vs Combined for one cohort
#
# Usage: Rscript plot_auc_comparison.R <COHORT>
# Example: Rscript plot_auc_comparison.R ERpHER2n
################################################################################

library(ggplot2)

################################################################################
# 1. GET COHORT FROM COMMAND LINE
################################################################################

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 1) {
  cat("\nUSAGE: Rscript plot_auc_comparison.R <COHORT>\n")
  cat("\nExamples:\n")
  cat("  Rscript plot_auc_comparison.R ERpHER2n\n")
  cat("  Rscript plot_auc_comparison.R TNBC\n")
  cat("  Rscript plot_auc_comparison.R All\n\n")
  stop("Please provide exactly one argument: the cohort name")
}

COHORT <- args[1]

# Check valid cohort
if (!COHORT %in% c("TNBC", "ERpHER2n", "All")) {
  stop(sprintf("Invalid cohort '%s'. Must be: TNBC, ERpHER2n, or All", COHORT))
}

cat(sprintf("\n========================================\n"))
cat(sprintf("Cohort: %s\n", COHORT))
cat(sprintf("========================================\n\n"))

################################################################################
# 2. DEFINE FILE PATHS
################################################################################

# Base directory (adjust this if your output is elsewhere)
base_dir <- "./output/FineGray"

# Paths to the three performance summary files
clinical_file <- file.path(base_dir, COHORT, "Clinical", "final_results", "performance_summary.csv")
meth_file <- file.path(base_dir, COHORT, "Methylation", "Unadjusted", "final_results", "performance_summary.csv")
combined_file <- file.path(base_dir, COHORT, "Combined", "Unadjusted", "final_results", "performance_summary.csv")

# Output directory
output_dir <- file.path(base_dir, COHORT, "Comparison")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

################################################################################
# 3. LOAD DATA
################################################################################

cat("Loading performance files...\n")

all_data <- data.frame()

# Load Clinical
if (file.exists(clinical_file)) {
  cat(sprintf("  ✓ Clinical: %s\n", clinical_file))
  clinical <- read.csv(clinical_file)
  clinical$mode <- "Clinical"
  all_data <- rbind(all_data, clinical)
} else {
  cat(sprintf("  ✗ Clinical not found: %s\n", clinical_file))
}

# Load Methylation
if (file.exists(meth_file)) {
  cat(sprintf("  ✓ Methylation: %s\n", meth_file))
  meth <- read.csv(meth_file)
  meth$mode <- "Methylation"
  all_data <- rbind(all_data, meth)
} else {
  cat(sprintf("  ✗ Methylation not found: %s\n", meth_file))
}

# Load Combined
if (file.exists(combined_file)) {
  cat(sprintf("  ✓ Combined: %s\n", combined_file))
  combined <- read.csv(combined_file)
  combined$mode <- "Combined"
  all_data <- rbind(all_data, combined)
} else {
  cat(sprintf("  ✗ Combined not found: %s\n", combined_file))
}

# Check we have at least one file
if (nrow(all_data) == 0) {
  stop("\nERROR: No performance files found. Please run the pipeline first.")
}

cat(sprintf("\nLoaded %d mode(s)\n\n", length(unique(all_data$mode))))

################################################################################
# 4. EXTRACT AUC DATA
################################################################################

# Keep only AUC metrics (auc_1yr, auc_2yr, etc.)
auc_data <- all_data[grepl("^auc_\\d+yr$", all_data$metric), ]

# Extract time from metric name (e.g., "auc_5yr" -> 5)
auc_data$time <- as.numeric(gsub("auc_(\\d+)yr", "\\1", auc_data$metric))

# Make mode a factor with specific order
auc_data$mode <- factor(auc_data$mode, levels = c("Clinical", "Methylation", "Combined"))

cat(sprintf("Time points: %s\n", paste(sort(unique(auc_data$time)), collapse = ", ")))
cat(sprintf("Modes: %s\n\n", paste(sort(unique(auc_data$mode)), collapse = ", ")))

################################################################################
# 5. CREATE PLOT
################################################################################

cat("Creating plot...\n")

# Title
title_text <- switch(COHORT,
                     "TNBC" = "Triple-Negative Breast Cancer",
                     "ERpHER2n" = "ER+/HER2- Breast Cancer",
                     "All" = "All Patients"
)

# Plot
p <- ggplot(auc_data, aes(x = time, y = mean, color = mode, fill = mode)) +
  
  # Confidence interval ribbons (semi-transparent)
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.2, color = NA) +
  
  # Lines connecting points
  geom_line(size = 1.2) +
  
  # Points
  geom_point(size = 3) +
  
  # Reference line at 0.5 (random chance)
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray50") +
  
  # Colors
  scale_color_manual(
    name = "Model",
    values = c("Clinical" = "#E69F00", "Methylation" = "#56B4E9", "Combined" = "#009E73")
  ) +
  scale_fill_manual(
    name = "Model",
    values = c("Clinical" = "#E69F00", "Methylation" = "#56B4E9", "Combined" = "#009E73")
  ) +
  
  # Axis settings
  scale_x_continuous(breaks = sort(unique(auc_data$time))) +
  ylim(0.0, 1.0) +
  
  # Labels
  labs(
    title = paste0("AUC Comparison: ", title_text),
    subtitle = "Mean ± 95% CI across CV folds",
    x = "Time (years)",
    y = "AUC"
  ) +
  
  # Theme
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

################################################################################
# 6. SAVE PLOT
################################################################################

pdf_file <- file.path(output_dir, "auc_comparison.pdf")
png_file <- file.path(output_dir, "auc_comparison.png")

ggsave(pdf_file, p, width = 10, height = 6)
ggsave(png_file, p, width = 10, height = 6, dpi = 300)

cat(sprintf("\n✓ Saved:\n"))
cat(sprintf("  %s\n", pdf_file))
cat(sprintf("  %s\n", png_file))

################################################################################
# 7. PRINT SUMMARY
################################################################################

cat(sprintf("\n========================================\n"))
cat("Summary at Key Time Points:\n")
cat(sprintf("========================================\n\n"))

for (t in c(3, 5, 10)) {
  t_data <- auc_data[auc_data$time == t, ]
  
  if (nrow(t_data) > 0) {
    cat(sprintf("%d years:\n", t))
    for (i in 1:nrow(t_data)) {
      cat(sprintf("  %12s: %.3f (95%% CI: %.3f-%.3f)\n",
                  t_data$mode[i], 
                  t_data$mean[i],
                  t_data$ci_lower[i],
                  t_data$ci_upper[i]))
    }
    cat("\n")
  }
}

cat("✓ Done!\n\n")