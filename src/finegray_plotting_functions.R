################################################################################
#
# FINE-GRAY COMPETING RISKS - PUBLICATION PLOTTING FUNCTIONS
# Author: Lennart Hohmann
#
# Purpose: Publication-quality visualizations for competing risks analysis
#          Handles both methylation-only and combined (methylation + clinical)
#          Handles cohort-specific differences (TNBC: 5yr max, ERpHER2n: 10yr)
#
################################################################################

library(ggplot2)
library(gridExtra)
library(dplyr)
library(reshape2)

################################################################################
# THEME
################################################################################

get_publication_theme <- function() {
  theme_bw(base_size = 11) +
    theme(
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "white"),
      legend.position = "right",
      plot.title = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(size = 9.5, color = "gray30")
    )
}

################################################################################
# FIGURE 1: COXNET POOLED FEATURES - SELECTION STABILITY HEATMAP
################################################################################

#' Plot CoxNet Pooled Feature Selection Stability
#'
#' Shows which features (CpGs + clinical) were selected across CV folds
#' after pooling RFI and Death models.
#'
#' @param plot_dir Output directory
#' @param outer_fold_results List of fold results
#' @param min_freq Minimum selection frequency to display (default: 0.2)
#'
plot_coxnet_stability_heatmap <- function(plot_dir, 
                                          outer_fold_results,
                                          min_freq = 0.2) {
  
  cat("Creating Figure 1: CoxNet pooled features stability...\n")
  
  # Get completed folds
  completed <- which(!sapply(outer_fold_results, is.null))
  n_folds <- length(completed)
  
  # Collect all pooled features
  all_features <- unique(unlist(lapply(completed, function(i) {
    outer_fold_results[[i]]$penalized_fg$input_cpgs
  })))
  
  
  if (length(all_features) == 0) {
    cat("  WARNING: No features selected. Skipping.\n\n")
    return(invisible(NULL))
  }
  
  # Create selection matrix
  sel_mat <- sapply(completed, function(i) {
    fold_features <- outer_fold_results[[i]]$penalized_fg$input_cpgs
    as.integer(all_features %in% fold_features)
  })
  
  rownames(sel_mat) <- all_features
  colnames(sel_mat) <- paste0("Fold", completed)
  
  # Calculate selection frequency
  sel_freq <- rowSums(sel_mat) / ncol(sel_mat)
  
  # Filter by minimum frequency
  keep <- sel_freq >= min_freq
  if (sum(keep) == 0) {
    cat(sprintf("  WARNING: No features with freq ≥ %.0f%%. Skipping.\n\n", 
                min_freq * 100))
    return(invisible(NULL))
  }
  
  sel_mat <- sel_mat[keep, , drop = FALSE]
  sel_freq <- sel_freq[keep]
  
  # Order by selection frequency
  feature_order <- names(sort(sel_freq, decreasing = TRUE))
  
  # Reshape for ggplot
  sel_df <- reshape2::melt(sel_mat)
  colnames(sel_df) <- c("Feature", "Fold", "Selected")
  sel_df$Feature <- factor(sel_df$Feature, levels = rev(feature_order))
  
  # Create plot
  p <- ggplot(sel_df, aes(x = Fold, y = Feature, fill = factor(Selected))) +
    geom_tile(color = "white", linewidth = 0.5) +
    scale_fill_manual(
      values = c("0" = "gray95", "1" = "#2166AC"),
      labels = c("Not selected", "Selected"),
      name = ""
    ) +
    labs(
      title = "CoxNet Feature Selection Stability (RFI ∪ Death)",
      subtitle = sprintf("Features selected ≥%.0f%% of folds (n=%d)", 
                         min_freq * 100, sum(keep)),
      x = "Cross-Validation Fold",
      y = ""
    ) +
    get_publication_theme() +
    theme(
      panel.grid = element_blank(),
      legend.position = "right",
      axis.text.y = element_text(size = 8),
      axis.text.x = element_text(size = 10),
      axis.title.x = element_text(size = 16),    # X-axis label
      axis.title.y = element_text(size = 16),    # Y-axis label
      plot.title = element_text(size = 18, face = "bold"),
      plot.subtitle = element_text(size = 16, color = "gray30"),
      legend.title = element_text(size = 16, face = "bold"),
      legend.text = element_text(size = 14),
      legend.justification = c(1, 0),
      legend.background = element_rect(
        fill = "white",
        color = "gray80",
        linewidth = 0.3
      )
    )
  
  
  # Save
  height <- max(6, sum(keep) * 0.2)
  ggsave(file.path(plot_dir, "Fig1_coxnet_stability.pdf"),
         p, width = 10, height = height)
  ggsave(file.path(plot_dir, "Fig1_coxnet_stability.png"),
         p, width = 10, height = height, dpi = 300)
  
  cat(sprintf("  Saved: Fig1_coxnet_stability.pdf/.png (%d features)\n\n", sum(keep)))
}


################################################################################
# FIGURE 2: COXNET COEFFICIENTS (RFI vs DEATH)
################################################################################

#' Plot CoxNet Coefficient Comparison
#'
#' Shows biological effects: which features predict RFI vs Death.
#' Important for understanding competing risks biology.
#'
#' @param plot_dir Output directory
#' @param outer_fold_results List of fold results
#' @param min_freq Minimum selection frequency to include
#'
plot_coxnet_coefficients <- function(plot_dir,
                                     outer_fold_results,
                                     min_freq = 0.4) {
  
  cat("Creating Figure 2: CoxNet coefficients (RFI vs Death)...\n")
  
  completed <- which(!sapply(outer_fold_results, is.null))
  
  # Collect all coefficients across folds
  all_coef_tables <- lapply(completed, function(i) {
    coef_table <- outer_fold_results[[i]]$coxnet$coefficients
    coef_table$fold <- i
    coef_table
  })
  
  combined <- do.call(rbind, all_coef_tables)
  
  # Calculate mean coefficients per feature
  mean_coef <- combined %>%
    group_by(feature) %>%
    summarise(
      rfi_coef = mean(cox_rfi_coef, na.rm = TRUE),
      death_coef = mean(cox_death_coef, na.rm = TRUE),
      n_folds_rfi = sum(!is.na(cox_rfi_coef)),
      n_folds_death = sum(!is.na(cox_death_coef)),
      .groups = "drop"
    ) %>%
    mutate(
      rfi_coef = ifelse(is.nan(rfi_coef), 0, rfi_coef),
      death_coef = ifelse(is.nan(death_coef), 0, death_coef),
      sel_freq_rfi = n_folds_rfi / length(completed),
      sel_freq_death = n_folds_death / length(completed),
      max_sel_freq = pmax(sel_freq_rfi, sel_freq_death)
    )
  
  # Filter to stable features
  stable <- mean_coef %>%
    filter(max_sel_freq >= min_freq & (rfi_coef != 0 | death_coef != 0))
  
  if (nrow(stable) == 0) {
    cat(sprintf("  WARNING: No features with freq ≥ %.0f%%. Skipping.\n\n",
                min_freq * 100))
    return(invisible(NULL))
  }
  
  # Reshape for plotting
  coef_long <- data.frame(
    feature = rep(stable$feature, 2),
    model = rep(c("RFI", "Death"), each = nrow(stable)),
    coefficient = c(stable$rfi_coef, stable$death_coef)
  )
  
  # Remove zeros
  coef_long <- coef_long[coef_long$coefficient != 0, ]
  
  # Order features by absolute RFI coefficient
  feature_order <- stable$feature[order(abs(stable$rfi_coef), decreasing = TRUE)]
  coef_long$feature <- factor(coef_long$feature, levels = rev(feature_order))
  
  # Determine axis limits
  max_abs <- max(abs(coef_long$coefficient), na.rm = TRUE)
  axis_lim <- ceiling(max_abs * 1.1 * 10) / 10
  
  # Create plot
  p <- ggplot(coef_long, aes(x = coefficient, y = feature, color = model)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.8) +
    geom_point(size = 3, position = position_dodge(width = 0.5)) +
    scale_color_manual(
      values = c("RFI" = "#D73027", "Death" = "#4575B4"),
      name = "Model"
    ) +
    scale_x_continuous(limits = c(-axis_lim, axis_lim)) +
    labs(
      title = "CoxNet Coefficients: Biology of Competing Risks",
      subtitle = sprintf("Features selected ≥%.0f%% of folds (mean coefficients)", 
                         min_freq * 100),
      x = "Coefficient (log hazard ratio)",
      y = ""
    ) +
    get_publication_theme() +
    theme(
      axis.text.y = element_text(size = 7),
      legend.position = c(0.85, 0.15),
      legend.background = element_rect(fill = "white", color = "gray80")
    )
  
  # Save
  height <- max(6, nrow(stable) * 0.25)
  ggsave(file.path(plot_dir, "Fig2_coxnet_coefficients.pdf"),
         p, width = 9, height = height)
  ggsave(file.path(plot_dir, "Fig2_coxnet_coefficients.png"),
         p, width = 9, height = height, dpi = 300)
  
  cat(sprintf("  Saved: Fig2_coxnet_coefficients.pdf/.png (%d features)\n\n", 
              nrow(stable)))
}


################################################################################
# FIGURE 3: PENALIZED FG CpG COEFFICIENTS (MRS COMPONENTS)
################################################################################

#
plot_penalized_fg_coefficients <- function(plot_dir, cpg_coefs, outer_fold_results) {
  
  cat("Creating Figure 3: Penalized FG CpG coefficients (MRS)...\n")
  
  # Final CpGs
  df <- data.frame(
    feature = names(cpg_coefs),
    coefficient = as.numeric(cpg_coefs),
    stringsAsFactors = FALSE
  )
  
  # Collect CoxNet sources across folds
  rfi_cpgs <- unique(unlist(lapply(outer_fold_results, function(f) {
    f$coxnet$features_rfi
  })))
  
  death_cpgs <- unique(unlist(lapply(outer_fold_results, function(f) {
    f$coxnet$features_death
  })))
  
  # Assign source
  df$source <- "RFI CoxNet"
  df$source[df$feature %in% death_cpgs] <- "Death CoxNet"
  df$source[df$feature %in% rfi_cpgs & df$feature %in% death_cpgs] <- "Both"
  
  df$source <- factor(df$source,
                      levels = c("RFI CoxNet", "Death CoxNet", "Both"))
  
  # Order by coefficient
  df <- df[order(df$coefficient), ]
  df$feature <- factor(df$feature, levels = df$feature)
  
  # Axis limits
  max_abs <- max(abs(df$coefficient))
  axis_lim <- ceiling(max_abs * 1.1 * 10) / 10
  
  # Plot
  p <- ggplot(df, aes(x = coefficient, y = feature, color = source)) +
    geom_vline(xintercept = 0, linetype = "dashed",
               color = "gray50", linewidth = 1) +
    geom_point(size = 6) +
    scale_x_continuous(limits = c(-axis_lim, axis_lim)) +
    scale_color_manual(
      values = c(
        "RFI CoxNet"   = "#D73027",
        "Death CoxNet" = "#4575B4",
        "Both"         = "#7B3294"
      ),
      name = "Outer-CV CoxNet selection"
    ) +
    labs(
      title = "MeRS construction: Penalized Fine-Gray Model",
      subtitle = sprintf(
        "Stable CpGs (n=%d); selected ≥40%% of nCV folds",
        nrow(df)
      ),
      x = "Coefficient",
      y = ""
    ) +
    get_publication_theme() +
    theme(
      axis.text.y = element_text(size = 10),
      axis.text.x = element_text(size = 10),
      axis.title.x = element_text(size = 16),    # X-axis label
      axis.title.y = element_text(size = 16),    # Y-axis label
      plot.title = element_text(size = 18, face = "bold"),
      plot.subtitle = element_text(size = 16, color = "gray30"),
      legend.title = element_text(size = 16, face = "bold"),
      legend.text = element_text(size = 14),
      legend.position = c(0.98, 0.02),           # bottom-right inside
      legend.justification = c(1, 0),
      legend.background = element_rect(
        fill = "white",
        color = "gray80",
        linewidth = 0.3
      )
    )
  
  
  # Save
  height <- max(6, nrow(df) * 0.3)
  ggsave(file.path(plot_dir, "Fig3_penalized_fg_coefficients.pdf"),
         p, width = 9, height = height, limitsize = FALSE)
  ggsave(file.path(plot_dir, "Fig3_penalized_fg_coefficients.png"),
         p, width = 9, height = height, dpi = 300, limitsize = FALSE)
  
  cat(sprintf("  Saved: Fig3_penalized_fg_coefficients.pdf/.png (%d CpGs)\n\n", 
              nrow(df)))
}

################################################################################
# FIGURE 4: FINAL FG MODEL VARIABLE IMPORTANCE
################################################################################
plot_fg_forest_hr <- function(plot_dir,
                              vimp_fg,
                              top_n = NULL,
                              filename = "Fig_final_model_forest") {
  
  stopifnot(is.data.frame(vimp_fg))
  required_cols <- c("feature", "coefficient", "se", "HR")
  missing <- setdiff(required_cols, colnames(vimp_fg))
  if (length(missing) > 0) {
    stop("Missing required columns: ", paste(missing, collapse = ", "))
  }
  
  # Calculate 95% CI for HR
  vimp_fg$HR_lower <- exp(vimp_fg$coefficient - 1.96 * vimp_fg$se)
  vimp_fg$HR_upper <- exp(vimp_fg$coefficient + 1.96 * vimp_fg$se)
  
  # Direction for coloring
  vimp_fg$direction <- ifelse(vimp_fg$HR > 1, "Risk ↑", "Protective ↓")
  
  # Select top features if requested
  if (!is.null(top_n)) {
    vimp_fg <- vimp_fg[order(abs(vimp_fg$coefficient), decreasing = TRUE), ]
    vimp_fg <- head(vimp_fg, top_n)
  }
  
  # Order factor for plotting
  vimp_fg$feature <- factor(vimp_fg$feature, levels = rev(vimp_fg$feature))
  
  # Create plot
  p <- ggplot(vimp_fg, aes(x = HR, y = feature, color = direction)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "gray50", linewidth = 0.8) +
    geom_point(size = 3) +
    geom_errorbarh(aes(xmin = HR_lower, xmax = HR_upper), height = 0.3, linewidth = 0.8) +
    scale_x_log10(
      breaks = c(0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4),
      labels = scales::number_format(accuracy = 0.01)
    ) +
    scale_color_manual(values = c("Risk ↑" = "#B2182B", "Protective ↓" = "#2166AC")) +
    labs(
      title = "Final Fine–Gray Model Variable Effects",
      subtitle = sprintf("Hazard ratios (95%% CI) for %d variables", nrow(vimp_fg)),
      x = "Subdistribution Hazard Ratio (log scale)",
      y = "",
      color = "Effect"
    ) +
    get_publication_theme() +
    theme(
      axis.text.y = element_text(size = 10),
      legend.position = "bottom"
    )
  
  # Save
  height <- max(4, nrow(vimp_fg) * 0.35)
  ggsave(file.path(plot_dir, paste0(filename, ".pdf")), p, width = 7, height = height)
  ggsave(file.path(plot_dir, paste0(filename, ".png")), p, width = 7, height = height, dpi = 300)
  
  cat(sprintf("Saved forest plot: %s (.pdf/.png) for %d variables\n", filename, nrow(vimp_fg)))
  invisible(p)
}


################################################################################
# FIGURE 5: TIME-DEPENDENT AUC
################################################################################

#' Plot Time-Dependent AUC
#'
#' Shows discrimination performance across all available timepoints.
#' Automatically adapts to cohort (TNBC: up to 5yr, ERpHER2n: up to 10yr).
#'
#' @param plot_dir Output directory
#' @param perf_results Performance from aggregate_cv_performance()
#' @param cohort Cohort name for title (optional)
#'
plot_time_dependent_auc <- function(plot_dir, perf_results, cohort = NULL) {
  
  cat("Creating Figure 5: Time-dependent AUC...\n")
  
  perf_summary <- perf_results$summary
  auc_metrics <- perf_summary[grep("^auc_", perf_summary$metric), ]
  
  if (nrow(auc_metrics) == 0) {
    cat("  WARNING: No AUC metrics. Skipping.\n\n")
    return(invisible(NULL))
  }
  
  # Extract time points
  auc_metrics$time <- as.numeric(gsub("auc_(\\d+)yr", "\\1", auc_metrics$metric))
  max_time <- max(auc_metrics$time)
  
  # Create subtitle
  subtitle <- sprintf("Discrimination across %d-year follow-up (mean ± 95%% CI)", max_time)
  if (!is.null(cohort)) {
    subtitle <- paste0(subtitle, sprintf(" - %s cohort", cohort))
  }
  
  # Create plot
  p <- ggplot(auc_metrics, aes(x = time, y = mean)) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), 
                alpha = 0.2, fill = "#2166AC") +
    geom_line(color = "#2166AC", linewidth = 1.3) +
    geom_point(color = "#2166AC", size = 3.5) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "red", 
               linewidth = 0.6, alpha = 0.7) +
    scale_x_continuous(breaks = unique(auc_metrics$time)) +
    scale_y_continuous(limits = c(0.4, 1), breaks = seq(0.4, 1, 0.1)) +
    labs(
      title = "Time-Dependent Discrimination",
      subtitle = subtitle,
      x = "Time Since Diagnosis (years)",
      y = "AUC",
      caption = "Red dashed line: random chance (AUC=0.5)"
    ) +
    get_publication_theme() +
    
    theme(
      plot.caption = element_text(hjust = 0, size = 9, color = "gray40")
    )
  
  # Save
  ggsave(file.path(plot_dir, "Fig5_time_dependent_auc.pdf"),
         p, width = 8, height = 6)
  ggsave(file.path(plot_dir, "Fig5_time_dependent_auc.png"),
         p, width = 8, height = 6, dpi = 300)
  
  cat(sprintf("  Saved: Fig5_time_dependent_auc.pdf/.png (%d timepoints)\n\n", 
              nrow(auc_metrics)))
}


################################################################################
# FIGURE 6: ROC CURVES AT KEY TIMEPOINTS
################################################################################

#' Plot ROC Curves at Key Timepoints
#'
#' Shows sensitivity vs specificity at requested years.
#' EVAL_TIMES is already cohort-specific (TNBC: 1-5, ERpHER2n: 1-10).
#'
#' @param plot_dir Output directory
#' @param cv_predictions Predictions across CV folds
#' @param requested_times Times to plot (from EVAL_TIMES)
#' @param cohort Cohort name for title (optional)
#'
plot_roc_curves <- function(plot_dir, 
                            cv_predictions,
                            requested_times = seq(1, 10),
                            cohort = NULL) {
  
  cat("Creating Figure 6: ROC curves...\n")
  
  if (is.null(cv_predictions) || nrow(cv_predictions) == 0) {
    cat("  WARNING: No predictions. Skipping.\n\n")
    return(invisible(NULL))
  }
  
  # Check pROC availability
  if (!require("pROC", quietly = TRUE)) {
    cat("  WARNING: pROC package not available. Skipping.\n\n")
    return(invisible(NULL))
  }
  
  # Determine available times
  available_cols <- grep("^risk_\\d+yr$", names(cv_predictions), value = TRUE)
  available_times <- as.numeric(gsub("risk_(\\d+)yr", "\\1", available_cols))
  
  # Use intersection of requested and available
  plot_times <- intersect(requested_times, available_times)
  
  if (length(plot_times) == 0) {
    cat("  WARNING: No requested timepoints available. Skipping.\n\n")
    return(invisible(NULL))
  }
  
  # Calculate ROC for each timepoint
  roc_list <- list()
  
  for (t in plot_times) {
    risk_col <- paste0("risk_", t, "yr")
    
    # Define outcome at time t
    cv_predictions$outcome_at_t <- ifelse(
      cv_predictions$time <= t & cv_predictions$rfi_event == 1, 1, 0
    )
    
    # Eligible patients
    eligible <- cv_predictions$time >= t | cv_predictions$rfi_event == 1
    data_t <- cv_predictions[eligible, ]
    
    # Check sufficient data
    n_cases <- sum(data_t$outcome_at_t == 1)
    n_controls <- sum(data_t$outcome_at_t == 0)
    
    if (n_cases < 5 || n_controls < 5) {
      cat(sprintf("  WARNING: Insufficient data at %dyr (cases=%d, controls=%d). Skipping.\n",
                  t, n_cases, n_controls))
      next
    }
    
    # Calculate ROC
    roc_obj <- pROC::roc(
      response = data_t$outcome_at_t,
      predictor = data_t[[risk_col]],
      levels = c(0, 1),
      direction = "<",
      quiet = TRUE
    )
    
    auc_val <- round(pROC::auc(roc_obj), 3)
    
    roc_list[[as.character(t)]] <- data.frame(
      time = sprintf("%d-year (AUC=%.3f)", t, auc_val),
      sensitivity = roc_obj$sensitivities,
      specificity = roc_obj$specificities,
      time_numeric = t
    )
  }
  
  if (length(roc_list) == 0) {
    cat("  WARNING: No ROC curves generated. Skipping.\n\n")
    return(invisible(NULL))
  }
  
  # Combine
  roc_df <- do.call(rbind, roc_list)
  roc_df$time <- factor(roc_df$time, levels = unique(roc_df$time))
  
  # Create subtitle
  subtitle <- "Recurrence prediction across CV folds"
  if (!is.null(cohort)) {
    subtitle <- paste0(subtitle, sprintf(" - %s cohort", cohort))
  }
  
  # Create plot
  p <- ggplot(roc_df, aes(x = 1 - specificity, y = sensitivity, color = time)) +
    geom_line(linewidth = 1.2) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", 
                color = "gray50", linewidth = 0.6) +
    scale_color_brewer(palette = "Set1", name = "Timepoint") +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(
      title = "ROC Curves at Key Timepoints",
      subtitle = subtitle,
      x = "1 - Specificity (False Positive Rate)",
      y = "Sensitivity (True Positive Rate)"
    ) +
    get_publication_theme() +
    theme(
      legend.position = c(0.7, 0.25),
      legend.background = element_rect(fill = "white", color = "gray80")
    )
  
  # Save
  ggsave(file.path(plot_dir, "Fig6_roc_curves.pdf"),
         p, width = 7, height = 7)
  ggsave(file.path(plot_dir, "Fig6_roc_curves.png"),
         p, width = 7, height = 7, dpi = 300)
  
  cat(sprintf("  Saved: Fig6_roc_curves.pdf/.png (%d timepoints)\n\n", 
              length(roc_list)))
}


################################################################################
# END OF PUBLICATION PLOTTING FUNCTIONS
################################################################################

plot_penalized_fg_coefficients_simple <- function(plot_dir, cpg_coefs) {
  
  cat("Creating Figure 3: Penalized FG CpG coefficients (MRS)...\n")
  
  # Convert to data frame
  df <- data.frame(
    feature = names(cpg_coefs),
    coefficient = as.numeric(cpg_coefs),
    stringsAsFactors = FALSE
  )
  
  # Order by coefficient
  df <- df[order(df$coefficient), ]
  df$feature <- factor(df$feature, levels = df$feature)
  
  # Axis limits
  max_abs <- max(abs(df$coefficient))
  axis_lim <- ceiling(max_abs * 1.1 * 10) / 10
  
  # Create plot
  p <- ggplot(df, aes(x = coefficient, y = feature)) +
    geom_vline(xintercept = 0, linetype = "dashed",
               color = "gray50", linewidth = 0.8) +
    geom_point(size = 3) +
    scale_x_continuous(limits = c(-axis_lim, axis_lim)) +
    labs(
      title = "Methylation Risk Score Components",
      subtitle = sprintf("Final Fine–Gray model CpGs (n=%d)", nrow(df)),
      x = "Coefficient",
      y = ""
    ) +
    get_publication_theme() +
    theme(
      axis.text.y = element_text(size = 8)
    )
  
  # Save
  height <- max(6, nrow(df) * 0.3)
  ggsave(file.path(plot_dir, "Fig3_penalized_fg_coefficients.pdf"),
         p, width = 9, height = height, limitsize = FALSE)
  ggsave(file.path(plot_dir, "Fig3_penalized_fg_coefficients.png"),
         p, width = 9, height = height, dpi = 300, limitsize = FALSE)
  
  cat(sprintf("  Saved: Fig3_penalized_fg_coefficients.pdf/.png (%d CpGs)\n\n", 
              nrow(df)))
}
