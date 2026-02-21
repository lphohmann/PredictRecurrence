################################################################################
#
# FINE-GRAY COMPETING RISKS - PUBLICATION PLOTTING FUNCTIONS
# Author: Lennart Hohmann
#
################################################################################
# LIBRARIES
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
      panel.grid.minor  = element_blank(),
      strip.background  = element_rect(fill = "white"),
      legend.position   = "right",
      plot.title        = element_text(face = "bold", size = 13),
      plot.subtitle     = element_text(size = 9.5, color = "gray30")
    )
}

################################################################################
# FIGURE: TIME-DEPENDENT AUC
################################################################################

plot_time_dependent_auc <- function(perf_results, models = NULL, cohort = NULL,
                                    legend = NULL, x_max = 10, plot_dir = NULL) {
  if (is.null(models)) models <- names(perf_results)
  
  auc_list <- lapply(models, function(m) {
    df     <- perf_results[[m]]$summary
    auc_df <- df[grep("^auc_", df$metric), ]
    auc_df$time  <- as.numeric(gsub("auc_(\\d+)yr", "\\1", auc_df$metric))
    auc_df$model <- m
    auc_df
  })
  auc_metrics <- do.call(rbind, auc_list)
  
  if (nrow(auc_metrics) == 0) {
    warning("No AUC metrics found.")
    return(invisible(NULL))
  }
  
  model_colors <- c(
    "FGR_MRS"      = "#2166AC",
    "FGR_CLIN"     = "#B2182B",
    "FGR_COMBINED" = "#1B9E77"
  )
  
  p <- ggplot(auc_metrics, aes(x = time, y = mean, color = model, fill = model)) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.15, color = NA) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 2.5) +
    geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray40", alpha = 0.7) +
    scale_x_continuous(breaks = 1:x_max, limits = c(1, x_max)) +
    scale_y_continuous(limits = c(0, 1.1), breaks = seq(0, 1, 0.1)) +
    scale_color_manual(values = model_colors) +
    scale_fill_manual(values = model_colors) +
    labs(
      title    = paste0(cohort, ": Time-Dependent Discrimination"),
      subtitle = sprintf("Discrimination across %d-year follow-up (mean ± 95%% CI)", max(auc_metrics$time)),
      x        = "Time Since Diagnosis (years)",
      y        = "Time-dependent AUC",
      color    = "Model",
      fill     = "Model"
    ) +
    get_publication_theme() +
    theme(
      plot.caption    = element_text(hjust = 0, size = 9, color = "gray40"),
      legend.position = if (is.null(legend)) "none" else "top"
    )
  
  if (!is.null(plot_dir)) {
    if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)
    ggsave(file.path(plot_dir, "time_dependent_auc.pdf"), p, width = 8, height = 5)
    ggsave(file.path(plot_dir, "time_dependent_auc.png"), p, width = 8, height = 5, dpi = 300)
    cat(sprintf("  Saved: time_dependent_auc.pdf/.png\n"))
  }
  
  invisible(p)
}

################################################################################
# FIGURE: PENALIZED FG CpG COEFFICIENTS (MRS COMPONENTS)
################################################################################

plot_penalized_fg_coefficients <- function(cpg_coefs, outer_fold_results,
                                               plot_dir = NULL, cohort = NULL,
                                               legend = NULL) {
  cat("Creating Penalized FG CpG coefficients (MRS) plot...\n")
  
  df <- data.frame(
    feature     = names(cpg_coefs),
    coefficient = as.numeric(cpg_coefs),
    stringsAsFactors = FALSE
  )
  
  rfi_cpgs   <- unique(unlist(lapply(outer_fold_results, function(f) f$coxnet$features_rfi)))
  death_cpgs <- unique(unlist(lapply(outer_fold_results, function(f) f$coxnet$features_death)))
  
  cat("RFI CpGs total:", length(rfi_cpgs), "\n")
  cat("Death CpGs total:", length(death_cpgs), "\n")
  cat("Features in df:", nrow(df), "\n")
  
  df$source <- "RFI CoxNet"
  df$source[df$feature %in% death_cpgs] <- "DWR CoxNet"
  df$source[df$feature %in% rfi_cpgs & df$feature %in% death_cpgs] <- "Both"
  df$source <- factor(df$source, levels = c("RFI CoxNet", "DWR CoxNet", "Both"))
  
  df          <- df[order(df$coefficient), ]
  df$feature  <- factor(df$feature, levels = df$feature)
  axis_lim    <- ceiling(max(abs(df$coefficient)) * 1.1 * 10) / 10
  
  source_colors <- c("RFI CoxNet" = "#D73027", "DWR CoxNet" = "#4575B4", "Both" = "#7B3294")
  
  p <- ggplot(df, aes(x = coefficient, y = feature, color = source)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 1) +
    geom_point(size = 6) +
    scale_x_continuous(limits = c(-axis_lim, axis_lim)) +
    scale_color_manual(values = source_colors, name = "Outer-CV CoxNet selection") +
    labs(
      title    = paste0(cohort, ": Ridge Fine-Gray Model"),
      subtitle = sprintf("CpGs (n=%d)", nrow(df)),
      x        = "Coefficient",
      y        = ""
    ) +
    get_publication_theme() +
    theme(
      axis.text.y      = element_text(size = 10),
      axis.text.x      = element_text(size = 10),
      axis.title.x     = element_text(size = 16),
      axis.title.y     = element_text(size = 16),
      plot.title       = element_text(size = 18, face = "bold"),
      plot.subtitle    = element_text(size = 16, color = "gray30"),
      legend.title     = element_text(size = 16, face = "bold"),
      legend.text      = element_text(size = 14),
      legend.position  = if (is.null(legend)) "none" else "bottom",
      legend.background = element_rect(fill = "white", color = "gray80", linewidth = 0.3)
    )
  
  if (!is.null(plot_dir)) {
    if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)
    height <- max(6, nrow(df) * 0.3)
    ggsave(file.path(plot_dir, "penalized_fg_coefficients.pdf"), p, width = 9, height = height, limitsize = FALSE)
    ggsave(file.path(plot_dir, "penalized_fg_coefficients.png"), p, width = 9, height = height, dpi = 300, limitsize = FALSE)
    cat(sprintf("  Saved: penalized_fg_coefficients.pdf/.png (%d CpGs)\n\n", nrow(df)))
  }
  
  invisible(p)
}

################################################################################
# FIGURE: CpG SELECTION STABILITY HEATMAP
################################################################################

plot_coxnet_stability_heatmap <- function(outer_fold_results, plot_dir = NULL,
                                              min_freq = 0.2, cohort = NULL,
                                              legend = NULL) {
  cat("Creating CoxNet pooled features stability heatmap...\n")
  
  completed  <- which(!sapply(outer_fold_results, is.null))
  n_folds    <- length(completed)
  
  all_features <- unique(unlist(lapply(completed, function(i) {
    outer_fold_results[[i]]$penalized_fg$input_cpgs
  })))
  
  if (length(all_features) == 0) {
    cat("  WARNING: No features selected. Skipping.\n\n")
    return(invisible(NULL))
  }
  
  sel_mat <- sapply(completed, function(i) {
    as.integer(all_features %in% outer_fold_results[[i]]$penalized_fg$input_cpgs)
  })
  rownames(sel_mat) <- all_features
  colnames(sel_mat) <- paste0("Fold", completed)
  
  sel_freq <- rowSums(sel_mat) / ncol(sel_mat)
  keep     <- sel_freq >= min_freq
  
  if (sum(keep) == 0) {
    cat(sprintf("  WARNING: No features with freq >= %.0f%%. Skipping.\n\n", min_freq * 100))
    return(invisible(NULL))
  }
  
  sel_mat  <- sel_mat[keep, , drop = FALSE]
  sel_freq <- sel_freq[keep]
  
  rfi_cpgs   <- unique(unlist(lapply(outer_fold_results, function(f) f$coxnet$features_rfi)))
  death_cpgs <- unique(unlist(lapply(outer_fold_results, function(f) f$coxnet$features_death)))
  
  source_df <- data.frame(Feature = rownames(sel_mat), source = "RFI CoxNet", stringsAsFactors = FALSE)
  source_df$source[source_df$Feature %in% death_cpgs] <- "DWR CoxNet"
  source_df$source[source_df$Feature %in% rfi_cpgs & source_df$Feature %in% death_cpgs] <- "Both"
  source_df$source <- factor(source_df$source, levels = c("RFI CoxNet", "DWR CoxNet", "Both"))
  
  feature_order <- names(sort(sel_freq, decreasing = TRUE))
  
  sel_df         <- reshape2::melt(sel_mat)
  colnames(sel_df) <- c("Feature", "Fold", "Selected")
  sel_df$Feature <- factor(sel_df$Feature, levels = rev(feature_order))
  sel_df         <- merge(sel_df, source_df, by = "Feature")
  
  sel_df$fill_val <- as.character(sel_df$source)
  sel_df$fill_val[sel_df$Selected == 0] <- "Not selected"
  sel_df$fill_val <- factor(sel_df$fill_val, levels = c("Not selected", "RFI CoxNet", "DWR CoxNet", "Both"))
  
  p <- ggplot(sel_df, aes(x = Fold, y = Feature, fill = fill_val)) +
    geom_tile(color = "white", linewidth = 0.5) +
    scale_fill_manual(
      values = c("Not selected" = "gray95", "RFI CoxNet" = "#D73027",
                 "DWR CoxNet" = "#4575B4", "Both" = "#7B3294"),
      name = "CoxNet selection"
    ) +
    labs(
      title    = paste0(cohort, ": nCV CpG Selection Stability"),
      subtitle = sprintf("Features selected ≥%.0f%% of folds (n=%d)", min_freq * 100, sum(keep)),
      x        = "Cross-Validation Fold",
      y        = ""
    ) +
    get_publication_theme() +
    theme(
      panel.grid       = element_blank(),
      legend.position  = if (is.null(legend)) "none" else "bottom",
      axis.text.y      = element_text(size = 8),
      axis.text.x      = element_text(size = 10),
      axis.title.x     = element_text(size = 16),
      plot.title       = element_text(size = 18, face = "bold"),
      plot.subtitle    = element_text(size = 16, color = "gray30"),
      legend.title     = element_text(size = 16, face = "bold"),
      legend.text      = element_text(size = 14),
      legend.background = element_rect(fill = "white", color = "gray80", linewidth = 0.3)
    )
  
  if (!is.null(plot_dir)) {
    if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)
    height <- max(6, sum(keep) * 0.2)
    ggsave(file.path(plot_dir, "coxnet_stability.pdf"), p, width = 10, height = height)
    ggsave(file.path(plot_dir, "coxnet_stability.png"), p, width = 10, height = height, dpi = 300)
    cat(sprintf("  Saved: coxnet_stability.pdf/.png (%d features)\n\n", sum(keep)))
  }
  
  invisible(p)
}

################################################################################
# FIGURE: FOREST PLOT - FINAL MODEL HAZARD RATIOS
################################################################################

plot_fg_forest_hr <- function(vimp_fg, plot_dir = NULL, top_n = NULL,
                                  filename = "Fig_final_model_forest", cohort = NULL) {
  stopifnot(is.data.frame(vimp_fg))
  required_cols <- c("feature", "coefficient", "se", "HR")
  missing <- setdiff(required_cols, colnames(vimp_fg))
  if (length(missing) > 0) stop("Missing required columns: ", paste(missing, collapse = ", "))
  
  vimp_fg$HR_lower <- exp(vimp_fg$coefficient - 1.96 * vimp_fg$se)
  vimp_fg$HR_upper <- exp(vimp_fg$coefficient + 1.96 * vimp_fg$se)
  
  if (!is.null(top_n)) {
    vimp_fg <- head(vimp_fg[order(abs(vimp_fg$coefficient), decreasing = TRUE), ], top_n)
  }
  
  vimp_fg$feature <- factor(vimp_fg$feature, levels = rev(vimp_fg$feature))
  
  p <- ggplot(vimp_fg, aes(x = HR, y = feature)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "gray50", linewidth = 0.8) +
    geom_point(size = 3) +
    geom_errorbarh(aes(xmin = HR_lower, xmax = HR_upper), height = 0.3, linewidth = 0.8) +
    scale_x_log10(
      breaks = c(0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4),
      labels = scales::number_format(accuracy = 0.01)
    ) +
    labs(
      title    = paste0(cohort, ": Fine-Gray Model Variable Effects"),
      subtitle = sprintf("Hazard ratios (95%% CI) for %d variables", nrow(vimp_fg)),
      x        = "Subdistribution Hazard Ratio (log scale)",
      y        = ""
    ) +
    get_publication_theme() +
    theme(
      axis.text.y     = element_text(size = 10),
      legend.position = "bottom"
    )
  
  if (!is.null(plot_dir)) {
    if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)
    height <- max(4, nrow(vimp_fg) * 0.35)
    ggsave(file.path(plot_dir, paste0(filename, ".pdf")), p, width = 7, height = height)
    ggsave(file.path(plot_dir, paste0(filename, ".png")), p, width = 7, height = height, dpi = 300)
    cat(sprintf("Saved: %s (.pdf/.png) for %d variables\n", filename, nrow(vimp_fg)))
  }
  
  invisible(p)
}