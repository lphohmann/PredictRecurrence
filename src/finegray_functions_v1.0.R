################################################################################
# HELPER FUNCTIONS FOR FINE-GRAY COMPETING RISKS PIPELINE
################################################################################

################################################################################
# DATA LOADING AND PREPARATION
################################################################################

load_training_data <- function(train_ids, beta_path, clinical_path) {
  # Load and prepare clinical and methylation data for specified sample IDs
  #
  # Args:
  #   train_ids: Vector of sample IDs to load
  #   beta_path: Path to methylation beta values CSV
  #   clinical_path: Path to clinical data CSV
  #
  # Returns:
  #   List with beta_matrix (samples x CpGs) and clinical_data
  
  # Load clinical data
  clinical_data <- read_csv(clinical_path, show_col_types = FALSE)
  clinical_data <- as.data.frame(clinical_data)
  rownames(clinical_data) <- clinical_data$Sample
  clinical_data <- clinical_data[train_ids, , drop = FALSE]
  
  # Load methylation data
  cat("Loading methylation data...\n")
  beta_matrix <- fread(beta_path, header = TRUE, data.table = FALSE)
  rownames(beta_matrix) <- beta_matrix[[1]]
  beta_matrix <- beta_matrix[, -1]
  
  # Transpose to samples x CpGs format
  cat("Transposing...\n")
  beta_matrix <- t(beta_matrix)
  beta_matrix <- beta_matrix[train_ids, , drop = FALSE]
  beta_matrix <- as.data.frame(beta_matrix)
  
  cat(sprintf("Loaded %d samples x %d CpGs\n", nrow(beta_matrix), ncol(beta_matrix)))
  
  return(list(beta_matrix = beta_matrix, clinical_data = clinical_data))
}

################################################################################

beta_to_m <- function(beta, beta_threshold = 1e-3) {
  # Convert beta values to M-values using logit transformation
  #
  # M-values are more appropriate for linear modeling:
  #   - Beta values: bounded [0,1], heteroscedastic
  #   - M-values: unbounded, approximately normal, homoscedastic
  #
  # Formula: M = log2(beta / (1 - beta))
  #
  # Args:
  #   beta: Matrix or dataframe of beta values
  #   beta_threshold: Minimum/maximum threshold to prevent log(0)
  #
  # Returns:
  #   M-values in same format as input
  
  # Preserve structure
  was_df <- is.data.frame(beta)
  if (was_df) {
    row_names <- rownames(beta)
    col_names <- colnames(beta)
    beta <- as.matrix(beta)
  }
  
  # Cap beta values to prevent log(0) or log(inf)
  beta <- pmax(pmin(beta, 1 - beta_threshold), beta_threshold)
  
  # Apply logit transformation
  m_values <- log2(beta / (1 - beta))
  
  # Restore structure
  if (was_df) {
    m_values <- as.data.frame(m_values)
    rownames(m_values) <- row_names
    colnames(m_values) <- col_names
  }
  
  return(m_values)
}

################################################################################

apply_admin_censoring <- function(df, time_col, event_col, time_cutoff) {
  # Apply administrative censoring at specified time point
  #
  # Patients with follow-up exceeding cutoff are censored at cutoff.
  # This standardizes follow-up time across cohorts (e.g., 5 years for TNBC).
  #
  # Args:
  #   df: Dataframe with time and event columns
  #   time_col: Name of time column
  #   event_col: Name of event column (0/1)
  #   time_cutoff: Time at which to censor
  #
  # Returns:
  #   Modified dataframe with censoring applied
  
  mask <- df[[time_col]] > time_cutoff
  df[mask, time_col] <- time_cutoff
  df[mask, event_col] <- 0
  
  cat(sprintf("Applied censoring at %.1f for %s/%s (n=%d)\n", 
              time_cutoff, time_col, event_col, sum(mask)))
  
  return(df)
}

################################################################################

subset_methylation <- function(mval_matrix, cpg_ids_file) {
  # Subset methylation data to predefined CpG sites
  #
  # Useful for focusing on biologically relevant CpGs (e.g., from ATAC-seq)
  #
  # Args:
  #   mval_matrix: Matrix of M-values (samples x CpGs)
  #   cpg_ids_file: Path to text file with CpG IDs (one per line)
  #
  # Returns:
  #   Filtered matrix containing only specified CpGs
  
  cpg_ids <- trimws(readLines(cpg_ids_file))
  cpg_ids <- cpg_ids[cpg_ids != ""]
  valid_cpgs <- cpg_ids[cpg_ids %in% colnames(mval_matrix)]
  
  cat(sprintf("Subsetted to %d CpGs (from %d in file)\n", 
              length(valid_cpgs), length(cpg_ids)))
  
  return(mval_matrix[, valid_cpgs, drop = FALSE])
}

################################################################################

onehot_encode_clinical <- function(clin, clin_categorical) {
  # One-hot encode categorical clinical variables
  #
  # Creates k-1 dummy variables for each categorical variable with k levels.
  # Reference level is omitted to avoid multicollinearity.
  # Example: NHG with levels 1,2,3 → NHG2, NHG3 (NHG1 is reference)
  #
  # Args:
  #   clin: Dataframe with clinical variables
  #   clin_categorical: Vector of categorical variable names
  #
  # Returns:
  #   List with:
  #     - encoded_df: Dataframe with continuous + one-hot encoded variables
  #     - encoded_cols: Vector of one-hot encoded column names
  
  # Set factor levels for categorical variables
  for (var in clin_categorical) {
    if (var == "LN") {
      # Ensure N0 is reference, include any other levels present
      all_levels <- unique(as.character(clin[[var]]))
      ref_first <- c("N0", setdiff(all_levels, "N0"))
      clin[[var]] <- factor(clin[[var]], levels = ref_first)
    } else if (var == "NHG") {
      # Use natural ordering (1, 2, 3, ...)
      all_levels <- sort(unique(as.character(clin[[var]])))
      clin[[var]] <- factor(clin[[var]], levels = all_levels)
    } else {
      clin[[var]] <- as.factor(clin[[var]])
    }
  }
  
  # Keep continuous variables as-is
  continuous_vars <- setdiff(colnames(clin), clin_categorical)
  clin_encoded <- if (length(continuous_vars) > 0) {
    clin[, continuous_vars, drop = FALSE]
  } else {
    data.frame(row.names = rownames(clin))
  }
  
  # Create dummy variables for each categorical variable
  encoded_cols <- c()
  for (var in clin_categorical) {
    dummy_df <- as.data.frame(model.matrix(
      as.formula(paste("~", var)),
      data = clin
    ))
    
    # Remove intercept column
    dummy_df <- dummy_df[, -1, drop = FALSE]
    rownames(dummy_df) <- rownames(clin)
    
    encoded_cols <- c(encoded_cols, colnames(dummy_df))
    clin_encoded <- cbind(clin_encoded, dummy_df)
  }
  
  return(list(
    encoded_df = clin_encoded, 
    encoded_cols = encoded_cols
  ))
}

################################################################################
# FEATURE SELECTION AND PREPROCESSING
################################################################################

filter_by_variance <- function(X_train, variance_quantile = 0.75) {
  # Filter features by variance, keeping the most variable
  #
  # Reduces dimensionality by removing low-variance features that are
  # unlikely to be informative. Must be done on training data only.
  #
  # Args:
  #   X_train: Training data matrix (samples x features)
  #   variance_quantile: Quantile threshold (0.75 = keep top 25%)
  #
  # Returns:
  #   Filtered matrix with high-variance features only
  
  # Input validation
  if (ncol(X_train) == 0) {
    stop("X_train has no columns to filter")
  }
  if (variance_quantile < 0 || variance_quantile > 1) {
    stop("variance_quantile must be between 0 and 1")
  }
  
  # Calculate variance for each feature
  variances <- apply(X_train, 2, var, na.rm = TRUE)
  
  # Handle features with NA variance (all NA or constant)
  if (any(is.na(variances))) {
    n_na <- sum(is.na(variances))
    warning(sprintf("%d features have NA variance (all NA or constant). Removing them.", n_na))
    variances[is.na(variances)] <- -Inf  # Will be filtered out
  }
  
  # Find threshold and filter
  var_threshold <- quantile(variances, probs = variance_quantile, na.rm = TRUE)
  keep_var <- names(variances[variances >= var_threshold])
  X_filtered <- X_train[, keep_var, drop = FALSE]
  
  cat(sprintf("After variance filter (>= %.0fth percentile): %d features\n", 
              variance_quantile * 100, length(keep_var)))
  
  return(X_filtered)
}

################################################################################

prepare_filtered_features <- function(X, vars_preserve, variance_quantile = 0.75) {
  # Apply variance filtering while preserving specified variables
  #
  # Clinical variables are typically preserved to ensure they remain
  # in the model regardless of variance.
  #
  # Args:
  #   X: Feature matrix (samples x features)
  #   vars_preserve: Vector of variable names to preserve
  #   variance_quantile: Variance filter threshold
  #
  # Returns:
  #   Filtered feature matrix with preserved variables included
  
  # Separate preserved vs filterable features
  is_preserved <- colnames(X) %in% vars_preserve
  X_preserved <- X[, is_preserved, drop = FALSE]
  X_to_filter <- X[, !is_preserved, drop = FALSE]
  
  cat(sprintf("Features before filtering: %d (preserved: %d)\n", 
              ncol(X), sum(is_preserved)))
  
  # Apply variance filtering to non-preserved features
  X_filtered <- filter_by_variance(X_to_filter, variance_quantile = variance_quantile)
  
  # Combine filtered and preserved features
  X_final <- cbind(X_filtered, X_preserved)
  
  cat(sprintf("Total features for modeling: %d\n", ncol(X_final)))
  
  return(X_final)
}

################################################################################

scale_continuous_features <- function(X_train, X_test = NULL, dont_scale) {
  # Standardize continuous features using training set parameters
  #
  # One-hot encoded categorical variables are left unscaled.
  # Test set is scaled using TRAINING set mean/SD to prevent data leakage.
  #
  # Args:
  #   X_train: Training feature matrix
  #   X_test: Test feature matrix (optional)
  #   dont_scale: Vector of column names to not scale (e.g., one-hot encoded)
  #
  # Returns:
  #   List with:
  #     - X_train_scaled: Scaled training data
  #     - X_test_scaled: Scaled test data (NULL if X_test not provided)
  #     - centers: Mean values from training data
  #     - scales: SD values from training data
  #     - cont_cols: Names of scaled columns
  
  # Identify continuous columns
  cont_cols <- setdiff(colnames(X_train), dont_scale)
  
  # Scale continuous variables and extract parameters
  train_cont_scaled <- scale(X_train[, cont_cols, drop = FALSE])
  centers <- attr(train_cont_scaled, "scaled:center")
  scales  <- attr(train_cont_scaled, "scaled:scale")
  
  # Apply to training set
  X_train_scaled <- X_train
  X_train_scaled[, cont_cols] <- train_cont_scaled
  
  # Apply to test set using TRAINING parameters (prevents data leakage)
  if (!is.null(X_test)) {
    X_test_scaled <- X_test
    X_test_scaled[, cont_cols] <- scale(
      X_test[, cont_cols, drop = FALSE],
      center = centers,
      scale  = scales
    )
  } else {
    X_test_scaled <- NULL
  }
  
  cat(sprintf("Scaled %d continuous features\n", length(cont_cols)))
  
  return(list(
    X_train_scaled = X_train_scaled,
    X_test_scaled = X_test_scaled,
    centers = centers,
    scales = scales,
    cont_cols = cont_cols
  ))
}

################################################################################
# ELASTIC NET COX REGRESSION
################################################################################

cv_glmnet_alpha_grid <- function(X_train, y_train, alpha_grid, penalty_factor = NULL, 
                                 foldid = NULL, family = "cox", seed = 123) {
  # Cross-validate glmnet across multiple alpha values
  #
  # Alpha controls elastic net mixing:
  #   - alpha = 0: Ridge regression (L2 penalty)
  #   - alpha = 1: LASSO (L1 penalty)
  #   - 0 < alpha < 1: Elastic net (combined L1/L2)
  #
  # Args:
  #   X_train: Training feature matrix
  #   y_train: Survival object (from Surv())
  #   alpha_grid: Vector of alpha values to test
  #   penalty_factor: Feature-specific penalties (0 = no penalty)
  #   foldid: CV fold assignments
  #   family: "cox" for survival analysis
  #   seed: Random seed for reproducibility
  #
  # Returns:
  #   List of CV results for each alpha value
  
  cv_results <- list()
  
  for (alpha_val in alpha_grid) {
    set.seed(seed)
    cv_fit <- cv.glmnet(
      x = as.matrix(X_train),
      y = y_train,
      family = family,
      alpha = alpha_val,
      penalty.factor = penalty_factor,
      foldid = foldid,
      standardize = TRUE
    )
    
    # Extract performance at optimal lambda
    perf_min <- cv_fit$cvm[cv_fit$lambda == cv_fit$lambda.min]
    perf_se <- cv_fit$cvsd[cv_fit$lambda == cv_fit$lambda.min]
    
    # Get selected features
    beta <- coef(cv_fit, s = "lambda.min")
    selected_features <- rownames(beta)[as.vector(beta != 0)]
    
    cv_results[[as.character(alpha_val)]] <- list(
      alpha = alpha_val,
      lambda = cv_fit$lambda.min,
      cvm_min = perf_min,
      cvm_se = perf_se,
      n_features = length(selected_features),
      features = selected_features,
      fit = cv_fit
    )
    
    cat(sprintf(
      "  Alpha=%.2f: CV-deviance=%.4f (SE=%.4f), Lambda=%.6f, Features=%d\n",
      alpha_val, perf_min, perf_se, cv_fit$lambda.min, length(selected_features)
    ))
  }
  
  return(cv_results)
}

################################################################################

tune_and_fit_coxnet <- function(X_train, y_train, clinical_train, event_col,
                                alpha_grid, penalty_factor = NULL, 
                                n_inner_folds = 5, seed = 123,
                                outcome_name = "outcome") {
  # Tune CoxNet hyperparameters via inner CV, then fit final model
  #
  # Performs stratified cross-validation on event occurrence to find
  # optimal alpha and lambda values, then fits model on full training set.
  #
  # Args:
  #   X_train: Training feature matrix
  #   y_train: Survival object
  #   clinical_train: Clinical data for stratification
  #   event_col: Column name for event indicator
  #   alpha_grid: Vector of alpha values to test
  #   penalty_factor: Feature-specific penalties
  #   n_inner_folds: Number of CV folds
  #   seed: Random seed
  #   outcome_name: Name for logging
  #
  # Returns:
  #   List with CV results, best hyperparameters, and fitted model
  
  cat(sprintf("\n--- Inner CV: Tuning CoxNet for %s ---\n", outcome_name))
  
  # Create stratified inner folds based on event occurrence
  set.seed(seed)
  inner_folds <- createFolds(
    y = clinical_train[[event_col]],
    k = n_inner_folds,
    list = TRUE,
    returnTrain = FALSE
  )
  
  # Convert to foldid vector for cv.glmnet
  foldid <- rep(NA, nrow(X_train))
  for (i in 1:n_inner_folds) {
    foldid[inner_folds[[i]]] <- i
  }
  
  # Test each alpha value
  cv_results <- cv_glmnet_alpha_grid(
    X_train = X_train,
    y_train = y_train,
    alpha_grid = alpha_grid,
    penalty_factor = penalty_factor,
    foldid = foldid
  )
  
  # Select best hyperparameters based on CV deviance
  best_idx <- which.min(sapply(cv_results, function(x) x$cvm_min))
  best_result <- cv_results[[best_idx]]
  best_alpha <- best_result$alpha
  best_lambda <- best_result$lambda
  
  cat(sprintf("Best hyperparameters: alpha=%.2f, lambda=%.6f\n", 
              best_alpha, best_lambda))
  
  # Fit final model on whole training set with best hyperparameters
  final_fit <- glmnet(
    x = as.matrix(X_train),
    y = y_train,
    family = "cox",
    alpha = best_alpha,
    lambda = best_lambda,
    penalty.factor = penalty_factor
  )
  
  return(list(
    cv_results = cv_results,
    best_result = best_result,
    best_alpha = best_alpha,
    best_lambda = best_lambda,
    final_fit = final_fit
  ))
}

################################################################################

extract_nonzero_coefs <- function(model_fit, sort_by_abs = TRUE) {
  # Extract non-zero coefficients from fitted glmnet model
  #
  # Args:
  #   model_fit: Fitted glmnet model object
  #   sort_by_abs: Sort by absolute coefficient value (default TRUE)
  #
  # Returns:
  #   List with:
  #     - coef_df: Dataframe with feature names and coefficients
  #     - features: Vector of selected feature names
  
  coef_matrix <- coef(model_fit)
  
  # Filter to non-zero coefficients
  nonzero_mask <- as.vector(coef_matrix != 0)
  
  coef_df <- data.frame(
    feature = rownames(coef_matrix)[nonzero_mask],
    coefficient = as.vector(coef_matrix)[nonzero_mask]
  )
  
  # Sort by absolute coefficient value
  if (sort_by_abs) {
    coef_df <- coef_df[order(abs(coef_df$coefficient), decreasing = TRUE), ]
  }
  
  rownames(coef_df) <- NULL
  
  return(list(
    coef_df = coef_df,
    features = coef_df$feature
  ))
}

################################################################################
# PERFORMANCE AGGREGATION
################################################################################

aggregate_cv_performance <- function(outer_fold_results, 
                                     round_digits = 3, 
                                     conf_level = 0.95,
                                     verbose = TRUE) {
  # Aggregate performance metrics across CV folds
  #
  # Calculates mean, SE, SD, and confidence intervals for all metrics.
  #
  # Args:
  #   outer_fold_results: List of fold results from outer CV loop
  #   round_digits: Number of decimal places for rounding
  #   conf_level: Confidence level for intervals (default 0.95)
  #   verbose: Print summary to console
  #
  # Returns:
  #   List with:
  #     - summary: Dataframe with aggregated statistics
  #     - all_folds: Dataframe with individual fold results
  
  if (verbose) cat("\n========== AGGREGATED PERFORMANCE ==========\n")
  
  # Extract performance dataframes from all folds
  performance_list <- lapply(outer_fold_results, function(fold) fold$performance_df)
  all_performance <- do.call(rbind, performance_list)
  
  # Get metric column names (exclude 'model' column)
  metric_cols <- setdiff(names(all_performance), "model")
  
  # Calculate summary statistics for each metric
  performance_summary <- data.frame(
    metric = metric_cols,
    mean = sapply(metric_cols, function(col) mean(all_performance[[col]], na.rm = TRUE)),
    se = sapply(metric_cols, function(col) {
      sd(all_performance[[col]], na.rm = TRUE) / sqrt(nrow(all_performance))
    }),
    sd = sapply(metric_cols, function(col) sd(all_performance[[col]], na.rm = TRUE))
  )
  
  # Round to specified decimal places
  performance_summary[, c("mean", "se", "sd")] <- 
    round(performance_summary[, c("mean", "se", "sd")], round_digits)
  
  # Add confidence intervals
  z_score <- qnorm(1 - (1 - conf_level) / 2)
  performance_summary$ci_lower <- round(
    performance_summary$mean - z_score * performance_summary$se, round_digits
  )
  performance_summary$ci_upper <- round(
    performance_summary$mean + z_score * performance_summary$se, round_digits
  )
  
  rownames(performance_summary) <- NULL
  
  # Print results
  if (verbose) {
    cat("\nIndividual Fold Performance:\n")
    print(all_performance)
    
    cat(sprintf("\nPerformance Summary (Mean ± SE, %d%% CI):\n", conf_level * 100))
    for (i in 1:nrow(performance_summary)) {
      cat(sprintf("%15s: %.3f ± %.3f (95%% CI: %.3f - %.3f)\n",
                  performance_summary$metric[i],
                  performance_summary$mean[i],
                  performance_summary$se[i],
                  performance_summary$ci_lower[i],
                  performance_summary$ci_upper[i]))
    }
  }
  
  return(list(
    summary = performance_summary,
    all_folds = all_performance
  ))
}

################################################################################
# FEATURE SELECTION STABILITY
################################################################################

assess_feature_stability <- function(outer_fold_results, 
                                     min_folds = NULL,
                                     verbose = TRUE) {
  # Assess feature selection stability across CV folds
  #
  # Calculates selection frequency and coefficient consistency for features
  # selected by CoxNet (RFI and Death) and Fine-Gray models.
  #
  # Args:
  #   outer_fold_results: List of fold results from outer CV loop
  #   min_folds: Minimum folds for "stable" classification (default: majority)
  #   verbose: Print summary to console
  #
  # Returns:
  #   List with stability dataframes for each model type
  
  n_folds <- length(outer_fold_results)
  
  # Default: majority of folds
  if (is.null(min_folds)) {
    min_folds <- ceiling(n_folds / 2)
  }
  
  stability_threshold <- min_folds / n_folds
  
  if (verbose) cat("\n========== FEATURE SELECTION STABILITY ==========\n")
  
  # Extract coefficient tables from all folds
  coef_list <- lapply(outer_fold_results, function(fold) fold$fold_model_coefficients)
  
  # Get all unique features across all folds
  all_features <- unique(unlist(lapply(coef_list, function(df) df$feature)))
  
  # Calculate stability for each model type
  stability_rfi <- calculate_stability_metrics(
    all_features, coef_list, "cox_rfi_coef", n_folds
  )
  stability_death <- calculate_stability_metrics(
    all_features, coef_list, "cox_death_coef", n_folds
  )
  stability_fg <- calculate_stability_metrics(
    all_features, coef_list, "fg_coef", n_folds
  )
  
  # Print summary
  if (verbose) {
    cat(sprintf("\nTotal folds: %d\n", n_folds))
    cat(sprintf("Stability threshold: ≥%d folds (%.0f%%)\n\n", 
                min_folds, stability_threshold * 100))
    
    for (model_name in c("CoxNet RFI", "CoxNet Death", "Fine-Gray")) {
      stability_df <- switch(model_name,
                             "CoxNet RFI" = stability_rfi,
                             "CoxNet Death" = stability_death,
                             "Fine-Gray" = stability_fg)
      
      cat(sprintf("--- %s ---\n", model_name))
      cat(sprintf("Total features ever selected: %d\n", 
                  sum(stability_df$n_selected > 0)))
      cat(sprintf("Stable features (≥%d/%d folds): %d\n", 
                  min_folds, n_folds,
                  sum(stability_df$n_selected >= min_folds)))
      
      # Show stable features
      stable_features <- stability_df[stability_df$n_selected >= min_folds, ]
      if (nrow(stable_features) > 0) {
        cat("\nStable features:\n")
        print(stable_features)
      }
      cat("\n")
    }
  }
  
  return(list(
    cox_rfi_stability = stability_rfi,
    cox_death_stability = stability_death,
    finegray_stability = stability_fg,
    n_folds = n_folds,
    min_folds = min_folds
  ))
}

################################################################################

calculate_stability_metrics <- function(all_features, coef_list, coef_col_name, n_folds) {
  # Calculate stability metrics for one model type
  #
  # For each feature, computes:
  #   - Selection frequency across folds
  #   - Mean and SD of non-zero coefficients
  #   - Direction consistency (all positive or all negative)
  #
  # Args:
  #   all_features: Vector of all unique feature names
  #   coef_list: List of coefficient dataframes from each fold
  #   coef_col_name: Column name containing coefficients
  #   n_folds: Total number of folds
  #
  # Returns:
  #   Dataframe with stability metrics, sorted by selection frequency
  
  feature_stats <- lapply(all_features, function(feat) {
    
    # Extract coefficients for this feature across all folds
    coefs <- sapply(coef_list, function(df) {
      idx <- which(df$feature == feat)
      if (length(idx) == 0) return(0)
      df[[coef_col_name]][idx]
    })
    
    # Count selections and calculate frequency
    n_selected <- sum(coefs != 0)
    selection_freq <- n_selected / n_folds
    
    # Count direction
    n_positive <- sum(coefs > 0)
    n_negative <- sum(coefs < 0)
    
    # Calculate statistics for non-zero coefficients
    nonzero_coefs <- coefs[coefs != 0]
    if (length(nonzero_coefs) > 0) {
      mean_coef <- mean(nonzero_coefs)
      sd_coef <- sd(nonzero_coefs)
      
      # Check direction consistency
      all_positive <- all(nonzero_coefs > 0)
      all_negative <- all(nonzero_coefs < 0)
      direction_consistent <- all_positive || all_negative
    } else {
      mean_coef <- 0
      sd_coef <- 0
      direction_consistent <- NA
    }
    
    data.frame(
      feature = feat,
      n_selected = n_selected,
      selection_freq = selection_freq,
      n_positive = n_positive,
      n_negative = n_negative,
      direction_consistent = direction_consistent,
      mean_coef = mean_coef,
      sd_coef = sd_coef
    )
  })
  
  # Combine and sort by selection frequency
  stability_df <- do.call(rbind, feature_stats)
  stability_df <- stability_df[order(stability_df$selection_freq, decreasing = TRUE), ]
  rownames(stability_df) <- NULL
  
  # Round numeric columns
  stability_df$selection_freq <- round(stability_df$selection_freq, 3)
  stability_df$mean_coef <- round(stability_df$mean_coef, 3)
  stability_df$sd_coef <- round(stability_df$sd_coef, 3)
  
  return(stability_df)
}

################################################################################
# FINE-GRAY VARIABLE IMPORTANCE
################################################################################

calculate_fgr_importance <- function(fgr_model, encoded_cols = NULL, 
                                     top_n = NULL, verbose = TRUE) {
  # Calculate variable importance for Fine-Gray competing risks model
  #
  # Computes hazard ratios, Wald statistics, and p-values from model
  # coefficients and variance-covariance matrix.
  #
  # Args:
  #   fgr_model: Fitted Fine-Gray model (from FGR())
  #   encoded_cols: Names of one-hot encoded columns for type annotation
  #   top_n: Limit output to top N features (default: all)
  #   verbose: Print detailed summary
  #
  # Returns:
  #   Dataframe with importance metrics sorted by statistical significance
  
  # Extract coefficients and variance-covariance matrix
  fg_coef <- fgr_model$crrFit$coef
  fg_vcov <- fgr_model$crrFit$var
  fg_se <- sqrt(diag(fg_vcov))
  
  # Create comprehensive importance table
  vimp <- data.frame(
    feature = names(fg_coef),
    coefficient = as.vector(fg_coef),
    se = fg_se,
    HR = exp(fg_coef),
    abs_HR = exp(abs(fg_coef)),
    wald_z = fg_coef / fg_se,
    abs_wald_z = abs(fg_coef / fg_se),
    p_value = pchisq((fg_coef / fg_se)^2, df = 1, lower.tail = FALSE),
    stringsAsFactors = FALSE
  )
  
  # Annotate variable type if provided
  if (!is.null(encoded_cols)) {
    vimp$type <- ifelse(vimp$feature %in% encoded_cols, "categorical", "continuous")
  }
  
  # Sort by statistical significance (absolute Wald z-score)
  vimp <- vimp[order(vimp$abs_wald_z, decreasing = TRUE), ]
  rownames(vimp) <- NULL
  
  # Limit to top N if specified
  if (!is.null(top_n)) {
    vimp <- head(vimp, top_n)
  }
  
  # Round for readability
  vimp$coefficient <- round(vimp$coefficient, 3)
  vimp$se <- round(vimp$se, 3)
  vimp$HR <- round(vimp$HR, 2)
  vimp$abs_HR <- round(vimp$abs_HR, 2)
  vimp$wald_z <- round(vimp$wald_z, 2)
  vimp$abs_wald_z <- round(vimp$abs_wald_z, 2)
  vimp$p_value <- round(vimp$p_value, 4)
  
  # Print detailed summaries
  if (verbose) {
    cat("\n--- Fine-Gray Variable Importance ---\n")
    cat("\nVariable Importance (sorted by statistical significance):\n")
    print(vimp)
    
    cat("\n--- Summary Views ---\n")
    
    # Top features by effect size
    cat("\nTop 10 by effect size (absolute HR):\n")
    vimp_by_hr <- vimp[order(vimp$abs_HR, decreasing = TRUE), ]
    print(head(vimp_by_hr[, c("feature", "coefficient", "HR", "abs_HR")], 10))
    
    # Separate by variable type
    if (!is.null(encoded_cols)) {
      cat("\n--- By Variable Type ---\n")
      
      vimp_cat <- vimp[vimp$type == "categorical", ]
      if (nrow(vimp_cat) > 0) {
        cat("\nCategorical (vs reference):\n")
        print(vimp_cat[, c("feature", "HR", "wald_z", "p_value")])
      }
      
      vimp_cont <- vimp[vimp$type == "continuous", ]
      if (nrow(vimp_cont) > 0) {
        cat("\nContinuous (per 1 SD):\n")
        print(head(vimp_cont[, c("feature", "HR", "wald_z", "p_value")], 10))
      }
    }
  }
  
  return(vimp)
}