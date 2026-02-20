################################################################################
#
# FINE-GRAY COMPETING RISKS ANALYSIS - HELPER FUNCTIONS
#
# Purpose: Core functions for Fine-Gray competing risks modeling pipeline
#          including data loading, preprocessing, feature selection, model
#          fitting, and performance evaluation
#
################################################################################

################################################################################
# TABLE OF CONTENTS
################################################################################
#
# 1. DATA LOADING AND PREPARATION
#    - load_training_data()
#    - beta_to_m()
#    - apply_admin_censoring()
#    - subset_methylation()
#    - onehot_encode_clinical()
#
# 2. FEATURE SELECTION AND PREPROCESSING
#    - filter_by_variance()
#    - filter_by_univariate_cox()
#    - prepare_filtered_features()
#    - cv_data_prep()
#    - scale_continuous_features()
#
# 3. ELASTIC NET COX REGRESSION
#    - cv_glmnet_alpha_grid()
#    - tune_and_fit_coxnet()
#    - extract_nonzero_coefs()
#
# 4. PENALIZED FINE-GRAY MODELING
#    - evaluate_alpha_lambda_cv()
#    - fit_penalized_finegray_cv()
#    - calculate_methylation_risk_score()
#
# 5. FINE-GRAY MODEL FITTING AND EVALUATION
#    - fit_fine_gray_model()
#    - calculate_fgr_importance()
#
# 6. PERFORMANCE AGGREGATION AND STABILITY
#    - aggregate_cv_performance()
#    - assess_finegray_stability()
#
################################################################################


################################################################################
# 1. DATA LOADING AND PREPARATION
################################################################################

#' Load Training Data
#'
#' Load and prepare clinical and methylation data for specified sample IDs.
#' Handles both clinical metadata and methylation beta values, ensuring proper
#' sample alignment.
#'
#' @param train_ids Vector of sample IDs to load
#' @param beta_path Path to methylation beta values CSV file
#' @param clinical_path Path to clinical data CSV file
#'
#' @return List containing:
#'   \item{beta_matrix}{Matrix of beta values (samples x CpGs)}
#'   \item{clinical_data}{Data frame of clinical variables}
#'
#' @details
#' - Clinical data must have a 'Sample' column for matching
#' - Methylation data is transposed to samples x CpGs format
#' - Samples are subset to match train_ids exactly
#'
#' @export
load_training_data <- function(train_ids, beta_path, clinical_path) {
  
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
  
  return(list(
    beta_matrix = beta_matrix, 
    clinical_data = clinical_data
  ))
}


#' Convert Beta Values to M-Values
#'
#' Transform methylation beta values to M-values using logit transformation.
#' M-values are more appropriate for linear modeling as they are unbounded,
#' approximately normally distributed, and homoscedastic.
#'
#' @param beta Matrix or data frame of beta values (bounded [0,1])
#' @param beta_threshold Minimum/maximum threshold to prevent log(0) (default: 1e-3)
#'
#' @return M-values in same format as input
#'
#' @details
#' Formula: M = log2(beta / (1 - beta))
#' 
#' Beta values are capped at [beta_threshold, 1-beta_threshold] to prevent
#' infinite values from log(0) or log(inf).
#'
#' @export
beta_to_m <- function(beta, beta_threshold = 1e-3) {
  
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


#' Apply Administrative Censoring
#'
#' Censor patients with follow-up exceeding a specified cutoff time.
#' This standardizes follow-up time across cohorts (e.g., 5 years for TNBC).
#'
#' @param df Data frame with time and event columns
#' @param time_col Name of time column
#' @param event_col Name of event column (0 = censored, 1 = event)
#' @param time_cutoff Time at which to apply censoring
#'
#' @return Modified data frame with censoring applied
#'
#' @details
#' Patients with time > time_cutoff are censored at time_cutoff and their
#' event indicator is set to 0.
#'
#' @export
apply_admin_censoring <- function(df, time_col, event_col, time_cutoff) {
  
  mask <- df[[time_col]] > time_cutoff
  df[mask, time_col] <- time_cutoff
  df[mask, event_col] <- 0
  
  cat(sprintf("Applied censoring at %.1f for %s/%s (n=%d)\n", 
              time_cutoff, time_col, event_col, sum(mask)))
  
  return(df)
}


#' Subset Methylation Data to Specific CpGs
#'
#' Filter methylation matrix to only include predefined CpG sites.
#' Useful for focusing on biologically relevant CpGs (e.g., from ATAC-seq).
#'
#' @param mval_matrix Matrix of M-values (samples x CpGs)
#' @param cpg_ids_file Path to text file with CpG IDs (one per line)
#'
#' @return Filtered matrix containing only specified CpGs
#'
#' @details
#' CpGs not present in the matrix are silently dropped.
#' Empty lines in the file are ignored.
#'
#' @export
subset_methylation <- function(mval_matrix, cpg_ids_file) {
  
  cpg_ids <- trimws(readLines(cpg_ids_file))
  cpg_ids <- cpg_ids[cpg_ids != ""]
  valid_cpgs <- cpg_ids[cpg_ids %in% colnames(mval_matrix)]
  
  cat(sprintf("Subsetted to %d CpGs (from %d in file)\n", 
              length(valid_cpgs), length(cpg_ids)))
  
  return(mval_matrix[, valid_cpgs, drop = FALSE])
}


#' One-Hot Encode Categorical Clinical Variables
#'
#' Create k-1 dummy variables for each categorical variable with k levels.
#' The reference level is omitted to avoid multicollinearity.
#'
#' @param clin Data frame with clinical variables
#' @param clin_categorical Vector of categorical variable names
#'
#' @return List containing:
#'   \item{encoded_df}{Data frame with continuous + one-hot encoded variables}
#'   \item{encoded_cols}{Vector of one-hot encoded column names}
#'
#' @details
#' Special handling for specific variables:
#' - LN: "N0" as reference
#' - NHG: Natural ordering (1, 2, 3, ...)
#' - ER/PR/HER2: "negative" or "0" as reference
#'
#' Example: NHG with levels 1,2,3 creates NHG2, NHG3 (NHG1 is reference)
#'
#' @export
onehot_encode_clinical <- function(clin, clin_categorical) {
  
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
      
    } else if (var %in% c("ER", "PR", "HER2")) {
      # For receptor status: negative as reference
      all_levels <- unique(as.character(clin[[var]]))
      if ("negative" %in% all_levels) {
        ref_first <- c("negative", setdiff(all_levels, "negative"))
      } else if ("0" %in% all_levels) {
        ref_first <- c("0", setdiff(all_levels, "0"))
      } else {
        # Alphabetical if negative/0 not present
        ref_first <- sort(all_levels)
      }
      clin[[var]] <- factor(clin[[var]], levels = ref_first)
      
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
# 2. FEATURE SELECTION AND PREPROCESSING
################################################################################

#' Filter Features by Variance
#'
#' Reduce dimensionality by removing low-variance features that are unlikely
#' to be informative. Must be done on training data only to prevent data leakage.
#'
#' @param X_train Training data matrix (samples x features)
#' @param variance_quantile Quantile threshold (0.75 = keep top 25%, default)
#'
#' @return Filtered matrix with high-variance features only
#'
#' @details
#' Features with NA variance (all NA or constant) are automatically removed.
#' The variance threshold is determined from the training data quantile.
#'
#' @export
filter_by_variance <- function(X_train, variance_quantile = 0.75) {
  
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


#' Filter Features by Univariate Cox Regression
#'
#' Select features most strongly associated with survival outcome through
#' univariate Cox regression. This captures features that discriminate between
#' events and non-events, even if they have low variance.
#'
#' @param X_train Training data matrix (samples x features)
#' @param y_train Survival object (from Surv())
#' @param selection_method Method for feature selection:
#'   \itemize{
#'     \item "top_n": Keep top N features by p-value (default)
#'     \item "p_threshold": Keep all features with p < threshold
#'     \item "top_proportion": Keep top X% by p-value
#'   }
#' @param top_n Number of features to keep (if method = "top_n", default: 5000)
#' @param p_threshold P-value threshold (if method = "p_threshold", default: 0.05)
#'
#' @return Filtered matrix with features associated with outcome
#'
#' @details
#' For each feature, fits: coxph(y_train ~ feature)
#' Extracts p-value from Wald test and selects based on chosen method.
#' Failed models are assigned p = 1.0 and filtered out.
#'
#' @export
filter_by_univariate_cox <- function(X_train, y_train, 
                                     selection_method = "top_n",
                                     top_n = 5000,
                                     p_threshold = 0.05) {
  
  # Input validation
  if (ncol(X_train) == 0) {
    stop("X_train has no columns to filter")
  }
  if (!selection_method %in% c("top_n", "p_threshold", "top_proportion")) {
    stop("selection_method must be 'top_n', 'p_threshold', or 'top_proportion'")
  }
  
  n_features <- ncol(X_train)
  cat(sprintf("Running univariate Cox regression on %d features...\n", n_features))
  
  # Run univariate Cox regression for each feature
  p_values <- numeric(n_features)
  names(p_values) <- colnames(X_train)
  
  for (i in seq_len(n_features)) {
    p_values[i] <- tryCatch({
      # Fit univariate Cox model
      fit <- coxph(y_train ~ X_train[, i], ties = "efron")
      # Extract p-value from Wald test
      summary(fit)$coefficients[1, "Pr(>|z|)"]
    }, error = function(e) {
      # If model fails, set p-value to 1 (will be filtered out)
      1.0
    })
    
    # Progress indicator every 10,000 features
    if (i %% 10000 == 0) {
      cat(sprintf("  Processed %d / %d features...\n", i, n_features))
    }
  }
  
  # Select features based on method
  if (selection_method == "top_n") {
    # Keep top N features by p-value
    n_keep <- min(top_n, n_features)
    top_indices <- order(p_values)[1:n_keep]
    keep_features <- names(p_values)[top_indices]
    
    cat(sprintf("Keeping top %d features by p-value\n", n_keep))
    cat(sprintf("  P-value range: %.2e to %.2e\n", 
                min(p_values[top_indices]), max(p_values[top_indices])))
    
  } else if (selection_method == "p_threshold") {
    # Keep all features with p < threshold
    keep_features <- names(p_values[p_values < p_threshold])
    
    cat(sprintf("Keeping features with p < %.3f: %d features\n", 
                p_threshold, length(keep_features)))
    
  } else if (selection_method == "top_proportion") {
    # Keep top X% by p-value
    proportion <- top_n  # Reuse parameter (expects value like 0.1 for top 10%)
    if (proportion <= 0 || proportion > 1) {
      stop("For top_proportion method, top_n should be between 0 and 1")
    }
    
    n_keep <- ceiling(n_features * proportion)
    top_indices <- order(p_values)[1:n_keep]
    keep_features <- names(p_values)[top_indices]
    
    cat(sprintf("Keeping top %.1f%% features by p-value: %d features\n", 
                proportion * 100, n_keep))
  }
  
  # Handle case where no features pass threshold
  if (length(keep_features) == 0) {
    warning("No features passed selection criteria. Keeping top 100 by p-value.")
    top_indices <- order(p_values)[1:min(100, n_features)]
    keep_features <- names(p_values)[top_indices]
  }
  
  # Filter matrix
  X_filtered <- X_train[, keep_features, drop = FALSE]
  
  cat(sprintf("After univariate Cox filter: %d features\n", ncol(X_filtered)))
  
  return(X_filtered)
}


#' Apply Multi-Stage Feature Filtering
#'
#' Two-stage filtering approach:
#' 1. Variance filtering: Removes low-variance, uninformative features
#' 2. Univariate Cox (optional): Keeps features associated with outcome
#'
#' Clinical variables are preserved at all stages to ensure they remain in the
#' model regardless of variance or univariate association.
#'
#' @param X Feature matrix (samples x features)
#' @param vars_preserve Vector of variable names to preserve (e.g., clinical variables)
#' @param variance_quantile Variance filter threshold (default: 0.75 = top 25%)
#' @param apply_cox_filter Whether to apply univariate Cox filtering (default: FALSE)
#' @param y_train Survival object (required if apply_cox_filter = TRUE)
#' @param cox_selection_method How to select features ("top_n", "p_threshold", "top_proportion")
#' @param cox_top_n Number of features to keep (for "top_n" method, default: 5000)
#' @param cox_p_threshold P-value threshold (for "p_threshold" method, default: 0.05)
#'
#' @return Filtered feature matrix with preserved variables included
#'
#' @export
prepare_filtered_features <- function(X, 
                                      vars_preserve = NULL, 
                                      variance_quantile = 0.75,
                                      apply_cox_filter = FALSE,
                                      y_train = NULL,
                                      cox_selection_method = "top_n",
                                      cox_top_n = 5000,
                                      cox_p_threshold = 0.05) {
  
  # Input validation
  if (apply_cox_filter && is.null(y_train)) {
    stop("y_train must be provided when apply_cox_filter = TRUE")
  }
  
  # Separate preserved vs filterable features
  if (!is.null(vars_preserve)) {
    is_preserved <- colnames(X) %in% vars_preserve
    X_preserved <- X[, is_preserved, drop = FALSE]
    X_to_filter <- X[, !is_preserved, drop = FALSE]
    n_preserved <- sum(is_preserved)
  } else {
    is_preserved <- rep(FALSE, ncol(X))
    X_preserved <- NULL
    X_to_filter <- X
    n_preserved <- 0
  }
  
  cat(sprintf("\n========== FEATURE FILTERING ==========\n"))
  cat(sprintf("Starting features: %d (preserved: %d, to filter: %d)\n", 
              ncol(X), n_preserved, ncol(X_to_filter)))
  
  # Step 1: Variance Filtering
  cat(sprintf("\n--- Step 1: Variance Filtering ---\n"))
  
  if (ncol(X_to_filter) > 0) {
    X_var_filtered <- filter_by_variance(
      X_to_filter, 
      variance_quantile = variance_quantile
    )
  } else {
    cat("No features to filter by variance (all preserved)\n")
    X_var_filtered <- X_to_filter
  }
  
  # Step 2: Univariate Cox Filtering (Optional)
  if (apply_cox_filter && ncol(X_var_filtered) > 0) {
    cat(sprintf("\n--- Step 2: Univariate Cox Filtering ---\n"))
    
    X_cox_filtered <- filter_by_univariate_cox(
      X_train = X_var_filtered,
      y_train = y_train,
      selection_method = cox_selection_method,
      top_n = cox_top_n,
      p_threshold = cox_p_threshold
    )
  } else {
    if (apply_cox_filter) {
      cat("\n--- Step 2: Univariate Cox Filtering ---\n")
      cat("Skipped (no features after variance filtering)\n")
    }
    X_cox_filtered <- X_var_filtered
  }
  
  # Combine Filtered and Preserved Features
  if (!is.null(X_preserved) && ncol(X_preserved) > 0) {
    X_final <- cbind(X_cox_filtered, X_preserved)
  } else {
    X_final <- X_cox_filtered
  }
  
  cat(sprintf("\n========== FILTERING SUMMARY ==========\n"))
  cat(sprintf("Initial features:          %d\n", ncol(X)))
  cat(sprintf("After variance filter:     %d\n", ncol(X_var_filtered)))
  if (apply_cox_filter) {
    cat(sprintf("After Cox filter:          %d\n", ncol(X_cox_filtered)))
  }
  cat(sprintf("Preserved variables:       %d\n", n_preserved))
  cat(sprintf("TOTAL for modeling:        %d\n", ncol(X_final)))
  cat(sprintf("=======================================\n\n"))
  
  return(X_final)
}


#' Prepare Cross-Validation Data Splits
#'
#' Subset features and clinical data for train/test splits in cross-validation.
#'
#' @param X Feature matrix (samples x features)
#' @param clinical Clinical data frame (samples x variables)
#' @param train_idx Vector of training sample indices
#' @param test_idx Vector of test sample indices
#'
#' @return List containing:
#'   \item{X_train}{Training feature matrix}
#'   \item{X_test}{Test feature matrix}
#'   \item{clinical_train}{Training clinical data}
#'   \item{clinical_test}{Test clinical data}
#'
#' @export
cv_data_prep <- function(X, clinical, train_idx, test_idx) {
  
  # Subset features
  X_train <- X[train_idx, , drop = FALSE]
  X_test  <- X[test_idx, , drop = FALSE]
  
  # Subset clinical data
  clinical_train <- clinical[train_idx, , drop = FALSE]
  clinical_test  <- clinical[test_idx, , drop = FALSE]
  
  return(list(
    X_train = X_train,
    X_test = X_test,
    clinical_train = clinical_train,
    clinical_test = clinical_test
  ))
}


#' Standardize Continuous Features
#'
#' Standardize continuous features using training set parameters.
#' One-hot encoded categorical variables are left unscaled.
#' Test set is scaled using TRAINING set mean/SD to prevent data leakage.
#'
#' @param X_train Training feature matrix
#' @param X_test Test feature matrix (optional, can be NULL)
#' @param dont_scale Vector of column names to not scale (e.g., one-hot encoded)
#' @param verbose Print scaling summary (default: FALSE)
#'
#' @return List containing:
#'   \item{X_train_scaled}{Scaled training data}
#'   \item{X_test_scaled}{Scaled test data (NULL if X_test not provided)}
#'   \item{centers}{Mean values from training data}
#'   \item{scales}{SD values from training data}
#'   \item{cont_cols}{Names of scaled columns}
#'
#' @export
scale_continuous_features <- function(X_train, X_test = NULL, dont_scale = NULL, 
                                      verbose = FALSE) {
  
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
      scale = scales
    )
  } else {
    X_test_scaled <- NULL
  }
  
  if (verbose) {
    cat(sprintf("Scaled %d continuous features\n", length(cont_cols)))
  }
  
  return(list(
    X_train_scaled = X_train_scaled,
    X_test_scaled = X_test_scaled,
    centers = centers,
    scales = scales,
    cont_cols = cont_cols
  ))
}


################################################################################
# 3. ELASTIC NET COX REGRESSION
################################################################################

#' Cross-Validate Elastic Net Across Alpha Grid
#'
#' Perform cross-validation for glmnet across multiple alpha values.
#' Alpha controls elastic net mixing:
#'   - alpha = 0: Ridge regression (L2 penalty)
#'   - alpha = 1: LASSO (L1 penalty)
#'   - 0 < alpha < 1: Elastic net (combined L1/L2)
#'
#' @param X_train Training feature matrix
#' @param y_train Survival object (from Surv())
#' @param alpha_grid Vector of alpha values to test
#' @param penalty_factor Feature-specific penalties (0 = no penalty, default: NULL)
#' @param foldid CV fold assignments (default: NULL)
#' @param family Model family (default: "cox" for survival analysis)
#' @param seed Random seed for reproducibility (default: 123)
#'
#' @return List of CV results for each alpha value, each containing:
#'   \item{alpha}{Alpha value tested}
#'   \item{lambda}{Optimal lambda value}
#'   \item{cvm_min}{CV deviance at optimal lambda}
#'   \item{cvm_se}{Standard error of CV deviance}
#'   \item{n_features}{Number of selected features}
#'   \item{features}{Names of selected features}
#'   \item{fit}{Full cv.glmnet object}
#'
#' @export
cv_glmnet_alpha_grid <- function(X_train, y_train, alpha_grid, 
                                 penalty_factor = NULL, foldid = NULL, 
                                 family = "cox", seed = 123) {
  
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


#' Tune and Fit Cox Elastic Net Model
#'
#' Perform stratified cross-validation to find optimal alpha and lambda values,
#' then fit final model on full training set.
#'
#' @param X_train Training feature matrix
#' @param y_train Survival object
#' @param clinical_train Clinical data for stratification
#' @param event_col Column name for event indicator
#' @param alpha_grid Vector of alpha values to test
#' @param penalty_factor Feature-specific penalties (default: NULL)
#' @param n_inner_folds Number of CV folds (default: 5)
#' @param seed Random seed (default: 123)
#' @param outcome_name Name for logging (default: "outcome")
#'
#' @return List containing:
#'   \item{cv_results}{Full CV results for all alpha values}
#'   \item{best_result}{Best result object}
#'   \item{best_alpha}{Optimal alpha value}
#'   \item{best_lambda}{Optimal lambda value}
#'   \item{final_fit}{Fitted glmnet model on full training set}
#'
#' @export
tune_and_fit_coxnet <- function(X_train, y_train, clinical_train, event_col,
                                alpha_grid, penalty_factor = NULL,
                                n_inner_folds = 5, seed = 123,
                                outcome_name = "outcome") {
  
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


#' Extract Non-Zero Coefficients from glmnet Model
#'
#' Extract and optionally sort coefficients from fitted glmnet model.
#'
#' @param model_fit Fitted glmnet model object
#' @param sort_by_abs Sort by absolute coefficient value (default: TRUE)
#'
#' @return List containing:
#'   \item{coef_df}{Data frame with feature names and coefficients}
#'   \item{features}{Vector of selected feature names}
#'
#' @export
extract_nonzero_coefs <- function(model_fit, sort_by_abs = TRUE) {
  
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
# 4. PENALIZED FINE-GRAY MODELING
################################################################################

#' Evaluate Alpha and Lambda Through Inner Cross-Validation
#'
#' For a given alpha value, perform inner CV to find optimal lambda for
#' penalized Fine-Gray model using fastcmprsk.
#'
#' @param X_input Input feature matrix
#' @param clinical_input Clinical data
#' @param alpha Alpha value to test
#' @param lambda_seq Sequence of lambda values (default: NULL, auto-generated)
#' @param cr_time Column name for time to competing risk event (default: "time_to_CompRisk_event")
#' @param cr_event Column name for competing risk event indicator (default: "CompRisk_event_coded")
#' @param penalty Penalty type (default: "ENET" for elastic net)
#' @param n_inner_folds Number of inner CV folds (default: 3)
#'
#' @return List containing:
#'   \item{alpha}{Alpha value tested}
#'   \item{best_lambda}{Optimal lambda value}
#'   \item{best_auc}{Mean AUC at optimal lambda}
#'
#' @details
#' This function is called internally by fit_penalized_finegray_cv() for each
#' alpha in the grid search.
#'
#' @export
evaluate_alpha_lambda_cv <- function(X_input, clinical_input,
                                     alpha, lambda_seq,
                                     cr_time = "time_to_CompRisk_event",
                                     cr_event = "CompRisk_event_coded",
                                     penalty = "ENET",
                                     n_inner_folds = 3) {
  
  # Generate lambda sequence if not provided
  if (is.null(lambda_seq)) {
    lambda_seq <- 10^seq(log10(0.2), log10(0.001), length.out = 25)
  }
  
  # Create inner folds
  set.seed(123)
  inner_folds <- caret::createFolds(
    y = clinical_input[[cr_event]],
    k = n_inner_folds,
    list = TRUE,
    returnTrain = FALSE
  )
  
  # Matrix to store performance
  perf_mat <- matrix(NA_real_, nrow = n_inner_folds, ncol = length(lambda_seq))
  
  # Inner CV loop
  for (inner_k in seq_len(n_inner_folds)) {
    
    inner_val_idx <- inner_folds[[inner_k]]
    inner_train_idx <- setdiff(seq_len(nrow(X_input)), inner_val_idx)
    
    inner_data <- cv_data_prep(
      X = X_input,
      clinical = clinical_input,
      train_idx = inner_train_idx,
      test_idx = inner_val_idx
    )
    
    scale_res <- scale_continuous_features(
      X_train = inner_data$X_train,
      X_test = inner_data$X_test,
      dont_scale = NULL
    )
    
    fgr_inner_train <- cbind(
      inner_data$clinical_train[c(cr_time, cr_event)],
      scale_res$X_train_scaled
    )
    fgr_inner_val <- cbind(
      inner_data$clinical_test[c(cr_time, cr_event)],
      scale_res$X_test_scaled
    )
    
    feature_cols <- setdiff(colnames(fgr_inner_train), c(cr_time, cr_event))
    feature_matrix_train <- as.matrix(fgr_inner_train[, feature_cols])
    feature_matrix_val <- as.matrix(fgr_inner_val[, feature_cols])
    
    # Fit for this alpha across all lambdas
    fit_inner <- fastcmprsk::fastCrrp(
      Crisk(fgr_inner_train[[cr_time]], 
            fgr_inner_train[[cr_event]]) ~ feature_matrix_train,
      lambda = lambda_seq,
      penalty = penalty,
      alpha = alpha,
      standardize = FALSE
    )
    
    # Evaluate each lambda
    for (lam_idx in seq_along(lambda_seq)) {
      
      if (is.nan(fit_inner$coef[1, lam_idx]) || !fit_inner$converged[lam_idx]) {
        next
      }
      
      coefs_lambda <- fit_inner$coef[, lam_idx]
      eta_val <- as.vector(feature_matrix_val %*% coefs_lambda)
      
      score_data <- fgr_inner_val[, c(cr_time, cr_event)]
      score_data$risk_score <- eta_val
      
      score_res <- Score(
        list("risk_score" = score_data$risk_score),
        formula = Hist(time_to_CompRisk_event, CompRisk_event_coded) ~ 1,
        data = score_data,
        cause = 1,
        times = EVAL_TIMES,
        metrics = "auc",
        cens.model = "cox"
      )
      
      perf_mat[inner_k, lam_idx] <- mean(score_res$AUC$score$AUC, na.rm = TRUE)
    }
  }
  
  # Find best lambda for this alpha
  mean_perf <- colMeans(perf_mat, na.rm = TRUE)
  best_lambda_idx <- which.max(mean_perf)
  best_lambda <- lambda_seq[best_lambda_idx]
  best_auc <- mean_perf[best_lambda_idx]
  
  # Count number of selected (non-zero) features at best lambda
  coefs_best <- fit_inner$coef[, best_lambda_idx]
  n_selected <- sum(abs(coefs_best) > 1e-8)
  
  # Print summary for this alpha
  cat(sprintf(
    "Alpha = %.2f | Best lambda = %.5g | Mean AUC = %.3f | N Selected = %d\n",
    alpha, best_lambda, best_auc, n_selected
  ))
  
  return(list(
    alpha = alpha,
    best_lambda = best_lambda,
    best_auc = best_auc
  ))
}


#' Fit Penalized Fine-Gray Model with Cross-Validation
#'
#' Search over alpha and lambda parameters, then fit final penalized Fine-Gray
#' model on full training set.
#'
#' @param X_input Input feature matrix
#' @param clinical_input Clinical data
#' @param cr_time Column name for time to competing risk event (default: "time_to_CompRisk_event")
#' @param cr_event Column name for competing risk event indicator (default: "CompRisk_event_coded")
#' @param penalty Penalty type (default: "ENET" for elastic net)
#' @param alpha_seq Single alpha value or vector to search (default: 0.5)
#' @param lambda_seq Sequence of lambda values (default: NULL, auto-generated)
#' @param n_inner_folds Number of inner CV folds (default: 3)
#'
#' @return List containing:
#'   \item{model}{Fitted fastCrrp model object}
#'   \item{results_table}{Data frame with feature coefficients and scaling parameters}
#'   \item{best_alpha}{Optimal alpha value}
#'   \item{best_lambda}{Optimal lambda value}
#'
#' @export
fit_penalized_finegray_cv <- function(X_input, clinical_input,
                                      cr_time = "time_to_CompRisk_event",
                                      cr_event = "CompRisk_event_coded",
                                      penalty = "ENET",
                                      alpha_seq = 0.5,
                                      lambda_seq = NULL,
                                      n_inner_folds = 3) {
  
  cat(sprintf("\n=== Penalized Fine-Gray with CV (%s) ===\n", penalty))
  
  # If only one alpha, just use it
  if (length(alpha_seq) == 1) {
    cat(sprintf("Using alpha = %.2f\n", alpha_seq))
    best_alpha <- alpha_seq
    cv_result <- evaluate_alpha_lambda_cv(
      X_input, clinical_input, 
      alpha = best_alpha,
      lambda_seq = lambda_seq,
      cr_time = cr_time,
      cr_event = cr_event,
      penalty = penalty,
      n_inner_folds = n_inner_folds
    )
    best_lambda <- cv_result$best_lambda
    
  } else {
    # If multiple alphas, search over them
    cat(sprintf("Searching over %d alpha values...\n", length(alpha_seq)))
    
    # Test each alpha
    results <- lapply(alpha_seq, function(a) {
      cat(sprintf("  Testing alpha = %.2f... ", a))
      res <- evaluate_alpha_lambda_cv(
        X_input, clinical_input,
        alpha = a,
        lambda_seq = lambda_seq,
        cr_time = cr_time,
        cr_event = cr_event,
        penalty = penalty,
        n_inner_folds = n_inner_folds
      )
      cat(sprintf("best lambda = %.4f, AUC = %.3f\n", res$best_lambda, res$best_auc))
      return(res)
    })
    
    # Pick best alpha
    aucs <- sapply(results, `[[`, "best_auc")
    best_idx <- which.max(aucs)
    best_alpha <- results[[best_idx]]$alpha
    best_lambda <- results[[best_idx]]$best_lambda
    
    cat(sprintf("\nBest: alpha = %.2f, lambda = %.4f, AUC = %.3f\n",
                best_alpha, best_lambda, aucs[best_idx]))
  }
  
  # Refit on full data
  cat("\nRefitting on full training set...\n")
  
  scale_res_full <- scale_continuous_features(
    X_train = X_input, 
    X_test = NULL, 
    dont_scale = NULL
  )
  
  fgr_train_final <- cbind(
    clinical_input[c(cr_time, cr_event)],
    scale_res_full$X_train_scaled
  )
  
  feature_cols <- setdiff(colnames(fgr_train_final), c(cr_time, cr_event))
  feature_matrix_final <- as.matrix(fgr_train_final[, feature_cols])
  
  final_model <- fastcmprsk::fastCrrp(
    Crisk(fgr_train_final[[cr_time]], 
          fgr_train_final[[cr_event]]) ~ feature_matrix_final,
    lambda = best_lambda,
    penalty = penalty,
    alpha = best_alpha,
    standardize = FALSE
  )
  
  final_coefs <- final_model$coef[, 1]
  all_features <- colnames(feature_matrix_final)
  selected_idx <- abs(final_coefs) > 1e-8
  
  cat(sprintf("Selected %d/%d features\n", sum(selected_idx), length(all_features)))
  
  results_df <- data.frame(
    feature = all_features,
    pFG_coefficient = final_coefs,
    selected = selected_idx,
    scale_center = scale_res_full$centers[all_features],
    scale_scale = scale_res_full$scales[all_features],
    stringsAsFactors = FALSE
  )
  
  return(list(
    model = final_model,
    results_table = results_df,
    best_alpha = best_alpha,
    best_lambda = best_lambda
  ))
}


#' Calculate Methylation Risk Score (MRS)
#'
#' Calculate methylation risk score from CpG data and coefficients.
#' This is the linear predictor (weighted sum) from the penalized Fine-Gray model.
#'
#' @param X_data Data frame or matrix with CpG columns (samples x CpGs)
#' @param cpg_coefficients Named numeric vector of coefficients 
#'                         (names = CpG IDs, values = coefficients)
#' @param scaling_params List with 'centers' and 'scales' for CpG scaling
#' @param verbose Print diagnostic information (default: TRUE)
#'
#' @return List containing:
#'   \item{mrs}{Numeric vector of methylation risk scores}
#'   \item{input_scaling_params}{Scaling parameters used}
#'   \item{input_cpg_coefs}{CpG coefficients used}
#'
#' @details
#' The function:
#' 1. Validates that all required CpGs are present in X_data
#' 2. Extracts and orders CpGs to match coefficients
#' 3. Applies provided scaling parameters
#' 4. Calculates weighted sum: MRS = X_scaled %*% coefficients
#'
#' @export
calculate_methylation_risk_score <- function(X_data,
                                             cpg_coefficients,
                                             scaling_params,
                                             verbose = TRUE) {
  
  # Input validation
  if (!is.vector(cpg_coefficients) || is.null(names(cpg_coefficients))) {
    stop("cpg_coefficients must be a named numeric vector")
  }
  
  cpg_names <- names(cpg_coefficients)
  n_cpgs <- length(cpg_names)
  
  # Check that all required CpGs are in the data
  missing_cpgs <- setdiff(cpg_names, colnames(X_data))
  if (length(missing_cpgs) > 0) {
    stop(sprintf("Missing CpGs in X_data: %s", 
                 paste(missing_cpgs, collapse = ", ")))
  }
  
  # Extract relevant CpGs (in correct order matching coefficients)
  X_cpgs <- X_data[, cpg_names, drop = FALSE]
  
  if (verbose) {
    cat(sprintf("\n--- Calculating Methylation Risk Score ---\n"))
    cat(sprintf("  Number of CpGs: %d\n", n_cpgs))
    cat(sprintf("  Number of samples: %d\n", nrow(X_cpgs)))
  }
  
  # Validate scaling parameters
  if (is.null(scaling_params$center) || is.null(scaling_params$scale)) {
    stop("scaling_params must contain 'center' and 'scale' vectors")
  }
  
  # Verify CpG names match
  if (!all(cpg_names %in% names(scaling_params$center))) {
    stop("CpG names in coefficients don't match scaling parameters")
  }
  
  # Apply scaling with provided parameters
  X_cpgs_scaled <- scale(
    X_cpgs, 
    center = scaling_params$center[cpg_names],
    scale = scaling_params$scale[cpg_names]
  )
  
  if (verbose) {
    cat("  Scaling: Using provided parameters\n")
  }
  
  # Calculate MRS (weighted sum of scaled CpGs)
  # Ensure coefficients are in same order as columns
  coefs_ordered <- cpg_coefficients[colnames(X_cpgs_scaled)]
  
  # Calculate weighted sum (linear predictor)
  mrs <- as.vector(as.matrix(X_cpgs_scaled) %*% coefs_ordered)
  
  if (verbose) {
    cat(sprintf("  MRS range: [%.3f, %.3f]\n", 
                min(mrs, na.rm = TRUE), 
                max(mrs, na.rm = TRUE)))
  }
  
  return(list(
    mrs = mrs,
    input_scaling_params = scaling_params,
    input_cpg_coefs = cpg_coefficients
  ))
}


################################################################################
# 5. FINE-GRAY MODEL FITTING AND EVALUATION
################################################################################

#' Fit Fine-Gray Competing Risks Model
#'
#' Fit a Fine-Gray subdistribution hazard model for competing risks data.
#'
#' @param fgr_data Data frame with features and outcome columns
#' @param cr_time Column name for time to competing risk event
#' @param cr_event Column name for competing risk event indicator
#' @param cause Cause of interest (default: 1)
#'
#' @return List containing:
#'   \item{model}{Fitted FGR model object}
#'   \item{formula}{Formula used for modeling}
#'
#' @details
#' All columns except cr_time and cr_event are used as features in the model.
#' Uses the FGR() function from the riskRegression package.
#'
#' @export
fit_fine_gray_model <- function(fgr_data, cr_time, cr_event, cause = 1) {
  
  cat(sprintf("\n--- Fitting Fine-Gray Model ---\n"))
  
  # Identify feature columns by excluding outcome columns
  feature_cols <- setdiff(colnames(fgr_data), c(cr_time, cr_event))
  
  # Build model formula
  formula_str <- sprintf(
    "Hist(%s, %s) ~ %s",
    cr_time,
    cr_event,
    paste(feature_cols, collapse = " + ")
  )
  
  # Convert string to formula object
  formula_fg <- as.formula(formula_str)
  
  # Fit the Fine-Gray model
  fgr_model <- FGR(
    formula = formula_fg,
    data = fgr_data,
    cause = cause
  )
  
  cat(sprintf("Fine-Gray model fitted with %d features\n", length(feature_cols)))
  
  return(list(
    model = fgr_model,
    formula = formula_fg
  ))
}


#' Calculate Variable Importance for Fine-Gray Model
#'
#' Compute hazard ratios, Wald statistics, and p-values from Fine-Gray model
#' coefficients and variance-covariance matrix.
#'
#' @param fgr_model Fitted Fine-Gray model (from FGR())
#' @param encoded_cols Names of one-hot encoded columns for type annotation (default: NULL)
#' @param top_n Limit output to top N features (default: NULL, shows all)
#' @param verbose Print detailed summary (default: TRUE)
#'
#' @return Data frame with importance metrics sorted by statistical significance:
#'   \item{feature}{Feature name}
#'   \item{coefficient}{Raw coefficient from model}
#'   \item{se}{Standard error}
#'   \item{HR}{Hazard ratio (exp(coefficient))}
#'   \item{HR_magnitude}{Effect magnitude (always >= 1)}
#'   \item{wald_z}{Wald z-score}
#'   \item{abs_wald_z}{Absolute Wald z-score}
#'   \item{p_value}{Two-tailed p-value}
#'   \item{type}{Variable type (categorical or continuous), if encoded_cols provided}
#'
#' @details
#' HR_magnitude is calculated as: ifelse(HR >= 1, HR, 1/HR)
#' This makes effect sizes symmetric around the null (HR=1).
#'
#' @export
calculate_fgr_importance <- function(fgr_model, encoded_cols = NULL, 
                                     top_n = NULL, verbose = TRUE) {
  
  # Extract coefficients and variance-covariance matrix
  fg_coef <- fgr_model$crrFit$coef
  fg_vcov <- fgr_model$crrFit$var
  fg_se <- sqrt(diag(fg_vcov))
  
  # Calculate hazard ratios
  HR <- exp(fg_coef)
  
  # Calculate effect magnitude (HR scale, always >= 1)
  # For HR < 1, use reciprocal so magnitude is symmetric around null
  HR_magnitude <- ifelse(HR >= 1, HR, 1/HR)
  
  # Create comprehensive importance table
  vimp <- data.frame(
    feature = names(fg_coef),
    coefficient = as.vector(fg_coef),
    se = fg_se,
    HR = HR,
    HR_magnitude = HR_magnitude,
    wald_z = fg_coef / fg_se,
    abs_wald_z = abs(fg_coef / fg_se),
    p_value = 2 * pnorm(abs(fg_coef / fg_se), lower.tail = FALSE),
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
  vimp$HR_magnitude <- round(vimp$HR_magnitude, 2)
  vimp$wald_z <- round(vimp$wald_z, 2)
  vimp$abs_wald_z <- round(vimp$abs_wald_z, 2)
  vimp$p_value <- ifelse(vimp$p_value < 0.001, "<0.001", 
                         sprintf("%.3f", vimp$p_value))
  
  # Print detailed summaries
  if (verbose) {
    cat("\n========================================\n")
    cat("FINE-GRAY VARIABLE IMPORTANCE\n")
    cat("========================================\n\n")
    
    cat("Top features by statistical significance:\n")
    print(head(vimp[, c("feature", "HR", "wald_z", "p_value")], 20))
    
    cat("\n--- Summary Statistics ---\n")
    cat(sprintf("Total features: %d\n", nrow(vimp)))
    cat(sprintf("Significant (p < 0.05): %d\n", 
                sum(as.numeric(ifelse(vimp$p_value == "<0.001", 0, vimp$p_value)) < 0.05)))
    
    # Separate by variable type
    if (!is.null(encoded_cols) && "type" %in% colnames(vimp)) {
      n_cat <- sum(vimp$type == "categorical")
      n_cont <- sum(vimp$type == "continuous")
      cat(sprintf("Categorical variables: %d\n", n_cat))
      cat(sprintf("Continuous variables: %d\n", n_cont))
      
      if (n_cat > 0) {
        cat("\nTop categorical variables:\n")
        vimp_cat <- vimp[vimp$type == "categorical", ]
        print(head(vimp_cat[, c("feature", "HR", "wald_z", "p_value")], 5))
      }
      
      if (n_cont > 0) {
        cat("\nTop continuous variables:\n")
        vimp_cont <- vimp[vimp$type == "continuous", ]
        print(head(vimp_cont[, c("feature", "HR", "wald_z", "p_value")], 10))
      }
    }
    
    cat("\n--- Effect Size Distribution ---\n")
    cat(sprintf("Protective effects (HR < 1): %d\n", sum(vimp$HR < 1)))
    cat(sprintf("Risk-increasing effects (HR > 1): %d\n", sum(vimp$HR > 1)))
    cat(sprintf("Median HR magnitude: %.2f\n", median(vimp$HR_magnitude)))
    cat(sprintf("Max HR magnitude: %.2f\n", max(vimp$HR_magnitude)))
    
    cat("\n========================================\n\n")
  }
  
  return(vimp)
}


################################################################################
# 6. PERFORMANCE AGGREGATION AND STABILITY
################################################################################

#' Aggregate Performance Across Cross-Validation Folds
#'
#' Calculate mean, SE, SD, and confidence intervals for all performance metrics
#' across CV folds. Also tracks the number of features in each fold's model.
#'
#' @param outer_fold_results List of fold results from outer CV loop
#' @param round_digits Number of decimal places for rounding (default: 3)
#' @param conf_level Confidence level for intervals (default: 0.95)
#' @param verbose Print summary to console (default: TRUE)
#'
#' @return List containing:
#'   \item{summary}{Data frame with aggregated statistics}
#'   \item{all_folds}{Data frame with individual fold results}
#'
#' @export
aggregate_cv_performance <- function(outer_fold_results, 
                                     round_digits = 3, 
                                     conf_level = 0.95,
                                     verbose = TRUE) {
  
  if (verbose) cat("\n========== AGGREGATED PERFORMANCE ==========\n")
  
  # Extract performance dataframes from all folds
  performance_list <- lapply(outer_fold_results, function(fold) fold$final_fg$performance)
  all_performance <- do.call(rbind, performance_list)
  
  # Extract number of features in each fold's pFG model (MRS score)
  n_features_per_fold <- sapply(outer_fold_results, function(fold) {
    coef_df <- fold$penalized_fg$coefficients_table
    sum(coef_df$pen_fg_coef != 0)
  })
  
  # Add feature count column
  all_performance$n_fg_features <- n_features_per_fold
  
  # Calculate feature count statistics
  feature_row <- data.frame(
    metric = "n_fg_features",
    mean = round(mean(n_features_per_fold), 1),
    se = round(sd(n_features_per_fold) / sqrt(length(n_features_per_fold)), 1),
    sd = round(sd(n_features_per_fold), 1),
    ci_lower = min(n_features_per_fold),
    ci_upper = max(n_features_per_fold)
  )
  
  # Get metric column names (exclude 'model' and 'n_fg_features')
  metric_cols <- setdiff(names(all_performance), c("model", "n_fg_features"))
  
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
  
  # Combine: feature row first, then performance metrics
  performance_summary <- rbind(feature_row, performance_summary)
  rownames(performance_summary) <- NULL
  
  # Print results
  if (verbose) {
    cat("\nIndividual Fold Performance:\n")
    print(all_performance)
    
    cat(sprintf("\nPerformance Summary (Mean ± SE, %d%% CI):\n", conf_level * 100))
    for (i in 1:nrow(performance_summary)) {
      metric_name <- performance_summary$metric[i]
      
      # Special formatting for feature count row (shows range instead of CI)
      if (metric_name == "n_fg_features") {
        cat(sprintf("%15s: %.1f ± %.1f (range: %d - %d)\n",
                    metric_name,
                    performance_summary$mean[i],
                    performance_summary$se[i],
                    performance_summary$ci_lower[i],
                    performance_summary$ci_upper[i]))
      } else {
        cat(sprintf("%15s: %.3f ± %.3f (95%% CI: %.3f - %.3f)\n",
                    metric_name,
                    performance_summary$mean[i],
                    performance_summary$se[i],
                    performance_summary$ci_lower[i],
                    performance_summary$ci_upper[i]))
      }
    }
  }
  
  return(list(
    summary = performance_summary,
    all_folds = all_performance
  ))
}


#' Assess Feature Selection Stability for Penalized Fine-Gray Models
#'
#' Evaluate which CpGs were consistently selected for methylation risk score (MRS)
#' construction across cross-validation folds.
#'
#' @param outer_fold_results List of fold results from nested CV
#' @param verbose Print summary statistics (default: TRUE)
#'
#' @return List containing:
#'   \item{stability_metrics}{Data frame with CpG statistics}
#'   \item{selection_matrix}{Data frame with CpG column + binary fold indicators}
#'   \item{coefficient_matrix}{Data frame with CpG column + actual coefficients per fold}
#'
#' @details
#' For each CpG, calculates:
#' - Selection frequency across folds
#' - Direction consistency (all positive or all negative)
#' - Mean, median, and SD of coefficients (when selected)
#' - Number of positive/negative coefficient occurrences
#'
#' @export
assess_finegray_stability <- function(outer_fold_results, verbose = TRUE) {
  
  # Setup
  n_folds <- length(outer_fold_results)
  
  # Extract penalized FG coefficient tables from all folds
  coef_list <- lapply(outer_fold_results, function(fold) {
    fold$penalized_fg$coefficients_table
  })
  
  # Get all unique CpGs selected across any fold
  all_cpgs <- unique(unlist(lapply(coef_list, function(df) df$feature)))
  
  # Build combined table (one row per CpG)
  combined_table <- do.call(rbind, lapply(all_cpgs, function(cpg) {
    
    # Extract penalized FG coefficients for this CpG from each fold
    pen_fg_coefs <- sapply(coef_list, function(df) {
      cpg_idx <- which(df$feature == cpg)
      if (length(cpg_idx) == 0) return(NA)  # NA if not selected
      df$pen_fg_coef[cpg_idx]
    })
    
    # Binary selection indicators (NA means not selected)
    is_selected <- as.numeric(!is.na(pen_fg_coefs))
    
    # Selection statistics
    n_selected <- sum(is_selected)
    selection_freq <- n_selected / n_folds
    
    # Count positive/negative coefficients (only for selected folds)
    selected_coefs <- pen_fg_coefs[!is.na(pen_fg_coefs)]
    n_positive <- sum(selected_coefs > 0)
    n_negative <- sum(selected_coefs < 0)
    
    # Coefficient statistics (only for folds where selected)
    if (length(selected_coefs) > 0) {
      direction_consistent <- all(selected_coefs > 0) | all(selected_coefs < 0)
      mean_coef <- mean(selected_coefs)
      sd_coef <- sd(selected_coefs)
      median_coef <- median(selected_coefs)
    } else {
      direction_consistent <- FALSE
      mean_coef <- NA
      sd_coef <- NA
      median_coef <- NA
    }
    
    # Build complete row
    row_df <- data.frame(
      feature = cpg,
      n_selected = n_selected,
      selection_freq = round(selection_freq, 3),
      direction_consistent = direction_consistent,
      n_positive = n_positive,
      n_negative = n_negative,
      mean_coef = round(mean_coef, 4),
      median_coef = round(median_coef, 4),
      sd_coef = round(sd_coef, 4),
      stringsAsFactors = FALSE
    )
    
    # Add fold selection indicators (0/1)
    fold_selection_df <- as.data.frame(t(is_selected))
    colnames(fold_selection_df) <- paste0("Fold", 1:n_folds, "_selected")
    
    # Add fold coefficients (actual values or NA)
    fold_coef_df <- as.data.frame(t(pen_fg_coefs))
    colnames(fold_coef_df) <- paste0("Fold", 1:n_folds, "_coef")
    
    cbind(row_df, fold_selection_df, fold_coef_df)
  }))
  
  # Sort by selection frequency, then by absolute mean coefficient
  combined_table <- combined_table[order(
    combined_table$selection_freq,
    abs(combined_table$mean_coef),
    decreasing = TRUE,
    na.last = TRUE
  ), ]
  rownames(combined_table) <- NULL
  
  # Split into three outputs
  # 1. Metrics table: CpG + statistics (no fold columns)
  metric_cols <- c("feature", "n_selected", "selection_freq",
                   "direction_consistent", "n_positive", "n_negative", 
                   "mean_coef", "median_coef", "sd_coef")
  stability_metrics <- combined_table[, metric_cols]
  
  # 2. Selection matrix: CpG + fold selections (0/1)
  fold_selection_cols <- grep("_selected$", names(combined_table), value = TRUE)
  selection_matrix <- combined_table[, c("feature", fold_selection_cols)]
  
  # 3. Coefficient matrix: CpG + fold coefficients (actual values)
  fold_coef_cols <- grep("_coef$", names(combined_table), value = TRUE)
  coefficient_matrix <- combined_table[, c("feature", fold_coef_cols)]
  
  # Print summary (if requested)
  if (verbose) {
    cat("\n========== PENALIZED FINE-GRAY CpG STABILITY ==========\n")
    cat(sprintf("Total folds: %d\n\n", n_folds))
    cat(sprintf("CpGs ever selected: %d\n", 
                sum(stability_metrics$n_selected > 0)))
    cat(sprintf("Direction-consistent CpGs: %d\n", 
                sum(stability_metrics$direction_consistent, na.rm = TRUE)))
    cat(sprintf("CpGs selected in all folds: %d\n", 
                sum(stability_metrics$selection_freq == 1)))
    cat(sprintf("CpGs selected in ≥80%% folds: %d\n", 
                sum(stability_metrics$selection_freq >= 0.8)))
    cat(sprintf("CpGs selected in ≥40%% folds: %d\n\n", 
                sum(stability_metrics$selection_freq >= 0.4)))
    
    # Show most stable CpGs (selected in at least 50% of folds)
    stable_cpgs <- stability_metrics[stability_metrics$selection_freq >= 0.4, ]
    
    if (nrow(stable_cpgs) > 0) {
      cat("Most stable CpGs (≥40% selection frequency):\n")
      print(stable_cpgs)
    } else {
      cat("No CpGs selected in ≥50% of folds\n")
    }
    
    cat("\n")
  }
  
  return(list(
    stability_metrics = stability_metrics,
    selection_matrix = selection_matrix,
    coefficient_matrix = coefficient_matrix
  ))
}


################################################################################
# END OF HELPER FUNCTIONS
################################################################################


################################################################################
# 6. PERFORMANCE AGGREGATION AND STABILITY (UPDATED FOR THREE-MODEL STRUCTURE)
################################################################################

#' Aggregate Performance Across Cross-Validation Folds (Three-Model Version)
#'
#' Calculate mean, SE, SD, and confidence intervals for all performance metrics
#' across CV folds for all three models (MRS-only, Clinical-only, Combined).
#'
#' @param outer_fold_results List of fold results from outer CV loop
#' @param model_name Name of model to aggregate: "FGR_MRS", "FGR_CLIN", or "FGR_COMBINED"
#' @param round_digits Number of decimal places for rounding (default: 3)
#' @param conf_level Confidence level for intervals (default: 0.95)
#' @param verbose Print summary to console (default: TRUE)
#'
#' @return List containing:
#'   \item{summary}{Data frame with aggregated statistics}
#'   \item{all_folds}{Data frame with individual fold results}
#'
#' @export
aggregate_cv_performance_threefg <- function(outer_fold_results, 
                                     model_name = "FGR_COMBINED",
                                     round_digits = 3, 
                                     conf_level = 0.95,
                                     verbose = TRUE) {
  
  if (verbose) {
    cat(sprintf("\n========== AGGREGATED PERFORMANCE: %s ==========\n", model_name))
  }
  
  # Extract performance dataframes from all folds for specified model
  performance_list <- lapply(outer_fold_results, function(fold) {
    perf_df <- fold$unpenalized_fgs$performances
    # Filter to just this model
    perf_df[perf_df$model == model_name, , drop = FALSE]
  })
  
  all_performance <- do.call(rbind, performance_list)
  
  # Check if we got any results
  if (nrow(all_performance) == 0) {
    stop(sprintf("No performance data found for model '%s'. Available models: %s", 
                 model_name, 
                 paste(unique(outer_fold_results[[1]]$unpenalized_fgs$performances$model), 
                       collapse = ", ")))
  }
  
  # Extract number of features in each fold's pFG model (MRS score)
  n_features_per_fold <- sapply(outer_fold_results, function(fold) {
    coef_df <- fold$penalized_fg$coefficients
    sum(coef_df$pen_fg_coef != 0)
  })
  
  # Add feature count column
  all_performance$n_fg_features <- n_features_per_fold
  
  # Calculate feature count statistics
  feature_row <- data.frame(
    metric = "n_fg_features",
    mean = round(mean(n_features_per_fold), 1),
    se = round(sd(n_features_per_fold) / sqrt(length(n_features_per_fold)), 1),
    sd = round(sd(n_features_per_fold), 1),
    ci_lower = min(n_features_per_fold),
    ci_upper = max(n_features_per_fold)
  )
  
  # Get metric column names (exclude 'model' and 'n_fg_features')
  metric_cols <- setdiff(names(all_performance), c("model", "n_fg_features"))
  
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
  
  # Combine: feature row first, then performance metrics
  performance_summary <- rbind(feature_row, performance_summary)
  rownames(performance_summary) <- NULL
  
  # Print results
  if (verbose) {
    cat("\nIndividual Fold Performance:\n")
    print(all_performance)
    
    cat(sprintf("\nPerformance Summary (Mean ± SE, %d%% CI):\n", conf_level * 100))
    for (i in 1:nrow(performance_summary)) {
      metric_name <- performance_summary$metric[i]
      
      # Special formatting for feature count row (shows range instead of CI)
      if (metric_name == "n_fg_features") {
        cat(sprintf("%15s: %.1f ± %.1f (range: %d - %d)\n",
                    metric_name,
                    performance_summary$mean[i],
                    performance_summary$se[i],
                    performance_summary$ci_lower[i],
                    performance_summary$ci_upper[i]))
      } else {
        cat(sprintf("%15s: %.3f ± %.3f (95%% CI: %.3f - %.3f)\n",
                    metric_name,
                    performance_summary$mean[i],
                    performance_summary$se[i],
                    performance_summary$ci_lower[i],
                    performance_summary$ci_upper[i]))
      }
    }
  }
  
  return(list(
    summary = performance_summary,
    all_folds = all_performance
  ))
}


#' Aggregate All Three Models at Once
#'
#' Convenience function to aggregate performance for all three models.
#'
#' @param outer_fold_results List of fold results from outer CV loop
#' @param round_digits Number of decimal places for rounding (default: 3)
#' @param conf_level Confidence level for intervals (default: 0.95)
#' @param verbose Print summary to console (default: TRUE)
#'
#' @return List containing aggregated results for all three models
#'
#' @export
aggregate_all_models <- function(outer_fold_results,
                                 round_digits = 3,
                                 conf_level = 0.95,
                                 verbose = TRUE) {
  
  models <- c("FGR_MRS", "FGR_CLIN", "FGR_COMBINED")
  results <- list()
  
  for (model in models) {
    results[[model]] <- aggregate_cv_performance(
      outer_fold_results = outer_fold_results,
      model_name = model,
      round_digits = round_digits,
      conf_level = conf_level,
      verbose = verbose
    )
  }
  
  # Print comparative summary
  if (verbose) {
    cat("\n========== COMPARATIVE SUMMARY ==========\n")
    for (model in models) {
      mean_auc_row <- results[[model]]$summary[results[[model]]$summary$metric == "mean_auc", ]
      cat(sprintf("%-15s: AUC = %.3f ± %.3f\n", 
                  model, 
                  mean_auc_row$mean, 
                  mean_auc_row$se))
    }
    
    # Calculate improvements
    mrs_auc <- results[["FGR_MRS"]]$summary[results[["FGR_MRS"]]$summary$metric == "mean_auc", "mean"]
    clin_auc <- results[["FGR_CLIN"]]$summary[results[["FGR_CLIN"]]$summary$metric == "mean_auc", "mean"]
    comb_auc <- results[["FGR_COMBINED"]]$summary[results[["FGR_COMBINED"]]$summary$metric == "mean_auc", "mean"]
    
    cat("\n--- Incremental Value ---\n")
    cat(sprintf("MRS vs Clinical:   ΔAUC = %+.3f\n", mrs_auc - clin_auc))
    cat(sprintf("Combined vs MRS:   ΔAUC = %+.3f\n", comb_auc - mrs_auc))
    cat(sprintf("Combined vs Clin:  ΔAUC = %+.3f\n", comb_auc - clin_auc))
  }
  
  return(results)
}


#' Assess Feature Selection Stability for Penalized Fine-Gray Models
#'
#' Evaluate which CpGs were consistently selected for methylation risk score (MRS)
#' construction across cross-validation folds.
#'
#' @param outer_fold_results List of fold results from nested CV
#' @param verbose Print summary statistics (default: TRUE)
#'
#' @return List containing:
#'   \item{stability_metrics}{Data frame with CpG statistics}
#'   \item{selection_matrix}{Data frame with CpG column + binary fold indicators}
#'   \item{coefficient_matrix}{Data frame with CpG column + actual coefficients per fold}
#'
#' @details
#' For each CpG, calculates:
#' - Selection frequency across folds
#' - Direction consistency (all positive or all negative)
#' - Mean, median, and SD of coefficients (when selected)
#' - Number of positive/negative coefficient occurrences
#'
#' @export
assess_finegray_stability_threefg <- function(outer_fold_results, verbose = TRUE) {
  
  # Setup
  n_folds <- length(outer_fold_results)
  
  # Extract penalized FG coefficient tables from all folds
  # UPDATED: Changed from coefficients_table to coefficients
  coef_list <- lapply(outer_fold_results, function(fold) {
    fold$penalized_fg$coefficients
  })
  
  # Get all unique CpGs selected across any fold
  all_cpgs <- unique(unlist(lapply(coef_list, function(df) df$feature)))
  
  # Build combined table (one row per CpG)
  combined_table <- do.call(rbind, lapply(all_cpgs, function(cpg) {
    
    # Extract penalized FG coefficients for this CpG from each fold
    pen_fg_coefs <- sapply(coef_list, function(df) {
      cpg_idx <- which(df$feature == cpg)
      if (length(cpg_idx) == 0) return(NA)  # NA if not selected
      df$pen_fg_coef[cpg_idx]
    })
    
    # Binary selection indicators (NA means not selected)
    is_selected <- as.numeric(!is.na(pen_fg_coefs))
    
    # Selection statistics
    n_selected <- sum(is_selected)
    selection_freq <- n_selected / n_folds
    
    # Count positive/negative coefficients (only for selected folds)
    selected_coefs <- pen_fg_coefs[!is.na(pen_fg_coefs)]
    n_positive <- sum(selected_coefs > 0)
    n_negative <- sum(selected_coefs < 0)
    
    # Coefficient statistics (only for folds where selected)
    if (length(selected_coefs) > 0) {
      direction_consistent <- all(selected_coefs > 0) | all(selected_coefs < 0)
      mean_coef <- mean(selected_coefs)
      sd_coef <- sd(selected_coefs)
      median_coef <- median(selected_coefs)
    } else {
      direction_consistent <- FALSE
      mean_coef <- NA
      sd_coef <- NA
      median_coef <- NA
    }
    
    # Build complete row
    row_df <- data.frame(
      feature = cpg,
      n_selected = n_selected,
      selection_freq = round(selection_freq, 3),
      direction_consistent = direction_consistent,
      n_positive = n_positive,
      n_negative = n_negative,
      mean_coef = round(mean_coef, 4),
      median_coef = round(median_coef, 4),
      sd_coef = round(sd_coef, 4),
      stringsAsFactors = FALSE
    )
    
    # Add fold selection indicators (0/1)
    fold_selection_df <- as.data.frame(t(is_selected))
    colnames(fold_selection_df) <- paste0("Fold", 1:n_folds, "_selected")
    
    # Add fold coefficients (actual values or NA)
    fold_coef_df <- as.data.frame(t(pen_fg_coefs))
    colnames(fold_coef_df) <- paste0("Fold", 1:n_folds, "_coef")
    
    cbind(row_df, fold_selection_df, fold_coef_df)
  }))
  
  # Sort by selection frequency, then by absolute mean coefficient
  combined_table <- combined_table[order(
    combined_table$selection_freq,
    abs(combined_table$mean_coef),
    decreasing = TRUE,
    na.last = TRUE
  ), ]
  rownames(combined_table) <- NULL
  
  # Split into three outputs
  # 1. Metrics table: CpG + statistics (no fold columns)
  metric_cols <- c("feature", "n_selected", "selection_freq",
                   "direction_consistent", "n_positive", "n_negative", 
                   "mean_coef", "median_coef", "sd_coef")
  stability_metrics <- combined_table[, metric_cols]
  
  # 2. Selection matrix: CpG + fold selections (0/1)
  fold_selection_cols <- grep("_selected$", names(combined_table), value = TRUE)
  selection_matrix <- combined_table[, c("feature", fold_selection_cols)]
  
  # 3. Coefficient matrix: CpG + fold coefficients (actual values)
  fold_coef_cols <- grep("_coef$", names(combined_table), value = TRUE)
  coefficient_matrix <- combined_table[, c("feature", fold_coef_cols)]
  
  # Print summary (if requested)
  if (verbose) {
    cat("\n========== PENALIZED FINE-GRAY CpG STABILITY ==========\n")
    cat(sprintf("Total folds: %d\n\n", n_folds))
    cat(sprintf("CpGs ever selected: %d\n", 
                sum(stability_metrics$n_selected > 0)))
    cat(sprintf("Direction-consistent CpGs: %d\n", 
                sum(stability_metrics$direction_consistent, na.rm = TRUE)))
    cat(sprintf("CpGs selected in all folds: %d\n", 
                sum(stability_metrics$selection_freq == 1)))
    cat(sprintf("CpGs selected in ≥80%% folds: %d\n", 
                sum(stability_metrics$selection_freq >= 0.8)))
    cat(sprintf("CpGs selected in ≥40%% folds: %d\n\n", 
                sum(stability_metrics$selection_freq >= 0.4)))
    
    # Show most stable CpGs (selected in at least 40% of folds)
    stable_cpgs <- stability_metrics[stability_metrics$selection_freq >= 0.4, ]
    
    if (nrow(stable_cpgs) > 0) {
      cat("Most stable CpGs (≥40% selection frequency):\n")
      print(stable_cpgs)
    } else {
      cat("No CpGs selected in ≥40% of folds\n")
    }
    
    cat("\n")
  }
  
  return(list(
    stability_metrics = stability_metrics,
    selection_matrix = selection_matrix,
    coefficient_matrix = coefficient_matrix
  ))
}


################################################################################
# END OF HELPER FUNCTIONS
################################################################################


## risk: an object class of "get_risk"  
## marker the continuous biomarquer 
## status the event for a binary outcome or status for a time-to-event oucome.
plotpredictiveness <- function(risk,marker,status){
  xlim <- c(0,100)
  breaks = seq(xlim[1], xlim[2], length.out = 5)
  k <- length(risk)
  x1<- sort(marker)
  y1 <- (((1:k)*100)/k)
  z <- sort(risk)
  prev<-sum(status)/length(status)
  Qt<-y1[which.min(abs(z-prev))]
  Tr<-quantile(x1, Qt/100)
  plot(y1,z,type='l',xlab="(%)population \n marker Value",ylab="Risk given marker (%)",main="Estimating the predictiveness curve of a continuous biomarker",
       ylim=c(0,1),axes=FALSE)
  axis(1,at= c(seq(from=0,to=100,by=20),Qt), label = c(seq(from=0,to=100,by=20),round(Qt, 2)))
  axis(2,at=seq(from=0,to=1,by=0.2),labels=seq(from=0,to=1,by=0.2)*100,las=2)
  axis(1,at= c(breaks,Qt), label = round(c(quantile(x1, prob = breaks/100),Tr), 2),pos=-0.26)
  abline(h=prev,col="blue",lty=2)
  abline(v=Qt,col="blue",lty=2)
  return(Threshold=Tr)
}



loadRData <- function(file.path){
  load(file.path)
  get(ls()[ls() != "file.path"])
}