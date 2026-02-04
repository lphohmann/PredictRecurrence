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

filter_by_univariate_cox <- function(X_train, y_train, 
                                     selection_method = "top_n",
                                     top_n = 5000,
                                     p_threshold = 0.05) {
  # Filter features by univariate Cox regression association with outcome
  #
  # Performs univariate Cox regression for each feature and selects those
  # most strongly associated with the outcome. This captures features that
  # discriminate between events and non-events, even if they have low variance.
  #
  # Args:
  #   X_train: Training data matrix (samples x features)
  #   y_train: Survival object (from Surv())
  #   selection_method: How to select features:
  #     - "top_n": Keep top N features by p-value (default)
  #     - "p_threshold": Keep all features with p < threshold
  #     - "top_proportion": Keep top X% by p-value
  #   top_n: Number of features to keep (if method = "top_n")
  #   p_threshold: P-value threshold (if method = "p_threshold")
  #
  # Returns:
  #   Filtered matrix with features associated with outcome
  
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
  # Store p-values and handle errors gracefully
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

################################################################################
prepare_filtered_features <- function(X, 
                                      vars_preserve = NULL, 
                                      variance_quantile = 0.75,
                                      apply_cox_filter = FALSE,
                                      y_train = NULL,
                                      cox_selection_method = "top_n",
                                      cox_top_n = 5000,
                                      cox_p_threshold = 0.05) {
  # Apply variance filtering (and optionally univariate Cox) while preserving specified variables
  #
  # Two-stage filtering approach:
  # 1. Variance filtering: Removes low-variance, uninformative features
  # 2. Univariate Cox (optional): Keeps features associated with outcome
  #
  # Clinical variables are preserved at all stages to ensure they remain
  # in the model regardless of variance or univariate association.
  #
  # Args:
  #   X: Feature matrix (samples x features)
  #   vars_preserve: Vector of variable names to preserve (e.g., clinical variables)
  #   variance_quantile: Variance filter threshold (default: 0.75 = top 25%)
  #   apply_cox_filter: Whether to apply univariate Cox filtering (default: FALSE)
  #   y_train: Survival object (required if apply_cox_filter = TRUE)
  #   cox_selection_method: How to select features ("top_n", "p_threshold", "top_proportion")
  #   cox_top_n: Number of features to keep (for "top_n" method)
  #   cox_p_threshold: P-value threshold (for "p_threshold" method)
  #
  # Returns:
  #   Filtered feature matrix with preserved variables included
  
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
  
  # --------------------------------------------------------------------------
  # STEP 1: Variance Filtering
  # --------------------------------------------------------------------------
  
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
  
  # --------------------------------------------------------------------------
  # STEP 2: Univariate Cox Filtering (Optional)
  # --------------------------------------------------------------------------
  
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
  
  # --------------------------------------------------------------------------
  # Combine Filtered and Preserved Features
  # --------------------------------------------------------------------------
  
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




##########################################
##########################################
##########################################
##########################################



################################################################################


# Function to get train/validation splits for a given inner fold


cv_data_prep <- function(X, clinical, train_idx, test_idx) {
  # Subset features
  X_train <- X[train_idx, , drop = FALSE]
  X_test   <- X[test_idx, , drop = FALSE]
  # Subset clinical data
  clinical_train <- clinical[train_idx, , drop = FALSE]
  clinical_test   <- clinical[test_idx, , drop = FALSE]
  # Return as list
  list(
    X_train = X_train,
    X_test   = X_test,
    clinical_train = clinical_train,
    clinical_test   = clinical_test
  )
}


################################################################################

scale_continuous_features <- function(X_train, X_test = NULL, dont_scale=NULL) {
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
# 
# cv_glmnet_alpha_grid <- function(X_train, y_train, alpha_grid, penalty_factor = NULL, 
#                                  foldid = NULL, family = "cox", seed = 123) {
#   # Cross-validate glmnet across multiple alpha values
#   #
#   # Alpha controls elastic net mixing:
#   #   - alpha = 0: Ridge regression (L2 penalty)
#   #   - alpha = 1: LASSO (L1 penalty)
#   #   - 0 < alpha < 1: Elastic net (combined L1/L2)
#   #
#   # Args:
#   #   X_train: Training feature matrix
#   #   y_train: Survival object (from Surv())
#   #   alpha_grid: Vector of alpha values to test
#   #   penalty_factor: Feature-specific penalties (0 = no penalty)
#   #   foldid: CV fold assignments
#   #   family: "cox" for survival analysis
#   #   seed: Random seed for reproducibility
#   #
#   # Returns:
#   #   List of CV results for each alpha value
#   
#   cv_results <- list()
#   
#   for (alpha_val in alpha_grid) {
#     set.seed(seed)
#     cv_fit <- cv.glmnet(
#       x = as.matrix(X_train),
#       y = y_train,
#       family = family,
#       alpha = alpha_val,
#       penalty.factor = penalty_factor,
#       foldid = foldid,
#       standardize = TRUE
#     )
#     
#     # Extract performance at optimal lambda
#     perf_min <- cv_fit$cvm[cv_fit$lambda == cv_fit$lambda.min]
#     perf_se <- cv_fit$cvsd[cv_fit$lambda == cv_fit$lambda.min]
#     
#     # Get selected features
#     beta <- coef(cv_fit, s = "lambda.min")
#     selected_features <- rownames(beta)[as.vector(beta != 0)]
#     
#     cv_results[[as.character(alpha_val)]] <- list(
#       alpha = alpha_val,
#       lambda = cv_fit$lambda.min,
#       cvm_min = perf_min,
#       cvm_se = perf_se,
#       n_features = length(selected_features),
#       features = selected_features,
#       fit = cv_fit
#     )
#     
#     cat(sprintf(
#       "  Alpha=%.2f: CV-deviance=%.4f (SE=%.4f), Lambda=%.6f, Features=%d\n",
#       alpha_val, perf_min, perf_se, cv_fit$lambda.min, length(selected_features)
#     ))
#   }
#   
#   return(cv_results)
# }
# 
# ################################################################################
# 
# tune_and_fit_coxnet <- function(X_train, y_train, clinical_train, event_col,
#                                 alpha_grid, penalty_factor = NULL, 
#                                 n_inner_folds = 5, seed = 123,
#                                 outcome_name = "outcome") {
#   # Tune CoxNet hyperparameters via inner CV, then fit final model
#   #
#   # Performs stratified cross-validation on event occurrence to find
#   # optimal alpha and lambda values, then fits model on full training set.
#   #
#   # Args:
#   #   X_train: Training feature matrix
#   #   y_train: Survival object
#   #   clinical_train: Clinical data for stratification
#   #   event_col: Column name for event indicator
#   #   alpha_grid: Vector of alpha values to test
#   #   penalty_factor: Feature-specific penalties
#   #   n_inner_folds: Number of CV folds
#   #   seed: Random seed
#   #   outcome_name: Name for logging
#   #
#   # Returns:
#   #   List with CV results, best hyperparameters, and fitted model
#   
#   cat(sprintf("\n--- Inner CV: Tuning CoxNet for %s ---\n", outcome_name))
#   
#   # Create stratified inner folds based on event occurrence
#   set.seed(seed)
#   inner_folds <- createFolds(
#     y = clinical_train[[event_col]],
#     k = n_inner_folds,
#     list = TRUE,
#     returnTrain = FALSE
#   )
#   
#   # Convert to foldid vector for cv.glmnet
#   foldid <- rep(NA, nrow(X_train))
#   for (i in 1:n_inner_folds) {
#     foldid[inner_folds[[i]]] <- i
#   }
#   
#   # Test each alpha value
#   cv_results <- cv_glmnet_alpha_grid(
#     X_train = X_train,
#     y_train = y_train,
#     alpha_grid = alpha_grid,
#     penalty_factor = penalty_factor,
#     foldid = foldid
#   )
#   
#   # Select best hyperparameters based on CV deviance
#   best_idx <- which.min(sapply(cv_results, function(x) x$cvm_min))
#   best_result <- cv_results[[best_idx]]
#   best_alpha <- best_result$alpha
#   best_lambda <- best_result$lambda
#   
#   cat(sprintf("Best hyperparameters: alpha=%.2f, lambda=%.6f\n", 
#               best_alpha, best_lambda))
#   
#   # Fit final model on whole training set with best hyperparameters
#   final_fit <- glmnet( 
#     x = as.matrix(X_train),
#     y = y_train,
#     family = "cox",
#     alpha = best_alpha,
#     lambda = best_lambda,
#     penalty.factor = penalty_factor
#   )
#   
#   return(list(
#     cv_results = cv_results,
#     best_result = best_result,
#     best_alpha = best_alpha,
#     best_lambda = best_lambda,
#     final_fit = final_fit
#   ))
# }
# 

# NEW HELPER FUNCTION: Extract feature selections AND coefficient signs from each CV fold
get_fold_selections <- function(X_train, y_train, foldid, alpha, lambda_target, 
                                penalty_factor = NULL, family = "cox") {
  # For each fold, train model on k-1 folds and extract selected features + coefficient signs
  #
  # Args:
  #   X_train: Training feature matrix
  #   y_train: Survival object
  #   foldid: Fold assignments (vector of fold numbers)
  #   alpha: Alpha value for this CV run
  #   lambda_target: The lambda value to use for feature extraction (typically lambda.min)
  #   penalty_factor: Feature-specific penalties
  #   family: Model family
  #
  # Returns:
  #   List with:
  #     - selection_matrix: Binary matrix (rows = features, columns = folds, 1 = selected)
  #     - sign_matrix: Sign of coefficients (1 = positive, -1 = negative, 0 = not selected)
  
  n_folds <- max(foldid, na.rm = TRUE)
  feature_names <- colnames(X_train)
  n_features <- length(feature_names)
  
  # Initialize binary selection matrix
  selection_matrix <- matrix(0, nrow = n_features, ncol = n_folds,
                             dimnames = list(feature_names, paste0("fold", 1:n_folds)))
  
  # Initialize coefficient sign matrix (NEW)
  sign_matrix <- matrix(0, nrow = n_features, ncol = n_folds,
                        dimnames = list(feature_names, paste0("fold", 1:n_folds)))
  
  # For each fold, train on other folds and extract selections at lambda_target
  for (fold in 1:n_folds) {
    # Identify samples in this fold
    train_idx <- which(foldid != fold)
    
    # Train model on k-1 folds
    fold_fit <- glmnet(
      x = as.matrix(X_train[train_idx, , drop = FALSE]),
      y = y_train[train_idx],
      family = family,
      alpha = alpha,
      lambda = lambda_target,
      penalty.factor = penalty_factor,
      standardize = TRUE
    )
    
    # Extract coefficients at target lambda
    fold_coef <- as.vector(coef(fold_fit, s = lambda_target))
    
    # Mark selected features (non-zero coefficients)
    selected_idx <- which(fold_coef != 0)
    
    if (length(selected_idx) > 0) {
      selection_matrix[selected_idx, fold] <- 1
      # Store coefficient sign: +1 for positive, -1 for negative
      sign_matrix[selected_idx, fold] <- sign(fold_coef[selected_idx])
    }
  }
  
  return(list(
    selection_matrix = selection_matrix,
    sign_matrix = sign_matrix
  ))
}


################################################################################

cv_glmnet_alpha_grid <- function(X_train, y_train, alpha_grid, penalty_factor = NULL, 
                                 foldid = NULL, family = "cox", seed = 123,
                                 compute_stability = FALSE) {
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
  #   compute_stability: Whether to compute fold-level stability (default FALSE)
  #
  # Returns:
  #   List of CV results for each alpha value
  #   If compute_stability=TRUE, includes stability_matrix, sign_matrix, stability_scores
  
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
    perf_se <- cv_fit$cvsd[cv_fit$lambda == cv_fit$lambda.1se]
    
    # Get selected features from full CV model
    beta <- coef(cv_fit, s = "lambda.min")
    selected_features <- rownames(beta)[as.vector(beta != 0)]
    
    # Initialize result list with original fields
    result <- list(
      alpha = alpha_val,
      lambda = cv_fit$lambda.min,
      cvm_min = perf_min,
      cvm_se = perf_se,
      n_features = length(selected_features),
      features = selected_features,
      fit = cv_fit
    )
    
    # OPTIONAL: Compute fold-level feature selections for stability analysis
    # this is done for the best labda for this alpha, same 3splti resampling used
    if (compute_stability && !is.null(foldid)) {
      fold_results <- get_fold_selections(
        X_train = X_train,
        y_train = y_train,
        foldid = foldid,
        alpha = alpha_val, # that alpha
        lambda_target = cv_fit$lambda.min, # prev selected lambda
        penalty_factor = penalty_factor,
        family = family
      )
      
      selection_matrix <- fold_results$selection_matrix
      sign_matrix <- fold_results$sign_matrix
      
      # Calculate per-feature stability scores (proportion of folds where selected)
      n_folds <- ncol(selection_matrix)
      selected_any <- rowSums(selection_matrix) > 0
      
      stability_scores <- rowSums(selection_matrix) / n_folds
      stability_scores <- stability_scores[selected_any]
      
      # Calculate sign consistency for each feature
      # For each feature: check if all non-zero coefficients have same sign
      sign_consistency <- apply(sign_matrix, 1, function(signs) {
        nonzero_signs <- signs[signs != 0]
        if (length(nonzero_signs) == 0) return(NA)
        # All same sign = consistent (all positive OR all negative)
        all(nonzero_signs > 0) || all(nonzero_signs < 0)
      })
      
      # Create combined dataframe with stability info
      stability_info <- data.frame(
        feature = names(stability_scores),
        selection_freq = stability_scores,
        sign_consistent = sign_consistency[names(stability_scores)],
        stringsAsFactors = FALSE
      )
      stability_info <- stability_info[order(stability_info$selection_freq, decreasing = TRUE), ]
      rownames(stability_info) <- NULL
      
      # Add to result
      result$stability_matrix <- selection_matrix
      result$sign_matrix <- sign_matrix
      result$stability_scores <- stability_scores
      result$stability_info <- stability_info
      
      # Print stability summary
      if (nrow(stability_info) > 0) {
        high_stability <- sum(stability_info$selection_freq >= 0.8)
        med_stability <- sum(stability_info$selection_freq >= 0.6 & 
                               stability_info$selection_freq < 0.8)
        sign_consistent_count <- sum(stability_info$sign_consistent, na.rm = TRUE)
        
        cat(sprintf(
          "    Stability: %d features ≥80%%, %d features 60-80%%, mean=%.2f\n",
          high_stability, med_stability, mean(stability_info$selection_freq)
        ))
        cat(sprintf(
          "    Sign consistency: %d/%d features have consistent coefficient direction\n",
          sign_consistent_count, nrow(stability_info)
        ))
      }
    }
    
    cv_results[[as.character(alpha_val)]] <- result
    
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
                                outcome_name = "outcome",
                                compute_stability = FALSE) {
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
  #   compute_stability: Whether to compute feature selection stability (default FALSE)
  #
  # Returns:
  #   List with CV results, best hyperparameters, and fitted model
  #   If compute_stability=TRUE, cv_results includes stability_matrix, sign_matrix, stability_info
  
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
    foldid = foldid,
    compute_stability = compute_stability
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
  # Also tracks the number of features in each fold's Fine-Gray model.
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
  
  # Extract number of features in each fold's FG model
  n_features_per_fold <- sapply(outer_fold_results, function(fold) {
    coef_df <- fold$fold_model_coefficients
    sum(coef_df$fg_coef != 0)
  })
  
  # Add feature count column to all_performance
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

################################################################################
# FEATURE SELECTION STABILITY
################################################################################
assess_finegray_stability <- function(outer_fold_results, 
                                      clinical_features = NULL,
                                      verbose = TRUE) {
  # Assess Fine-Gray feature selection stability across CV folds
  #
  # Args:
  #   outer_fold_results: List of fold results from nested CV
  #   clinical_features: Character vector of clinical feature names (optional)
  #   verbose: Whether to print summary statistics
  #
  # Returns:
  #   List with:
  #     - stability_metrics: Dataframe with feature statistics
  #     - selection_matrix: Dataframe with feature column + binary fold indicators
  
  # ============================================================================
  # Setup
  # ============================================================================
  n_folds <- length(outer_fold_results)
  
  # Extract coefficient tables from all folds
  coef_list <- lapply(outer_fold_results, function(fold) {
    fold$fold_model_coefficients
  })
  
  # Get all unique features selected across any fold
  all_features <- unique(unlist(lapply(coef_list, function(df) df$feature)))
  
  # ============================================================================
  # Build combined table (one row per feature)
  # ============================================================================
  combined_table <- do.call(rbind, lapply(all_features, function(feat) {
    
    # Extract FG coefficients for this feature from each fold
    fg_coefs <- sapply(coef_list, function(df) {
      feature_idx <- which(df$feature == feat)
      if (length(feature_idx) == 0) return(0)
      df$fg_coef[feature_idx]
    })
    
    # Binary selection indicators
    is_selected <- as.numeric(fg_coefs != 0)
    
    # Selection statistics
    n_selected <- sum(is_selected)
    selection_freq <- n_selected / n_folds
    n_positive <- sum(fg_coefs > 0)
    n_negative <- sum(fg_coefs < 0)
    
    # Coefficient statistics (only for folds where selected)
    nonzero_coefs <- fg_coefs[fg_coefs != 0]
    if (length(nonzero_coefs) > 0) {
      direction_consistent <- all(nonzero_coefs > 0) | all(nonzero_coefs < 0)
      mean_coef <- mean(nonzero_coefs)
      sd_coef <- sd(nonzero_coefs)
    } else {
      direction_consistent <- FALSE
      mean_coef <- 0
      sd_coef <- 0
    }
    
    # Clinical feature flag
    is_clinical <- if (!is.null(clinical_features)) {
      feat %in% clinical_features
    } else {
      FALSE
    }
    
    # Build complete row
    row_df <- data.frame(
      feature = feat,
      is_clinical = is_clinical,
      n_selected = n_selected,
      selection_freq = round(selection_freq, 3),
      direction_consistent = direction_consistent,
      n_positive = n_positive,
      n_negative = n_negative,
      mean_coef = round(mean_coef, 3),
      sd_coef = round(sd_coef, 3),
      stringsAsFactors = FALSE
    )
    
    # Add fold columns
    fold_df <- as.data.frame(t(is_selected))
    colnames(fold_df) <- paste0("Fold", 1:n_folds)
    
    cbind(row_df, fold_df)
  }))
  
  # Sort by selection frequency
  combined_table <- combined_table[order(combined_table$selection_freq, 
                                         decreasing = TRUE), ]
  rownames(combined_table) <- NULL
  
  # ============================================================================
  # Split into two outputs
  # ============================================================================
  
  # Metrics table: feature + statistics (no fold columns)
  metric_cols <- c("feature", "is_clinical", "n_selected", "selection_freq",
                   "direction_consistent", "n_positive", "n_negative", 
                   "mean_coef", "sd_coef")
  stability_metrics <- combined_table[, metric_cols]
  
  # Selection matrix: feature + fold selections
  fold_cols <- grep("^Fold", names(combined_table), value = TRUE)
  selection_matrix <- combined_table[, c("feature", fold_cols)]
  
  # ============================================================================
  # Print summary (if requested)
  # ============================================================================
  if (verbose) {
    cat("\n========== FINE-GRAY FEATURE STABILITY ==========\n")
    cat(sprintf("Total folds: %d\n\n", n_folds))
    cat(sprintf("Features ever selected: %d\n", 
                sum(stability_metrics$n_selected > 0)))
    
    if (!is.null(clinical_features)) {
      cat(sprintf("Clinical features: %d\n", 
                  sum(stability_metrics$is_clinical, na.rm = TRUE)))
    }
    
    cat(sprintf("Direction-consistent features: %d\n\n", 
                sum(stability_metrics$direction_consistent, na.rm = TRUE)))
    
    # Show top 10 most frequently selected features
    if (nrow(stability_metrics[stability_metrics$n_selected > 0, ]) > 0) {
      cat("Selected features:\n")
      print(stability_metrics[stability_metrics$n_selected > 0, ])
    }
    cat("\n")
  }
  
  # ============================================================================
  # Return results
  # ============================================================================
  return(list(
    stability_metrics = stability_metrics,
    selection_matrix = selection_matrix
  ))
}


################################################################################
# FINE-GRAY fitting
################################################################################

fit_fine_gray_model <- function(fgr_data, cr_time, cr_event, cause = 1) {

  cat(sprintf("\n--- Fitting Fine-Gray Model ---\n"))
  
  # Identify feature columns by excluding the outcome columns
  feature_cols <- setdiff(
    colnames(fgr_data), 
    c(cr_time, cr_event)
  )
  
  # Build the model formula
  formula_str <- sprintf(
    "Hist(%s, %s) ~ %s",
    cr_time,                              # First %s gets the time column name
    cr_event,                             # Second %s gets the event column name
    paste(feature_cols, collapse = " + ") # Third %s gets all features joined
  )
  
  # Convert string to formula object for modeling
  formula_fg <- as.formula(formula_str)
  
  # Fit the Fine-Gray model
  fgr_model <- FGR(
    formula = formula_fg,
    data    = fgr_data,
    cause   = cause
  )
  
  cat(sprintf("Fine-Gray model fitted with %d features\n", length(feature_cols)))
  
  # Return model and metadata
  return(list(
    model = fgr_model,
    formula = formula_fg
  ))
}

################################################################################
# PENALIZED FINE-GRAY fitting
################################################################################

fit_penalized_fine_gray <- function(fgr_data, cr_time, cr_event, 
                                    cause = 1, penalty = "LASSO",
                                    n_lambda = 25) {
  
  cat(sprintf("\n--- Fitting Penalized Fine-Gray Model (%s) ---\n", penalty))
  
  # Load required package
  if (!requireNamespace("fastcmprsk", quietly = TRUE)) {
    stop("Package 'fastcmprsk' is required. Install with: install.packages('fastcmprsk')")
  }
  
  # Identify feature columns by excluding the outcome columns
  # These will be the predictors in our model
  feature_cols <- setdiff(
    colnames(fgr_data), 
    c(cr_time, cr_event)
  )
  
  # Build the model formula programmatically
  # Crisk() creates the competing risks outcome object
  # The right side includes all features joined with ' + '
  formula_str <- sprintf(
    "Crisk(%s, %s) ~ %s",
    cr_time,                              # Time column name
    cr_event,                             # Event type column name
    paste(feature_cols, collapse = " + ") # All features joined
  )
  
  # Convert string to formula object
  formula_fg <- as.formula(formula_str)
  
  # Fit the penalized Fine-Gray model
  penalized_model <- fastcmprsk::fastCrrp(
    formula = formula_fg,
    data    = fgr_data,
    penalty = "LASSO",      # "LASSO", "SCAD", "MCP", or "ridge"
    lambda  = NULL # Sequence of tuning parameters
  )
  
  cat(sprintf("Penalized Fine-Gray model fitted with %d features\n", 
              length(feature_cols)))
  cat(sprintf("Solution path contains %d lambda values\n", 
              length(penalized_model$lambda)))
  
  # Return model and metadata
  return(list(
    model = penalized_model,
    formula = formula_fg,
    lambda_path = lambda_path,
    penalty = penalty,
    n_features = length(feature_cols)
  ))
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
    HR_magnitude = HR_magnitude,  # CORRECTED: proper effect magnitude
    wald_z = fg_coef / fg_se,
    abs_wald_z = abs(fg_coef / fg_se),
    p_value = 2 * pnorm(abs(fg_coef / fg_se), lower.tail = FALSE),  # CORRECTED: two-tailed test
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
# PLOTTING FUNCTIONS FOR FINE-GRAY RESULTS
# Add these to your finegray_functions.R file
################################################################################

#' Publication-quality ggplot theme
get_publication_theme <- function() {
  theme_bw(base_size = 12) +
    theme(
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "white"),
      legend.position = "right"
    )
}

#' Figure 1: Feature Selection Stability Heatmap
#'
#' Shows which features were selected in each CV fold as a binary heatmap.
plot_stability_heatmap <- function(plot_dir, features_pooled_all, outer_fold_results, N_OUTER_FOLDS) {
  
  cat("Creating Figure 1: Feature selection stability heatmap...\n")
  
  # Get completed folds only
  completed_fold_indices <- which(!sapply(outer_fold_results, is.null))
  n_completed_folds <- length(completed_fold_indices)
  
  if (n_completed_folds == 0) {
    cat("  WARNING: No completed folds. Skipping stability heatmap.\n\n")
    return(invisible(NULL))
  }
  
  # Create binary selection matrix
  selection_matrix <- sapply(completed_fold_indices, function(i) {
    fold_features <- outer_fold_results[[i]]$features_pooled
    as.integer(features_pooled_all %in% fold_features)
  })
  
  rownames(selection_matrix) <- features_pooled_all
  colnames(selection_matrix) <- paste0("Fold ", completed_fold_indices)
  
  # Order features by selection frequency
  selection_freq <- rowMeans(selection_matrix)
  ordered_features <- names(sort(selection_freq, decreasing = TRUE))
  
  # Reshape for ggplot
  sel_melted <- reshape2::melt(selection_matrix)
  colnames(sel_melted) <- c("Feature", "Fold", "Selected")
  sel_melted$Feature <- factor(sel_melted$Feature, levels = rev(ordered_features))
  
  # Create plot
  p <- ggplot2::ggplot(sel_melted, ggplot2::aes(x = Fold, y = Feature, fill = factor(Selected))) +
    ggplot2::geom_tile(color = "white", size = 0.5) +
    ggplot2::scale_fill_manual(
      values = c("0" = "grey90", "1" = "steelblue"),
      labels = c("Not selected", "Selected"),
      name = ""
    ) +
    ggplot2::labs(
      title = "Feature Selection Stability Across CV Folds",
      subtitle = paste0("All ", length(features_pooled_all), " features in final Fine-Gray model"),
      x = "Cross-Validation Fold",
      y = "Feature"
    ) +
    get_publication_theme() +
    ggplot2::theme(
      axis.text.y = ggplot2::element_text(size = 7),
      panel.grid = ggplot2::element_blank()
    )
  
  # Save
  ggplot2::ggsave(
    file.path(plot_dir, "1_stability_heatmap.pdf"),
    p, width = 8, height = max(10, length(features_pooled_all) * 0.25)
  )
  ggplot2::ggsave(
    file.path(plot_dir, "1_stability_heatmap.png"),
    p, width = 8, height = max(10, length(features_pooled_all) * 0.25), dpi = 300
  )
  
  cat("  >> Saved: 1_stability_heatmap.pdf/.png\n\n")
}

#' Figure 2: Variable Importance Forest Plot
#'
#' Forest plot showing hazard ratios with confidence intervals.
plot_variable_importance <- function(plot_dir, vimp_fg_final, features_rfi_all, 
                                     features_death_all, has_clinical) {
  
  cat("Creating Figure 2: Variable importance forest plot...\n")
  
  vimp_all <- vimp_fg_final
  
  # Add CoxNet selection annotation
  vimp_all$selected_rfi <- vimp_all$feature %in% features_rfi_all
  vimp_all$selected_death <- vimp_all$feature %in% features_death_all
  vimp_all$selection_source <- ifelse(
    vimp_all$selected_rfi & vimp_all$selected_death, "Both",
    ifelse(vimp_all$selected_rfi, "RFI only", "Death only")
  )
  
  # Add variable type labels
  if ("type" %in% colnames(vimp_all)) {
    vimp_all$var_label <- ifelse(
      vimp_all$type == "categorical",
      paste0(vimp_all$feature, " (cat)"),
      vimp_all$feature
    )
  } else {
    vimp_all$var_label <- vimp_all$feature
    vimp_all$type <- "continuous"
  }
  
  # Calculate confidence intervals
  vimp_all$HR_lower <- exp(vimp_all$coefficient - 1.96 * vimp_all$se)
  vimp_all$HR_upper <- exp(vimp_all$coefficient + 1.96 * vimp_all$se)
  
  # Order by absolute Wald z-score (already sorted from vimp_fg_final)
  vimp_all$var_label <- factor(vimp_all$var_label, levels = rev(vimp_all$var_label))
  
  # Create plot
  p <- ggplot2::ggplot(vimp_all, ggplot2::aes(x = HR, y = var_label)) +
    ggplot2::geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
    ggplot2::geom_errorbarh(
      ggplot2::aes(xmin = HR_lower, xmax = HR_upper),
      height = 0.3, size = 0.6, color = "grey40"
    ) +
    ggplot2::geom_point(ggplot2::aes(fill = selection_source, shape = type), size = 3) +
    ggplot2::scale_x_log10(
      breaks = c(0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
      labels = c("0.1", "0.25", "0.5", "1", "2", "4", "8", "16", "32"),
      limits = c(0.1, 32)  # Consistent log-scale limits
    ) +
    ggplot2::scale_fill_manual(
      values = c("RFI only" = "#E41A1C", "Both" = "#984EA3", "Death only" = "#377EB8"),
      name = "Selected by\nCoxNet"
    ) +
    ggplot2::scale_shape_manual(
      values = c("categorical" = 22, "continuous" = 21),
      name = "Variable Type",
      labels = c("Categorical (vs ref)", "Continuous (per SD)")
    ) +
    ggplot2::labs(
      title = "Variable Importance in Final Fine-Gray Model",
      subtitle = paste0("All ", nrow(vimp_all), " features, ordered by Wald z-score"),
      x = "Hazard Ratio (95% CI)",
      y = ""
    ) +
    get_publication_theme() +
    ggplot2::theme(
      legend.position = "right",
      axis.text.y = ggplot2::element_text(size = 7)
    )
  
  # Add guides conditionally
  if (has_clinical && any(vimp_all$type == "categorical")) {
    p <- p + ggplot2::guides(
      fill = ggplot2::guide_legend(override.aes = list(shape = 21, size = 4)),
      shape = ggplot2::guide_legend(override.aes = list(size = 4))
    )
  } else {
    p <- p + ggplot2::guides(
      fill = ggplot2::guide_legend(override.aes = list(shape = 21, size = 4)),
      shape = "none"
    )
  }
  
  # Save
  ggplot2::ggsave(
    file.path(plot_dir, "2_variable_importance_forest.pdf"),
    p, width = 10, height = max(10, nrow(vimp_all) * 0.25)
  )
  ggplot2::ggsave(
    file.path(plot_dir, "2_variable_importance_forest.png"),
    p, width = 10, height = max(10, nrow(vimp_all) * 0.25), dpi = 300
  )
  
  cat("  >> Saved: 2_variable_importance_forest.pdf/.png\n\n")
}

#' Figure 3: Feature Selection Comparison Bar Plot
#'
#' Shows how many features were selected by RFI, Death, or both models.
plot_feature_selection_comparison <- function(plot_dir, features_pooled_all, 
                                              features_rfi_all, features_death_all) {
  
  cat("Creating Figure 3: Feature selection comparison...\n")
  
  # Classify features
  feature_source <- data.frame(
    feature = features_pooled_all,
    selected_rfi = features_pooled_all %in% features_rfi_all,
    selected_death = features_pooled_all %in% features_death_all,
    stringsAsFactors = FALSE
  )
  
  feature_source$category <- ifelse(
    feature_source$selected_rfi & feature_source$selected_death, "Both",
    ifelse(feature_source$selected_rfi, "RFI only", "Death only")
  )
  
  # Count by category
  category_counts <- table(feature_source$category)
  category_df <- data.frame(
    category = names(category_counts),
    count = as.vector(category_counts)
  )
  category_df$category <- factor(
    category_df$category, 
    levels = c("RFI only", "Both", "Death only")
  )
  
  # Create plot
  p <- ggplot2::ggplot(category_df, ggplot2::aes(x = category, y = count, fill = category)) +
    ggplot2::geom_bar(stat = "identity", width = 0.7) +
    ggplot2::geom_text(ggplot2::aes(label = count), vjust = -0.5, size = 5) +
    ggplot2::scale_fill_manual(
      values = c("RFI only" = "#E41A1C", "Both" = "#984EA3", "Death only" = "#377EB8")
    ) +
    ggplot2::labs(
      title = "Feature Selection by CoxNet Model",
      subtitle = paste0("Total pooled features: ", length(features_pooled_all)),
      x = "Selected by",
      y = "Number of Features"
    ) +
    get_publication_theme() +
    ggplot2::theme(legend.position = "none") +
    ggplot2::ylim(0, max(category_counts) * 1.15)
  
  # Save
  ggplot2::ggsave(
    file.path(plot_dir, "3_feature_selection_by_model.pdf"),
    p, width = 8, height = 6
  )
  ggplot2::ggsave(
    file.path(plot_dir, "3_feature_selection_by_model.png"),
    p, width = 8, height = 6, dpi = 300
  )
  
  cat("  >> Saved: 3_feature_selection_by_model.pdf/.png\n\n")
}

#' Figure 4: CoxNet Coefficient Comparison
#'
#' Compares coefficients between RFI and Death models.
plot_coxnet_coefficient_comparison <- function(plot_dir, coef_comparison_final) {
  
  cat("Creating Figure 4: CoxNet coefficient comparison...\n")
  
  # Filter to features selected by at least one model
  all_coef_filtered <- coef_comparison_final[
    coef_comparison_final$cox_rfi_final_coef != 0 | 
      coef_comparison_final$cox_death_final_coef != 0, 
  ]
  
  # Reshape for plotting
  coef_long <- data.frame(
    feature = rep(all_coef_filtered$feature, 2),
    model = rep(c("CoxNet RFI", "CoxNet Death"), each = nrow(all_coef_filtered)),
    coefficient = c(all_coef_filtered$cox_rfi_final_coef, 
                    all_coef_filtered$cox_death_final_coef)
  )
  coef_long <- coef_long[coef_long$coefficient != 0, ]
  
  # Order features
  feature_order <- all_coef_filtered$feature[
    order(abs(all_coef_filtered$cox_rfi_final_coef), decreasing = TRUE)
  ]
  coef_long$feature <- factor(coef_long$feature, levels = rev(feature_order))
  
  # Determine symmetric axis limits
  max_abs_coef <- max(abs(coef_long$coefficient))
  axis_limit <- ceiling(max_abs_coef * 1.1 * 10) / 10  # Round up to nearest 0.1
  
  # Create plot
  p <- ggplot2::ggplot(coef_long, ggplot2::aes(x = coefficient, y = feature, color = model)) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
    ggplot2::geom_point(size = 3, position = ggplot2::position_dodge(width = 0.5)) +
    ggplot2::scale_color_manual(
      values = c("CoxNet RFI" = "#E41A1C", "CoxNet Death" = "#377EB8"),
      name = "Model"
    ) +
    ggplot2::scale_x_continuous(
      limits = c(-axis_limit, axis_limit)  # Symmetric around 0
    ) +
    ggplot2::labs(
      title = "CoxNet Coefficient Comparison",
      subtitle = "Features selected by elastic net for RFI or Death outcomes",
      x = "Coefficient",
      y = ""
    ) +
    get_publication_theme() +
    ggplot2::theme(
      legend.position = c(0.85, 0.15),
      axis.text.y = ggplot2::element_text(size = 7)
    )
  
  # Save
  ggplot2::ggsave(
    file.path(plot_dir, "4_coxnet_coefficient_comparison.pdf"),
    p, width = 10, height = max(8, nrow(all_coef_filtered) * 0.3)
  )
  ggplot2::ggsave(
    file.path(plot_dir, "4_coxnet_coefficient_comparison.png"),
    p, width = 10, height = max(8, nrow(all_coef_filtered) * 0.3), dpi = 300
  )
  
  cat("  >> Saved: 4_coxnet_coefficient_comparison.pdf/.png\n\n")
}

#' Figure 5: Performance Over Time
#'
#' Line plots of AUC and Brier scores across follow-up time.
plot_performance_over_time <- function(plot_dir, perf_results) {
  
  cat("Creating Figure 5: Performance over time...\n")
  
  perf_summary <- perf_results$summary
  
  # Separate metrics
  auc_metrics <- perf_summary[grep("^auc_", perf_summary$metric), ]
  brier_metrics <- perf_summary[grep("^brier_", perf_summary$metric), ]
  
  # Extract time points
  auc_metrics$time <- as.numeric(gsub("auc_(\\d+)yr", "\\1", auc_metrics$metric))
  brier_metrics$time <- as.numeric(gsub("brier_(\\d+)yr", "\\1", brier_metrics$metric))
  
  # AUC plot
  p_auc <- ggplot2::ggplot(auc_metrics, ggplot2::aes(x = time, y = mean)) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = ci_lower, ymax = ci_upper), 
                         alpha = 0.2, fill = "steelblue") +
    ggplot2::geom_line(color = "steelblue", size = 1) +
    ggplot2::geom_point(color = "steelblue", size = 3) +
    ggplot2::geom_hline(yintercept = 0.5, linetype = "dashed", color = "red", alpha = 0.5) +
    ggplot2::scale_x_continuous(breaks = unique(auc_metrics$time)) +
    ggplot2::ylim(0, 1) +
    ggplot2::labs(
      title = "Time-Dependent AUC",
      subtitle = "Mean +/- 95% CI across CV folds",
      x = "Time (years)",
      y = "AUC"
    ) +
    get_publication_theme()
  
  # Brier plot
  p_brier <- ggplot2::ggplot(brier_metrics, ggplot2::aes(x = time, y = mean)) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = ci_lower, ymax = ci_upper), 
                         alpha = 0.2, fill = "darkgreen") +
    ggplot2::geom_line(color = "darkgreen", size = 1) +
    ggplot2::geom_point(color = "darkgreen", size = 3) +
    ggplot2::scale_x_continuous(breaks = unique(brier_metrics$time)) +
    ggplot2::ylim(0, 0.4) +  # Consistent scale: 0 to 0.4 for Brier scores
    ggplot2::labs(
      title = "Time-Dependent Brier Score",
      subtitle = "Mean +/- 95% CI across CV folds (lower is better)",
      x = "Time (years)",
      y = "Brier Score"
    ) +
    get_publication_theme()
  
  # Combine
  p <- gridExtra::grid.arrange(p_auc, p_brier, ncol = 1)
  
  # Save
  ggplot2::ggsave(
    file.path(plot_dir, "5_performance_over_time.pdf"),
    p, width = 10, height = 10
  )
  ggplot2::ggsave(
    file.path(plot_dir, "5_performance_over_time.png"),
    p, width = 10, height = 10, dpi = 300
  )
  
  cat("  >> Saved: 5_performance_over_time.pdf/.png\n\n")
}

#' Figure 6: Performance Variability Boxplots
#'
#' Shows distribution of performance metrics across CV folds.
plot_performance_variability <- function(plot_dir, perf_results) {
  
  cat("Creating Figure 6: Performance variability boxplots...\n")
  
  all_folds <- perf_results$all_folds
  
  # Select key time points
  key_times <- c(1, 3, 5, 10)
  auc_cols <- paste0("auc_", key_times, "yr")
  brier_cols <- paste0("brier_", key_times, "yr")
  
  # Check which exist
  auc_cols <- auc_cols[auc_cols %in% names(all_folds)]
  brier_cols <- brier_cols[brier_cols %in% names(all_folds)]
  
  # Reshape
  auc_key <- all_folds[, c("model", auc_cols)]
  brier_key <- all_folds[, c("model", brier_cols)]
  
  auc_long <- reshape2::melt(auc_key, id.vars = "model", 
                             variable.name = "time", value.name = "AUC")
  auc_long$time <- gsub("auc_(\\d+)yr", "\\1 yr", auc_long$time)
  
  brier_long <- reshape2::melt(brier_key, id.vars = "model", 
                               variable.name = "time", value.name = "Brier")
  brier_long$time <- gsub("brier_(\\d+)yr", "\\1 yr", brier_long$time)
  
  # AUC boxplot
  p_auc <- ggplot2::ggplot(auc_long, ggplot2::aes(x = time, y = AUC)) +
    ggplot2::geom_boxplot(fill = "steelblue", alpha = 0.7, outlier.shape = NA) +
    ggplot2::geom_jitter(width = 0.1, alpha = 0.5, size = 2) +
    ggplot2::geom_hline(yintercept = 0.5, linetype = "dashed", color = "red") +
    ggplot2::ylim(0, 1) +
    ggplot2::labs(
      title = "AUC Variability Across CV Folds",
      x = "Time Point",
      y = "AUC"
    ) +
    get_publication_theme()
  
  # Brier boxplot
  p_brier <- ggplot2::ggplot(brier_long, ggplot2::aes(x = time, y = Brier)) +
    ggplot2::geom_boxplot(fill = "darkgreen", alpha = 0.7, outlier.shape = NA) +
    ggplot2::geom_jitter(width = 0.1, alpha = 0.5, size = 2) +
    ggplot2::ylim(0, 0.4) +  # Consistent scale: 0 to 0.4 for Brier scores
    ggplot2::labs(
      title = "Brier Score Variability Across CV Folds",
      x = "Time Point",
      y = "Brier Score"
    ) +
    get_publication_theme()
  
  # Combine
  p <- gridExtra::grid.arrange(p_auc, p_brier, ncol = 2)
  
  # Save
  ggplot2::ggsave(
    file.path(plot_dir, "6_performance_variability.pdf"),
    p, width = 12, height = 6
  )
  ggplot2::ggsave(
    file.path(plot_dir, "6_performance_variability.png"),
    p, width = 12, height = 6, dpi = 300
  )
  
  cat("  >> Saved: 6_performance_variability.pdf/.png\n\n")
}

#' Figure 7: ROC Curves
#'
#' Time-dependent ROC curves at key follow-up times.
plot_roc_curves <- function(plot_dir, cv_predictions, EVAL_TIMES) {
  
  cat("Creating Figure 7: ROC curves...\n")
  
  # Select time points
  roc_times <- c(1, 3, 5, 10)
  roc_times <- roc_times[roc_times %in% EVAL_TIMES]
  
  roc_list <- list()
  
  for (t in roc_times) {
    risk_col <- paste0("risk_", t, "yr")
    
    cv_predictions$outcome_at_t <- ifelse(
      cv_predictions$time <= t & cv_predictions$rfi_event == 1, 1, 0
    )
    
    eligible <- cv_predictions$time >= t | cv_predictions$rfi_event == 1
    data_t <- cv_predictions[eligible, ]
    
    # Check for sufficient data
    n_cases <- sum(data_t$outcome_at_t == 1)
    n_controls <- sum(data_t$outcome_at_t == 0)
    
    if (n_cases < 2 || n_controls < 2) {
      cat(sprintf("  WARNING: Skipping %d-year ROC (cases=%d, controls=%d)\n", 
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
    
    roc_list[[as.character(t)]] <- data.frame(
      time = paste0(t, " years (AUC = ", round(pROC::auc(roc_obj), 2), ")"),
      sensitivity = roc_obj$sensitivities,
      specificity = roc_obj$specificities
    )
  }
  
  # Check if any ROCs were generated
  if (length(roc_list) == 0) {
    cat("  WARNING: No ROC curves generated. Skipping.\n\n")
    return(invisible(NULL))
  }
  
  # Combine and plot
  roc_df <- do.call(rbind, roc_list)
  roc_df$time <- factor(roc_df$time, levels = unique(roc_df$time))
  
  p <- ggplot2::ggplot(roc_df, ggplot2::aes(x = 1 - specificity, y = sensitivity, color = time)) +
    ggplot2::geom_line(size = 1) +
    ggplot2::geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey50") +
    ggplot2::scale_color_brewer(palette = "Set1", name = "Time Point") +
    ggplot2::xlim(0, 1) +  # Consistent 0-1 scale
    ggplot2::ylim(0, 1) +  # Consistent 0-1 scale
    ggplot2::labs(
      title = "Time-Dependent ROC Curves",
      subtitle = "Prediction of RFI events across CV folds",
      x = "1 - Specificity (False Positive Rate)",
      y = "Sensitivity (True Positive Rate)"
    ) +
    ggplot2::coord_equal() +  # Square plot for proper ROC visualization
    get_publication_theme() +
    ggplot2::theme(legend.position = c(0.7, 0.3))
  
  # Save
  ggplot2::ggsave(
    file.path(plot_dir, "7_roc_curves.pdf"),
    p, width = 8, height = 8
  )
  ggplot2::ggsave(
    file.path(plot_dir, "7_roc_curves.png"),
    p, width = 8, height = 8, dpi = 300
  )
  
  cat("  >> Saved: 7_roc_curves.pdf/.png\n\n")
}

#' Figure 8: Calibration Plots
#'
#' Predicted vs observed risk at key time points.
plot_calibration <- function(plot_dir, cv_predictions, EVAL_TIMES) {
  
  cat("Creating Figure 8: Calibration plots...\n")
  
  calib_times <- c(3, 5, 10)
  calib_times <- calib_times[calib_times %in% EVAL_TIMES]
  
  calib_plots <- list()
  
  for (t in calib_times) {
    risk_col <- paste0("risk_", t, "yr")
    
    eligible <- cv_predictions$time >= t | cv_predictions$rfi_event == 1
    data_t <- cv_predictions[eligible, ]
    
    # Check for sufficient data
    n_events <- sum(data_t$time <= t & data_t$rfi_event == 1)
    n_total <- nrow(data_t)
    
    if (n_total < 10 || n_events < 5) {
      cat(sprintf("  WARNING: Skipping %d-year calibration (n=%d, events=%d)\n",
                  t, n_total, n_events))
      next
    }
    
    # Create risk deciles
    data_t$risk_group <- cut(
      data_t[[risk_col]],
      breaks = quantile(data_t[[risk_col]], probs = seq(0, 1, 0.1), na.rm = TRUE),
      include.lowest = TRUE,
      labels = FALSE
    )
    
    # Calculate observed vs predicted
    calib_data <- data.frame()
    for (g in unique(data_t$risk_group)) {
      group_data <- data_t[data_t$risk_group == g, ]
      mean_pred <- mean(group_data[[risk_col]], na.rm = TRUE)
      n_total_g <- nrow(group_data)
      n_rfi <- sum(group_data$time <= t & group_data$rfi_event == 1)
      obs_ci <- n_rfi / n_total_g
      se_obs <- sqrt(obs_ci * (1 - obs_ci) / n_total_g)
      
      calib_data <- rbind(calib_data, data.frame(
        risk_group = g,
        predicted = mean_pred,
        observed = obs_ci,
        se = se_obs,
        n = n_total_g
      ))
    }
    
    calib_data <- calib_data[!is.na(calib_data$risk_group), ]
    
    # Create plot
    p_calib <- ggplot2::ggplot(calib_data, ggplot2::aes(x = predicted, y = observed)) +
      ggplot2::geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey50") +
      ggplot2::geom_errorbar(
        ggplot2::aes(ymin = pmax(0, observed - 1.96*se), 
                     ymax = pmin(1, observed + 1.96*se)),
        width = 0.01, alpha = 0.5
      ) +
      ggplot2::geom_point(ggplot2::aes(size = n), alpha = 0.7, color = "steelblue") +
      ggplot2::geom_smooth(method = "loess", se = TRUE, color = "red", 
                           fill = "red", alpha = 0.2) +
      ggplot2::scale_size_continuous(name = "N patients", range = c(2, 6)) +
      ggplot2::xlim(0, 1) +  # Consistent 0-1 scale
      ggplot2::ylim(0, 1) +  # Consistent 0-1 scale
      ggplot2::labs(
        title = paste0("Calibration at ", t, " Years"),
        subtitle = "Predicted vs. observed cumulative incidence of RFI",
        x = "Predicted Risk",
        y = "Observed Cumulative Incidence"
      ) +
      ggplot2::coord_equal() +  # Square plot, 1:1 line at 45 degrees
      get_publication_theme()
    
    calib_plots[[as.character(t)]] <- p_calib
  }
  
  # Check if any calibration plots were generated
  if (length(calib_plots) == 0) {
    cat("  WARNING: No calibration plots generated. Skipping.\n\n")
    return(invisible(NULL))
  }
  
  # Combine and save
  if (length(calib_plots) == 3) {
    p <- gridExtra::grid.arrange(calib_plots[[1]], calib_plots[[2]], 
                                 calib_plots[[3]], ncol = 3)
    width <- 15
  } else if (length(calib_plots) == 2) {
    p <- gridExtra::grid.arrange(calib_plots[[1]], calib_plots[[2]], ncol = 2)
    width <- 10
  } else {
    p <- calib_plots[[1]]
    width <- 5
  }
  
  ggplot2::ggsave(
    file.path(plot_dir, "8_calibration_plots.pdf"),
    p, width = width, height = 5
  )
  ggplot2::ggsave(
    file.path(plot_dir, "8_calibration_plots.png"),
    p, width = width, height = 5, dpi = 300
  )
  
  cat("  >> Saved: 8_calibration_plots.pdf/.png\n\n")
}