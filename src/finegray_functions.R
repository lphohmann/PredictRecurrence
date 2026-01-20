################################################################################
# HELPER FUNCTIONS
################################################################################

load_training_data <- function(train_ids, beta_path, clinical_path) {
  # Load clinical data
  clinical_data <- read_csv(clinical_path, show_col_types = FALSE)
  clinical_data <- as.data.frame(clinical_data)
  rownames(clinical_data) <- clinical_data$Sample
  clinical_data <- clinical_data[train_ids, , drop = FALSE]
  
  # Load methylation data - fread is faster than read_csv for large files
  cat("Loading methylation data...\n")
  beta_matrix <- fread(beta_path, header = TRUE, data.table = FALSE)
  
  rownames(beta_matrix) <- beta_matrix[[1]]
  beta_matrix <- beta_matrix[, -1]
  
  # Transpose: rows = samples, columns = CpGs
  cat("Transposing...\n")
  beta_matrix <- t(beta_matrix)
  
  # Subset to train_ids
  beta_matrix <- beta_matrix[train_ids, , drop = FALSE]
  beta_matrix <- as.data.frame(beta_matrix)
  
  cat(sprintf("Loaded %d samples x %d CpGs\n", nrow(beta_matrix), ncol(beta_matrix)))
  
  return(list(beta_matrix = beta_matrix, clinical_data = clinical_data))
}

beta_to_m <- function(beta, beta_threshold = 1e-3) {
  # Convert beta values to M-values
  # M = log2(beta / (1 - beta))
  # 
  # Why M-values?
  # - Beta values are bounded [0,1], heteroscedastic
  # - M-values are unbounded, more normally distributed
  # - Better for linear modeling and differential analysis
  
  was_df <- is.data.frame(beta)
  if (was_df) {
    row_names <- rownames(beta)
    col_names <- colnames(beta)
    beta <- as.matrix(beta)
  }
  
  # Prevent log of 0 or 1 by capping at small threshold
  beta <- pmax(pmin(beta, 1 - beta_threshold), beta_threshold)
  m_values <- log2(beta / (1 - beta))
  
  if (was_df) {
    m_values <- as.data.frame(m_values)
    rownames(m_values) <- row_names
    colnames(m_values) <- col_names
  }
  return(m_values)
}

apply_admin_censoring <- function(df, time_col, event_col, time_cutoff) {
  # Apply administrative censoring at specified time
  # Patients with follow-up > cutoff are censored at cutoff
  # This standardizes follow-up time across cohorts
  
  mask <- df[[time_col]] > time_cutoff
  df[mask, time_col] <- time_cutoff
  df[mask, event_col] <- 0
  cat(sprintf("Applied censoring at %.1f for %s/%s (n=%d).\n", 
              time_cutoff, time_col, event_col, sum(mask)))
  return(df)
}

subset_methylation <- function(mval_matrix, cpg_ids_file) {
  # Subset to predefined CpGs (e.g., from ATAC-seq overlap)
  cpg_ids <- trimws(readLines(cpg_ids_file))
  cpg_ids <- cpg_ids[cpg_ids != ""]
  valid_cpgs <- cpg_ids[cpg_ids %in% colnames(mval_matrix)]
  cat(sprintf("Subsetted to %d CpGs (from %d in file).\n", 
              length(valid_cpgs), length(cpg_ids)))
  return(mval_matrix[, valid_cpgs, drop = FALSE])
}

onehot_encode_clinical <- function(clin, clin_categorical) {
  # One-hot encode categorical variables
  # Drops reference level to avoid multicollinearity
  # 
  # Example: NHG with levels 1,2,3 → NHG2, NHG3 (NHG1 is reference)
  
  for (var in clin_categorical) {
    if (var == "LN") {
      clin[[var]] <- factor(clin[[var]], levels = c("N0", "N+"))
    } else if (var == "NHG") {
      clin[[var]] <- factor(clin[[var]], levels = c("1", "2", "3"))
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
  
  # One-hot encode categorical variables
  encoded_cols <- c()
  for (var in clin_categorical) {
    # model.matrix creates intercept + k-1 dummies
    dummy_df <- as.data.frame(model.matrix(
      as.formula(paste("~", var)),
      data = clin
    ))
    
    # Drop intercept (first column)
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


filter_by_variance <- function(X_train, variance_quantile = 0.75) {
  # Calculate variance for each feature
  variances <- apply(X_train, 2, var, na.rm = TRUE)
  
  # Find threshold and filter
  var_threshold <- quantile(variances, probs = variance_quantile, na.rm = TRUE)
  keep_var <- names(variances[variances >= var_threshold])
  X_filtered <- X_train[, keep_var, drop = FALSE]
  
  cat(sprintf("After variance filter (>= %.0fth percentile): %d features\n", 
              variance_quantile * 100, length(keep_var)))
  
  return(X_filtered)
}


cv_glmnet_alpha_grid <- function(X_train, y_train, alpha_grid, penalty_factor = NULL, 
                                 foldid = NULL, family = "cox", seed = 123) {
  # Cross-validate glmnet across multiple alpha values to find optimal elastic net mixing
  # Returns list of CV results for each alpha
  
  cv_results <- list()
  
  for (alpha_val in alpha_grid) {
    set.seed(seed)
    cv_fit <- cv.glmnet(
      x = as.matrix(X_train),
      y = y_train,
      family = family,
      alpha = alpha_val,
      penalty.factor = penalty_factor,
      foldid = foldid
    )
    
    # Performance at lambda.min
    perf_min <- cv_fit$cvm[cv_fit$lambda == cv_fit$lambda.min]
    
    # Extract selected features
    beta <- coef(cv_fit, s = "lambda.min")
    selected_features <- rownames(beta)[as.vector(beta != 0)]
    
    cv_results[[as.character(alpha_val)]] <- list(
      alpha = alpha_val,
      lambda = cv_fit$lambda.min,
      cvm_min = perf_min,
      cvm_se = cv_fit$cvsd[cv_fit$lambda == cv_fit$lambda.min],
      n_features = length(selected_features),
      features = selected_features,
      fit = cv_fit
    )
    
    cat(sprintf(
      "  Alpha=%.2f: CV-deviance=%.4f (SE=%.4f), Lambda=%.6f, Features=%d\n",
      alpha_val,
      perf_min,
      cv_fit$cvsd[cv_fit$lambda == cv_fit$lambda.min],
      cv_fit$lambda.min,
      length(selected_features)
    ))
  }
  
  return(cv_results)
}



tune_and_fit_coxnet <- function(X_train, y_train, clinical_train, event_col,
                                alpha_grid, penalty_factor = NULL, 
                                n_inner_folds = 5, seed = 123,
                                outcome_name = "outcome") {
  # Tunes CoxNet via inner CV across alpha grid, then fits final model
  # Returns best hyperparameters and fitted model
  
  cat(sprintf("\n--- Inner CV: Tuning CoxNet for %s ---\n", outcome_name))
  
  # Create stratified inner folds
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
  
  # Select best hyperparameters
  best_idx <- which.min(sapply(cv_results, function(x) x$cvm_min))
  best_result <- cv_results[[best_idx]]
  best_alpha <- best_result$alpha
  best_lambda <- best_result$lambda
  
  cat(sprintf("Best hyperparameters: alpha=%.2f, lambda=%.6f\n", 
              best_alpha, best_lambda))
  
  # Fit final model on whole training set
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


scale_continuous_features <- function(X_train, X_test, dont_scale) {
  # Scale continuous features using training set parameters
  # One-hot encoded columns are left unscaled
  
  # Identify continuous columns (not one-hot encoded)
  cont_cols <- setdiff(colnames(X_train), dont_scale)
  
  # Scale continuous variables in training set and save parameters
  train_cont_scaled <- scale(X_train[, cont_cols, drop = FALSE])
  centers <- attr(train_cont_scaled, "scaled:center")
  scales  <- attr(train_cont_scaled, "scaled:scale")
  
  # Apply scaling to training set
  X_train_scaled <- X_train
  X_train_scaled[, cont_cols] <- train_cont_scaled
  
  # Scale test set using TRAINING set's mean and SD (no data leakage!)
  X_test_scaled <- X_test
  X_test_scaled[, cont_cols] <- scale(
    X_test[, cont_cols, drop = FALSE],
    center = centers,
    scale  = scales
  )
  
  cat(sprintf("Scaled %d continuous features\n", length(cont_cols)))
  
  return(list(
    X_train_scaled = X_train_scaled,
    X_test_scaled = X_test_scaled,
    centers = centers,
    scales = scales,
    cont_cols = cont_cols
  ))
}