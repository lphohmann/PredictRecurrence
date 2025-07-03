library(randomForestSRC)
library(pec)
library(riskRegression)
library(survival)

define_param_grid <- function() {
  expand.grid(
    ntree = c(100, 300),
    mtry = c(0.2, 0.5) * 10000,  # change if not using 10k CpGs
    nodesize = c(5, 10),
    maxnodes = c(5, 10, NA)
  )
}

run_nested_cv <- function(X, y, param_grid, outer_folds = 5, inner_folds = 3) {
  outer_folds_idx <- caret::createFolds(y$status, k = outer_folds)
  results <- list()

  for (i in seq_along(outer_folds_idx)) {
    log_msg(paste("Outer fold", i))
    test_idx <- outer_folds_idx[[i]]
    train_idx <- setdiff(seq_len(nrow(X)), test_idx)
    
    X_train <- X[train_idx, ]
    y_train <- y[train_idx, ]
    best_model <- NULL
    best_score <- -Inf

    for (j in 1:nrow(param_grid)) {
      params <- param_grid[j, ]
      rf <- rfsrc(Surv(time, status) ~ ., data = cbind(y_train, X_train),
                  ntree = params$ntree,
                  mtry = round(params$mtry),
                  nodesize = params$nodesize,
                  maxnodes = if (!is.na(params$maxnodes)) params$maxnodes else NULL,
                  importance = TRUE)
      cidx <- rf$err.rate[length(rf$err.rate)]
      if (-cidx > best_score) {
        best_score <- -cidx
        best_model <- rf
        best_params <- params
      }
    }
    
    results[[i]] <- list(
      model = best_model,
      params = best_params,
      train_idx = train_idx,
      test_idx = test_idx
    )
  }
  
  results
}

run_reg_nested_cv <- function(X, y, param_grid, outer_folds = 5, inner_folds = 3) {
  outer_folds_idx <- caret::createFolds(y$status, k = outer_folds)
  results <- list()

  for (i in seq_along(outer_folds_idx)) {
    log_msg(paste("Outer fold", i))
    test_idx <- outer_folds_idx[[i]]
    train_idx <- setdiff(seq_len(nrow(X)), test_idx)

    X_train <- X[train_idx, , drop = FALSE]
    y_train <- y[train_idx, , drop = FALSE]

    # ----- Cox-based predictor weights -----
    pvals <- apply(X_train, 2, function(x) {
      fit <- survival::coxph(survival::Surv(time, status) ~ x, data = data.frame(x = x, y_train))
      summary(fit)$coefficients[,"Pr(>|z|)"]
    })

    # Replace NA p-values with 1 (non-informative)
    pvals[is.na(pvals)] <- 1

    # Weight formula from Ishwaran: w_j = min(lambda, 1/p_j)
    lambda_p <- length(pvals)^(4/5)
    weights <- pmin(lambda_p, 1 / pvals)
    predictorWt <- weights / sum(weights)

    # Inner CV for hyperparameter selection
    best_model <- NULL
    best_score <- -Inf

    for (j in 1:nrow(param_grid)) {
      params <- param_grid[j, ]
      rf <- randomForestSRC::rfsrc(
        survival::Surv(time, status) ~ ., 
        data = cbind(y_train, X_train),
        ntree = params$ntree,
        mtry = round(params$mtry),
        nodesize = params$nodesize,
        maxnodes = if (!is.na(params$maxnodes)) params$maxnodes else NULL,
        importance = TRUE,
        predictorWt = predictorWt
      )
      
      # Use C-index (negated for maximization)
      cidx <- rf$err.rate[length(rf$err.rate)]
      if (-cidx > best_score) {
        best_score <- -cidx
        best_model <- rf
        best_params <- params
      }
    }

    results[[i]] <- list(
      model = best_model,
      params = best_params,
      train_idx = train_idx,
      test_idx = test_idx
    )
  }

  results
}


evaluate_models <- function(results, X, y, time_grid) {
  perf_list <- list()
  
  for (i in seq_along(results)) {
    model <- results[[i]]$model
    test_idx <- results[[i]]$test_idx
    train_idx <- results[[i]]$train_idx
    
    X_test <- X[test_idx, ]
    y_test <- y[test_idx, ]
    X_train <- X[train_idx, ]
    y_train <- y[train_idx, ]
    
    # AUC and Brier Score
    score <- Score(list(RSF = model),
                   formula = Surv(time, status) ~ 1,
                   data = cbind(y_test, X_test),
                   times = time_grid,
                   metrics = c("auc", "brier"),
                   summary = FALSE,
                   conf.int = FALSE,
                   .save = "matrix")
    
    perf_list[[i]] <- list(
      fold = i,
      auc = score$AUC$score,
      brier = score$Brier$score,
      ibs = score$Brier$IBS
    )
  }
  
  perf_list
}

select_best_model <- function(performance, metric = "auc") {
  scores <- sapply(performance, function(p) mean(p[[metric]], na.rm = TRUE))
  which.max(scores)
}


library(pec)
library(riskRegression)
library(survival)
library(ggplot2)

plot_auc_curves <- function(model, data, times, formula, plot_file = NULL) {
  # data: original data used in the model (must contain time & status columns)
  # formula: e.g., Surv(time, status) ~ .
  
  # Compute time-dependent AUC
  auc_obj <- Score(
    object = list(RSF = model),
    formula = formula,
    data = data,
    times = times,
    summary = "none",
    metrics = "AUC",
    conf.int = FALSE,
    conservative = TRUE,
    cens.model = "cox"
  )

  auc_df <- as.data.frame(auc_obj$AUC$score)
  
  # Plot
  p <- ggplot(auc_df, aes(x = times, y = AUC, group = model, color = model)) +
    geom_line(size = 1.2) +
    labs(x = "Time", y = "AUC", title = "Time-dependent AUC") +
    theme_minimal()
  
  if (!is.null(plot_file)) {
    ggsave(plot_file, plot = p)
  } else {
    print(p)
  }
}


plot_brier_scores <- function(model, data, times, formula, plot_file = NULL) {
  # data: should contain the full survival dataset
  # formula: Surv(time, status) ~ .
  
  brier_obj <- Score(
    object = list(RSF = model),
    formula = formula,
    data = data,
    times = times,
    metrics = "Brier",
    summary = "none",
    conf.int = FALSE,
    cens.model = "cox"
  )
  
  brier_df <- as.data.frame(brier_obj$Brier$score)
  
  # Plot
  p <- ggplot(brier_df, aes(x = times, y = Brier, group = model, color = model)) +
    geom_line(size = 1.2) +
    labs(x = "Time", y = "Brier Score", title = "Time-dependent Brier Score") +
    theme_minimal()
  
  if (!is.null(plot_file)) {
    ggsave(plot_file, plot = p)
  } else {
    print(p)
  }
}


vimp_rsf <- function(model, top_n = 20, plot = TRUE) {
  # model: output of rfsrc()
  importance_df <- data.frame(
    variable = names(model$importance),
    importance = model$importance
  ) %>%
    arrange(desc(importance))
  
  if (plot) {
    ggplot(importance_df[1:top_n, ], aes(x = reorder(variable, importance), y = importance)) +
      geom_bar(stat = "identity", fill = "steelblue") +
      coord_flip() +
      labs(x = "Variable", y = "Importance", title = "Top Variable Importances") +
      theme_minimal()
  }
  
  return(importance_df)
}
