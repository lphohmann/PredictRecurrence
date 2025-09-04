#!/usr/bin/env python
# Script: General functions
# Author: Lennart Hohmann

# ==============================================================================
# IMPORTS
# ==============================================================================

import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from src.coxnet_functions import estimate_alpha_grid
from sksurv.metrics import as_concordance_index_ipcw_scorer

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def log(msg):
    print(f"\n=== {msg} ===\n", flush=True)

# ==============================================================================

def load_training_data(train_ids, beta_path, clinical_path):
    """
    Loads and aligns clinical and beta methylation data using provided sample IDs.

    Args:
        train_ids (list): Sample IDs to subset data.
        beta_path (str): File path to methylation beta matrix.
        clinical_path (str): File path to clinical data.

    Returns:
        tuple: (beta_matrix, clinical_data) aligned to the same samples.
    """
    # load clin data
    clinical_data = pd.read_csv(clinical_path)
    clinical_data = clinical_data.set_index("Sample")
    clinical_data = clinical_data.loc[train_ids]

    # load beta values
    beta_matrix = pd.read_csv(beta_path,index_col=0).T

    # align dataframes
    beta_matrix = beta_matrix.loc[train_ids]

    return beta_matrix, clinical_data

# ==============================================================================

def apply_admin_censoring(df, time_col, event_col, time_cutoff=5.0, inplace=False):
    """
    Apply administrative censoring at 'time_cutoff' years for a given time/event pair.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing survival times and events.
    time_col : str
        Column name for survival time (e.g., 'OS_years').
    event_col : str
        Column name for event indicator (e.g., 'OS_event').
    time_cutoff : float, default=5.0
        Time (in years) at which administrative censoring is applied.
    inplace : bool, default=False
        If True, modify df in place. If False, return a new DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with censored times and events.
    """
    if inplace:
        df_to_use = df
    else:
        df_to_use = df.copy()

    mask = df_to_use[time_col] > time_cutoff
    df_to_use.loc[mask, time_col] = time_cutoff
    df_to_use.loc[mask, event_col] = 0

    return df_to_use

# ==============================================================================

def beta2m(beta, beta_threshold=1e-3):
    """
    Convert beta-values to M-values safely.

    Args:
        beta (float, array-like, or pd.DataFrame): Beta-values.
        beta_threshold (float): Lower and upper bound to avoid logit instability. 
                                Clips beta values to [beta_threshold, 1 - beta_threshold].

    Returns:
        Same shape as input: M-values.
    """
    beta = np.clip(beta, beta_threshold, 1 - beta_threshold)
    return np.log2(beta / (1 - beta))

# ==============================================================================

def m2beta(m):
    """
    Convert M-values to beta-values.

    Args:
        m (float, array-like, or pd.DataFrame): M-values.

    Returns:
        Same shape as input: Beta-values.
    """
    return 2**m / (2**m + 1)

# ==============================================================================

def variance_filter(df, min_variance=None, top_n=None):
    """
    Filter features (CpGs) based on variance. Extects CpGs as columns and patients as rows
    Args:
        X (pd.DataFrame): Feature matrix (samples x features).
        top_n (int, optional): Number of top features to select based on variance.
        min_variance (float, optional): Minimum variance threshold to retain features.
    Returns:
        pd.DataFrame: Filtered feature matrix.
    """
    variances = df.var(axis=0)
    
    if min_variance is not None:
        # Filter features with variance above threshold
        selected_features = variances[variances >= min_variance].index
    elif top_n is not None:
        # Select top_n features by variance
        selected_features = variances.sort_values(ascending=False).head(top_n).index
    else:
        raise ValueError("Either min_variance or top_n must be specified")
    
    return df.loc[:, selected_features]

# ==============================================================================

def filter_cpgs_with_cox_lasso(X_train, y_train,
                               initial_variance_top_n=50000,
                               l1_ratio_values=[0.9, 1.0],
                               cox_lasso_cv_folds=5,
                               log_prefix=""):
    """
    Perform CpG filtering on training data using variance + Cox Lasso.

    Steps:
      1. Variance filter
      2. Cox Lasso feature selection on training fold
      3. Return list of CpGs with non-zero coefficients

    Args:
        X_train (pd.DataFrame): Training methylation data (samples x CpGs).
        y_train (structured array): Survival labels (Surv object).
        initial_variance_top_n (int): Keep top N CpGs by variance before Cox Lasso.
        l1_ratio_values (list): L1 ratios to test in Cox Lasso.
        cox_lasso_cv_folds (int): Inner CV folds for Cox Lasso hyperparam tuning.
        log_prefix (str): Prefix for log messages (e.g. "[Fold 1] ").

    Returns:
        list: CpG IDs selected in this training fold.
    """
    log(f"{log_prefix}Starting CpG filtering on training fold ({X_train.shape[1]} CpGs).")

    # 1. Variance filter
    X_var_filtered = variance_filter(X_train, top_n=initial_variance_top_n)
    log(f"{log_prefix}Variance filter applied: {X_var_filtered.shape[1]} CpGs remain.")

    # 2. Cox Lasso
    cox_pipe = make_pipeline(CoxnetSurvivalAnalysis())

    # Estimate alphas
    alphas = estimate_alpha_grid(X_var_filtered, y_train,
                                 l1_ratio=l1_ratio_values[0],
                                 alpha_min_ratio=0.1, n_alphas=10)

    param_grid = {
        'coxnetsurvivalanalysis__alphas': [[a] for a in alphas],
        'coxnetsurvivalanalysis__l1_ratio': l1_ratio_values
    }

    event_indicator = y_train["RFi_event"]  
    stratifier = StratifiedKFold(n_splits=cox_lasso_cv_folds, shuffle=True, random_state=42)
    inner_cv = stratifier.split(X_var_filtered, event_indicator)

    grid_search = GridSearchCV(
        estimator=cox_pipe,
        param_grid=param_grid,
        cv=inner_cv,
        n_jobs=-1,
        error_score=0,
        verbose=0
    )

    grid_search.fit(X_var_filtered, y_train)
    best_model = grid_search.best_estimator_

    # 3. Extract non-zero CpGs
    cox_estimator = best_model.named_steps['coxnetsurvivalanalysis']
    coefs = pd.Series(cox_estimator.coef_.flatten(), index=X_var_filtered.columns)

    selected_cpgs = coefs[coefs != 0].index.tolist()
    log(f"{log_prefix}Cox Lasso selected {len(selected_cpgs)} CpGs.")

    return selected_cpgs


# ==============================================================================
# delete if not needed anymore
def preprocess_data(beta_matrix, top_n_cpgs):
    """
    Converts beta values to M-values and retains most variable CpGs.

    Args:
        beta_matrix (DataFrame): Raw beta methylation matrix.
        top_n_cpgs (int): Number of most variable CpGs to retain.

    Returns:
        DataFrame: Preprocessed M-values matrix.
    """

    print(f"\n=== Preprocessing: Converting to M-values and selecting top {top_n_cpgs} most variable CpGs ===\n", flush=True)

    # Convert beta values to M-values with a threshold 
    mvals = beta2m(beta_matrix, beta_threshold=0.001)
    # Apply variance filtering to retain top N most variable CpGs
    mvals = variance_filter(mvals, top_n=top_n_cpgs)
    return mvals

# ==============================================================================

def run_nested_cv(X, y, base_estimator, param_grid, 
                  outer_cv_folds=5, inner_cv_folds=3, 
                  filter_function=None):
    """
    Run nested cross-validation for survival models (RSF, Coxnet, GBM, etc.).

    Args:
        X (pd.DataFrame): Feature matrix.
        y (structured array): Survival outcome (event, time).
        base_estimator: Survival model (e.g. RandomSurvivalForest()).
        param_grid (dict): Hyperparameter grid.
        outer_cv_folds (int): Number of outer CV folds.
        inner_cv_folds (int): Number of inner CV folds.
        filter_function (callable): Function to filter features.
                                    Must accept (X_train, y_train) and return list of columns.
                                    If None, all features are used.

    Returns:
        list: Results from each outer fold with trained models and metadata.
    """
    
    print(f"\n=== Running nested CV with {outer_cv_folds} outer folds "
          f"and {inner_cv_folds} inner folds ===\n", flush=True)

    event_labels = y["RFi_event"]
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=96)
    inner_cv = KFold(n_splits=inner_cv_folds, shuffle=True, random_state=96)

    # Build pipeline (can extend with scaling/one-hot encoding if needed)
    pipe = make_pipeline(base_estimator)
    print(pipe.get_params().keys())

    # Default scorer = concordance index
    scorer_pipe = as_concordance_index_ipcw_scorer(pipe)

    # Inner CV model selection
    inner_model = GridSearchCV(
        scorer_pipe,
        param_grid=param_grid,
        cv=inner_cv,
        error_score=0.5,
        n_jobs=-1,
        refit=True
    )

    outer_models = []
    for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X, event_labels)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"\nOuter fold {fold_num}: {sum(y_train['RFi_event'])} events "
              f"in training set.", flush=True)

        try:
            # Apply filter function if provided
            if filter_function is not None:
                selected_cpgs = filter_function(X_train, y_train)
                if not selected_cpgs:
                    print(f"Skipping fold {fold_num} (no features selected).", flush=True)
                    outer_models.append({
                        "fold": fold_num,
                        "model": None,
                        "train_idx": train_idx,
                        "test_idx": test_idx,
                        "cv_results": None,
                        "error": "No features selected by filter function",
                        "selected_cpgs": None
                    })
                    continue
                X_train, X_test = X_train[selected_cpgs], X_test[selected_cpgs]
            else:
                selected_cpgs = X_train.columns  # use all features

            # Fit inner CV model
            inner_model.fit(X_train, y_train)
            best_model = inner_model.best_estimator_
            best_params = inner_model.best_params_

            print(f"\t--> Fold {fold_num}: Best params {best_params}", flush=True)

            outer_models.append({
                "fold": fold_num,
                "model": best_model,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "cv_results": inner_model.cv_results_,
                "error": None,
                "selected_cpgs": selected_cpgs
            })

        except Exception as e:
            print(f"Skipping fold {fold_num} due to error: {e}", flush=True)
            outer_models.append({
                "fold": fold_num,
                "model": None,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "cv_results": None,
                "error": str(e),
                "selected_cpgs": None
            })

    return outer_models

# ==============================================================================

def summarize_performance(performance):
    """
    Print summary statistics (mean ± std) for evaluation metrics across folds.
    Ignores NaNs in the calculations.
    """
    print("\n=== RSF Evaluation Summary ===\n", flush=True)
    
    cindexes = [p["cindex"] for p in performance]
    auc5y = [p["auc_at_5y"] for p in performance if p["auc_at_5y"] is not None]
    mean_aucs = [p["mean_auc"] for p in performance]
    ibs_vals = [p["ibs"] for p in performance]

    print(f"C-index: mean={np.nanmean(cindexes):.3f} ± {np.nanstd(cindexes):.3f}")
    if auc5y:
        print(f"AUC@5y: mean={np.nanmean(auc5y):.3f} ± {np.nanstd(auc5y):.3f}")
    print(f"Mean AUC over times: {np.nanmean(mean_aucs):.3f} ± {np.nanstd(mean_aucs):.3f}")
    print(f"Integrated Brier Score (IBS): {np.nanmean(ibs_vals):.3f} ± {np.nanstd(ibs_vals):.3f}")

# ==============================================================================

def select_best_model(performance, outer_models, metric):
    """
    Select the best model by given metric ('ibs', 'mean_auc', or 'auc_at_5y').
    """
    print(f"\n=== Selecting best model by {metric} ===\n", flush=True)
    assert metric in {"ibs", "mean_auc", "auc_at_5y"}
    if metric == "ibs":
        best = min(performance, key=lambda p: p["ibs"])
    else:  # maximize for AUC metrics
        best = max(performance, key=lambda p: p[metric])
    print(f"Best fold: {best['fold']}, {metric}={best[metric]:.3f}")
    best_outer = next((e for e in outer_models if e['fold']==best['fold']), None)
    return best_outer

# ==============================================================================

def summarize_outer_models(outer_models):
    """
    Print a summary of each outer CV fold result.
    """
    print("\n=== Summarizing outer models ===\n", flush=True)
    for entry in outer_models:
        print(f"Fold {entry['fold']}:")
        print(f"  Model: {type(entry['model']).__name__ if entry['model'] else None}")
        print(f"  Training samples: {len(entry.get('train_idx', []))}, "
              f"  Test samples: {len(entry.get('test_idx', []))}")
        print(f"  Selected_cpgs: {entry['selected_cpgs'] if entry['selected_cpgs'] else None}")

# ==============================================================================
