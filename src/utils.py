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
from sksurv.metrics import as_concordance_index_ipcw_scorer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
from sklearn.exceptions import FitFailedWarning
from lifelines import CoxPHFitter
from lifelines.utils import ConvergenceWarning
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sksurv.linear_model import CoxnetSurvivalAnalysis

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def filter_cpgs_univariate_cox(X_train: pd.DataFrame, 
                               y_train, 
                               time_col="RFi_years", 
                               event_col="RFi_event", 
                               top_n=5000,
                               penalizer_value=0.01):
    """
    Filter CpGs using univariate Cox regression.

    Steps:
      1. Run univariate Cox regression for each CpG in X_train.
      2. Rank CpGs by p-value (adjusted for multiple testing).
      3. Return top_n CpG IDs with smallest adjusted p-values.

    Args:
        X_train (pd.DataFrame): CpG matrix (samples x CpGs).
        y_train (structured array or DataFrame): Survival labels (must contain time_col & event_col if DataFrame).
        time_col (str): Column name for survival time in y_train (if DataFrame).
        event_col (str): Column name for event indicator in y_train (if DataFrame).
        top_n (int): Number of top CpGs to keep.
        penalizer_value (float): L2 penalizer for CoxPHFitter.

    Returns:
        list: Selected CpG IDs.
    """
    # Convert y_train to DataFrame if it is structured array
    if isinstance(y_train, np.ndarray) and y_train.dtype.names is not None:
        y_df = pd.DataFrame({
            time_col: y_train[time_col],
            event_col: y_train[event_col]
        }, index=X_train.index)
    else:
        y_df = y_train.copy()

    results = []
    total_cpgs = X_train.shape[1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)

        for i, cpg in enumerate(X_train.columns):
            if (i + 1) % 5000 == 0 or (i + 1) == total_cpgs:
                print(f"  Processing CpG {i + 1}/{total_cpgs}...")

            df = pd.concat([y_df[[time_col, event_col]].copy(), X_train[[cpg]]], axis=1).dropna()
            df.columns = ["time", "event", "cpg_value"]

            if df["cpg_value"].nunique() <= 1:
                results.append({"CpG_ID": cpg, "pval": np.nan})
                continue

            try:
                cph = CoxPHFitter(penalizer=penalizer_value)
                cph.fit(df, duration_col="time", event_col="event")
                summary = cph.summary.loc["cpg_value"]
                results.append({"CpG_ID": cpg, "pval": summary["p"]})
            except Exception as e:
                print(f"Warning: Failed for {cpg}: {e}", flush=True)
                results.append({"CpG_ID": cpg, "pval": np.nan})

    df_results = pd.DataFrame(results)

    # Adjust p-values using Benjamini-Hochberg
    pvals = df_results["pval"].values
    mask = ~pd.isna(pvals)
    padj = np.full_like(pvals, np.nan, dtype=np.float64)
    padj[mask] = multipletests(pvals[mask], method='fdr_bh')[1]
    df_results["padj"] = padj

    # Sort by adjusted p-value and select top_n
    df_results = df_results.sort_values("padj")
    selected_cpgs = df_results["CpG_ID"].iloc[:top_n].tolist()

    print(f"Selected {len(selected_cpgs)} CpGs by univariate Cox filter.")
    return selected_cpgs

# ==============================================================================

from sklearn.compose import ColumnTransformer

def estimate_alpha_grid(X, y, l1_ratio, n_alphas, alpha_min_ratio='auto',
                        top_n_variance=5000, dont_filter_vars=None, dont_scale_vars=None):
    """
    Estimate a suitable grid of alpha values for Coxnet hyperparameter tuning.
    Assumes categorical clinical variables are already one-hot encoded.

    Args:
        X (pd.DataFrame): Feature matrix (including encoded clinical vars).
        y (structured array): Survival labels (from sksurv).
        l1_ratio (float): Elastic net mixing parameter for alpha estimation.
        n_alphas (int): Number of alphas to generate.
        alpha_min_ratio: Passed to CoxnetSurvivalAnalysis.
        top_n_variance (int): Number of top-variance features to keep (variance_filter).
        clinvars_only_encoded (list or None): Encoded clinical variable names in X.

    Returns:
        np.array: Array of alpha values.
    """

    print(f"\n=== Estimating {n_alphas} alpha values for Coxnet tuning ===\n", flush=True)

    warnings.simplefilter("ignore", FitFailedWarning)
    warnings.simplefilter("ignore", UserWarning)

    # Filter CpGs by variance (keep clinvars)
    selected_cpgs = variance_filter(X, top_n=top_n_variance, keep_vars=dont_filter_vars)
    X = X[selected_cpgs]

    # Determine preprocessing
    if dont_scale_vars is not None:
        print("Not scaling scecified variables.", flush=True)

        # Continuous = everything not in encoded clin vars
        scale_cols = [c for c in X.columns if c not in dont_scale_vars]

        transformers = []
        if len(scale_cols) > 0:
            transformers.append(("scale", StandardScaler(), scale_cols))
        if len(dont_scale_vars) > 0:
            transformers.append(("passthrough_encoded", "passthrough", dont_scale_vars))

        preproc = ColumnTransformer(transformers=transformers, remainder="drop")

        pipe = make_pipeline(
            preproc,
            CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio,
                n_alphas=n_alphas,
                alpha_min_ratio=alpha_min_ratio
            )
        )

    else:
        print("Using StandardScaler for all features.", flush=True)
        pipe = make_pipeline(
            StandardScaler(),
            CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio,
                n_alphas=n_alphas,
                alpha_min_ratio=alpha_min_ratio
            )
        )

    # Fit to get alpha grid
    pipe.fit(X, y)
    alphas = pipe.named_steps["coxnetsurvivalanalysis"].alphas_

    print(f"Estimated {len(alphas)} alpha values.", flush=True)
    return alphas

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

def variance_filter(X, y=None, min_variance=None, top_n=None, keep_vars=None):
    """
    Filter features based on variance. Always keeps keep_vars.
    
    Args:
        X (pd.DataFrame): Feature matrix (samples x features).
        y (pd.Series, optional): Unused, kept for API compatibility.
        min_variance (float, optional): Minimum variance threshold to retain features.
        top_n (int, optional): Number of top features to select by variance.
        keep_vars (str or iterable of str, optional): Columns to always keep.
    
    Returns:
        list: Selected column names (keep_vars first, then selected features by variance).
    """

    # Normalize keep_vars to a list
    if keep_vars is None:
        keep_list = []
    elif isinstance(keep_vars, str):
        keep_list = [keep_vars]
    else:
        keep_list = list(keep_vars)

    # Columns to compute variance on = all except keep_vars
    pool_cols = [c for c in X.columns if c not in keep_list]

    if min_variance is None and top_n is None:
        raise ValueError("Either min_variance or top_n must be specified")

    # Compute variance
    variances = X[pool_cols].var(axis=0)

    # Select features
    if min_variance is not None:
        selected = variances[variances >= min_variance].index.tolist()
    else:
        top_n = min(top_n, len(variances))
        selected = variances.sort_values(ascending=False).head(top_n).index.tolist()

    # Combine keep_vars first
    final_selected = keep_list + selected

    print(f"{len(final_selected)} features selected (including {len(keep_list)} keep_vars).")
    return final_selected


'''def variance_filter(X, y=None, min_variance=None, top_n=None):
    """
    Filter features (CpGs) based on variance. Expects CpGs as columns and patients as rows.
    
    Args:
        X (pd.DataFrame): Feature matrix (samples x features).
        top_n (int, optional): Number of top features to select based on variance.
        min_variance (float, optional): Minimum variance threshold to retain features.
    
    Returns:
        list: Selected CpG IDs (column names).
    """

    log(f"Starting CpG filtering on training fold ({X.shape[1]} CpGs).")
    variances = X.var(axis=0)
    
    if min_variance is not None:
        selected_features = variances[variances >= min_variance].index
    elif top_n is not None:
        selected_features = variances.sort_values(ascending=False).head(top_n).index
    else:
        raise ValueError("Either min_variance or top_n must be specified")
    
    log(f"Filter applied: {len(selected_features.tolist())} CpGs remain after filtering.")
    return selected_features.tolist()
'''
# ==============================================================================

def filter_cpgs_with_cox_lasso(X_train, y_train,
                               initial_variance_top_n=5000,#50,#50000,
                               l1_ratio_values=[0.9],
                               cox_lasso_cv_folds=5,
                               log_prefix="",
                               est_alpha_min="auto"):
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
    #X_var_filtered = variance_filter(X_train, top_n=initial_variance_top_n)
    selected_by_variance = variance_filter(X_train, top_n=initial_variance_top_n)
    X_var_filtered = X_train[selected_by_variance]
    log(f"{log_prefix}Variance filter applied: {X_var_filtered.shape[1]} CpGs remain.")

    # 2. Cox Lasso
    cox_pipe = make_pipeline(#RobustScaler(),#RobustScaler(),#StandardScaler(),#RobustScaler(), # ADD BACK
                             CoxnetSurvivalAnalysis())

    # Estimate alphas
    # possible set min alpha to 0.5
    alphas = estimate_alpha_grid(X_var_filtered, y_train,
                                 l1_ratio=l1_ratio_values[0],
                                 n_alphas=10,top_n_variance=initial_variance_top_n,
                                 alpha_min_ratio=est_alpha_min) #alpha_min

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

    # Print optimal parameters
    log(f"{log_prefix}Optimal parameters found in Cox Lasso/Net filter:")
    log(f"{log_prefix}{grid_search.best_params_}")

    # 3. Extract non-zero CpGs
    cox_estimator = best_model.named_steps['coxnetsurvivalanalysis']
    coefs = pd.Series(cox_estimator.coef_.flatten(), index=X_var_filtered.columns)

    selected_cpgs = coefs[coefs != 0].index.tolist()
    log(f"{log_prefix}Cox Lasso selected {len(selected_cpgs)} CpGs.")

    return selected_cpgs

# ==============================================================================

def filter_cpgs_with_correlation(X_train, y_train,
                                 initial_variance_top_n=50000,
                                 correlation_threshold=0.9):
    """
    Perform CpG filtering on training data using variance + correlation.

    Steps:
      1. Variance filter: keep top N most variable CpGs
      2. Correlation filter: remove highly correlated CpGs to reduce redundancy
      3. Return list of selected CpGs

    Args:
        X_train (pd.DataFrame): Training methylation data (samples x CpGs).
        initial_variance_top_n (int): Keep top N CpGs by variance.
        correlation_threshold (float): Threshold above which features are considered redundant.
        log_prefix (str): Prefix for log messages.

    Returns:
        list: CpG IDs selected after filtering.
    """

    log(f"Starting CpG filtering on training fold ({X_train.shape[1]} CpGs).")

    # Compute variance for each CpG (column) across all samples (rows)
    variances = X_train.var(axis=0)

    # Select top N CpGs with highest variance
    top_variance_cpgs = variances.nlargest(initial_variance_top_n).index
    X_var_filtered = X_train[top_variance_cpgs].copy()

    log(f"Variance filter applied: {X_var_filtered.shape[1]} CpGs remain.")

    # Correlation filter
    log("Starting correlation filtering...")

    # Idea: Many CpGs are highly correlated (e.g., nearby CpGs on same gene/region)
    # Keeping all correlated features is redundant and can bias RSF variable importance
    # We will iterate through columns, keeping a CpG only if it is not highly correlated
    # with any CpG already selected.

    # Compute absolute correlation matrix of the filtered CpGs
    corr_matrix = X_var_filtered.corr().abs()

    selected_cpgs = []        # final list of selected CpGs
    already_selected = set()  # tracks CpGs that are too correlated to keep

    # Iterate over each CpG
    for cpg in corr_matrix.columns:
        if cpg not in already_selected:
            # Keep this CpG because it hasn't been excluded yet
            selected_cpgs.append(cpg)

            # Find all CpGs that are highly correlated with this one
            high_corr = corr_matrix[cpg][corr_matrix[cpg] > correlation_threshold].index.tolist()

            # Mark all of these as already selected so we skip them in future iterations
            already_selected.update(high_corr)

    log(f"Correlation filter applied: {len(selected_cpgs)} CpGs remain after filtering.")

    return selected_cpgs

# ==============================================================================

def run_nested_cv(X, y, base_estimator, param_grid, 
                  outer_cv_folds=5, inner_cv_folds=3, 
                  filter_function=None,
                  scaler=None):
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
        scaler (object or None): Optional sklearn scaler (e.g. RobustScaler()).
                                 If provided, will be inserted before the estimator in the pipeline.

    Returns:
        list: Results from each outer fold with trained models and metadata.
    """
    
    print(f"\n=== Running nested CV with {outer_cv_folds} outer folds "
          f"and {inner_cv_folds} inner folds ===\n", flush=True)

    event_labels = y["RFi_event"]
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=96)
    inner_cv = KFold(n_splits=inner_cv_folds, shuffle=True, random_state=96)

    # Build pipeline with optional scaler
    if scaler is not None:
        pipe = make_pipeline(scaler, base_estimator)
        print(f"--> Using pipeline with {scaler.__class__.__name__}", flush=True)
    else:
        pipe = make_pipeline(base_estimator)
        print("--> Using pipeline without scaler", flush=True)

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
        print(f"  Selected_cpgs: {len(entry['selected_cpgs']) if entry['selected_cpgs'] is not None else 0}")

# ==============================================================================
