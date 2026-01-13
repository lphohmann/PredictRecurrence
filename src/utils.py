#!/usr/bin/env python
# Script: General functions
# Author: Lennart Hohmann

# ==============================================================================
# IMPORTS
# ==============================================================================

import pandas as pd
import numpy as np
import os
import math
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored, brier_score, integrated_brier_score
from lifelines import CoxPHFitter
from joblib import Parallel, delayed
import warnings
from tqdm import tqdm

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

    print("Loaded training data.")

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

    print(f"Applied administrative censoring at {time_cutoff} for outcome {time_col} ; {event_col}.")
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

def variance_filter(X, 
                    y=None, 
                    min_variance=None, 
                    top_n=None, 
                    keep_vars=None,
                    exclude_top_perc=None):
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

    # Exclude top percentile of variance if requested
    if exclude_top_perc is not None:
        threshold = np.percentile(variances, 100 - exclude_top_perc)
        variances = variances[variances <= threshold]


    # Select features
    if min_variance is not None:
        selected = variances[variances >= min_variance].index.tolist()
    else:
        top_n = min(top_n, len(variances))
        selected = variances.sort_values(ascending=False).head(top_n).index.tolist()

    # Combine keep_vars first
    final_selected = keep_list + selected

    print(f"\t{len(final_selected)} features selected (including {len(keep_list)} keep_vars).")
    if exclude_top_perc: print(f"\tTop {exclude_top_perc} were discarded.")
    return final_selected

# ==============================================================================

from scipy.stats import t

def summarize_performance(performance):
    """
    Print summary statistics for evaluation metrics across folds.
    Shows mean ± SE (standard error of the mean) and 95% CI of the mean.
    Ignores NaNs in the calculations.
    """
    print("\n=== Evaluation Summary ===\n", flush=True)
    
    def mean_se_ci(vals):
        vals = np.array(vals)
        vals = vals[~np.isnan(vals)]
        n = len(vals)
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)        # sample standard deviation
        se = std / np.sqrt(n)              # standard error of the mean
        ci95 = t.ppf(0.975, df=n-1) * se  # 95% CI using t-distribution
        return mean, se, ci95

    cindexes = [p["cindex"] for p in performance]
    #auc5y = [p["auc_at_5y"] for p in performance if p["auc_at_5y"] is not None]
    mean_aucs = [p["mean_auc"] for p in performance]
    ibs_vals = [p["ibs"] for p in performance]

    mean, se, ci95 = mean_se_ci(cindexes)
    print(f"C-index: mean={mean:.3f} ± {se:.3f} (SE), 95% CI ±{ci95:.3f}")

    #if auc5y:
    #    mean, se, ci95 = mean_se_ci(auc5y)
    #    print(f"AUC@5y: mean={mean:.3f} ± {se:.3f} (SE), 95% #CI ±{ci95:.3f}")

    mean, se, ci95 = mean_se_ci(mean_aucs)
    print(f"Mean AUC over times: mean={mean:.3f} ± {se:.3f} (SE), 95% CI ±{ci95:.3f}")

    mean, se, ci95 = mean_se_ci(ibs_vals)
    print(f"Integrated Brier Score (IBS): mean={mean:.3f} ± {se:.3f} (SE), 95% CI ±{ci95:.3f}")

# ==============================================================================

def select_best_model(performance, outer_models, metric):
    """
    Select the best model by given metric ('ibs', 'mean_auc', or 'auc_at_5y').
    """
    print(f"\n=== Selecting best model by {metric} ===\n", flush=True)
    assert metric in {"ibs", "mean_auc"}
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
        print(f"  Input_training_features: {len(entry['input_training_features']) if entry['input_training_features'] is not None else 0}")

# ==============================================================================

def subset_methylation(mval_matrix,cpg_ids_file):
    with open(cpg_ids_file, 'r') as f:
        cpg_ids = [line.strip() for line in f if line.strip()] # empy line would be skipped
    print(f"Successfully loaded {len(cpg_ids)} CpG IDs for pre-filtering.")

    valid_cpgs = [cpg for cpg in cpg_ids if cpg in mval_matrix.columns]
    missing_cpgs = [cpg for cpg in cpg_ids if cpg not in mval_matrix.columns]
    
    if missing_cpgs:
        print(f"Warning: {len(missing_cpgs)} CpGs from the input file are not in the training data.")

    if len(valid_cpgs) == 0:
        print("Error: No valid pre-filtered CpGs found in the current methylation data columns.")
        raise ValueError("No valid CpGs to proceed with.")

    mval_matrix_filtered = mval_matrix.loc[:, valid_cpgs].copy()
    print(f"Successfully subsetted methylation data to {mval_matrix_filtered.shape[1]} pre-filtered CpGs.")

    return mval_matrix_filtered

# ==============================================================================
'''
def evaluate_outer_models(outer_models, X, y, time_grid):
    """
    Evaluate performance of models from outer CV folds.

    For each fold, computes:
    - AUC(t) over the specified time grid
    - Mean AUC
    - AUC at 5 years
    - Brier score at each time point
    - Integrated Brier Score (IBS)
    - Concordance index (C-index)

    Skips folds where the model is missing or linear predictor values indicate overflow risk.

    Args:
        outer_models (list): List of dicts with trained models and fold metadata.
        X (pd.DataFrame): Feature matrix.
        y (structured array): Survival outcome.
        time_grid (np.ndarray): Time points to evaluate metrics.

    Returns:
        list of dicts: One dictionary per fold with performance metrics.
    """

    print("\n=== Evaluating outer models ===\n", flush=True)
    print(f"Eval time grid: {time_grid}")
    performance = []

    for entry in outer_models:
        fold = entry["fold"]
        
        if entry["model"] is None:
            print(f"  Skipping fold {fold} (no model)", flush=True)
            continue

        model = entry["model"]
        test_idx = entry["test_idx"]
        train_idx = entry["train_idx"]

        model_name = model.named_steps[list(model.named_steps.keys())[-1]].__class__.__name__
        print(f"Evaluating fold {fold} ({model_name})...", flush=True)

        # Subset data
        # Use filter2 if available, otherwise filter1, otherwise all columns
        # Use the most refined feature set available
        if entry.get("features_after_filter2") is not None:
            features_to_use = entry["features_after_filter2"]
        elif entry.get("features_after_filter1") is not None:
            features_to_use = entry["features_after_filter1"]
        else:
            # This should never happen if training function works correctly
            raise ValueError(f"Fold {fold}: No feature list found in entry. "
                     "Check that training function stores 'features_after_filter1'.")

        X_test = X.iloc[test_idx][features_to_use]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"  Fold {fold} - test set min time: {y_test['RFi_years'].min():.3f}, max time: {y_test['RFi_years'].max():.3f}", flush=True)
        print(f"  Fold {fold} - train set min time: {y_train['RFi_years'].min():.3f}, max time: {y_train['RFi_years'].max():.3f}", flush=True)

        # Compute linear predictor and check for overflow
        #coefs = model.named_steps["coxnetsurvivalanalysis"].coef_
        #linear_pred = X_test @ coefs
        #if linear_pred.max().item() > 700 or linear_pred.min().item() < -700:
        #    print(f"  ⚠️ Fold {fold} skipped due to overflow risk.", flush=True)
        #    continue

        # Predict risk scores and survival functions
        pred_scores = model.predict(X_test)
        surv_funcs = model.predict_survival_function(X_test)
        preds = np.row_stack([[fn(t) for t in time_grid] for fn in surv_funcs])
        
        # Compute time-dependent AUC
        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, pred_scores, times=time_grid)
        
        # Compute C-index
        cindex = concordance_index_censored(
            y_test["RFi_event"], 
            y_test["RFi_years"], 
            pred_scores
        )[0]

        # Compute Brier scores and IBS
        brier_scores = brier_score(y_train, y_test, preds, time_grid)[1]
        ibs = integrated_brier_score(y_train, y_test, preds, time_grid)

        performance.append({
            "fold": fold,
            "auc": auc,
            "mean_auc": mean_auc,
            #"auc_at_5y": auc[np.where(time_grid == 5.0)[0][0]] if 5.0 in time_grid else None,
            "brier_t": brier_scores,
            "ibs": ibs,
            "cindex": cindex
        })

        print(f"  Fold {fold} - C-index: {cindex:.3f}, Mean AUC: {mean_auc:.3f}, IBS: {ibs:.3f}", flush=True)

    return performance
'''
#NEW check differences to above
def evaluate_outer_models(outer_models, X, y, time_grid):
    """
    Evaluate performance of models from outer CV folds.
    For each fold, computes metrics at all time points where test set has coverage.
    
    Args:
        outer_models (list): List of dicts with trained models and fold metadata.
        X (pd.DataFrame): Feature matrix.
        y (structured array): Survival outcome.
        time_grid (np.ndarray): Desired time points to evaluate metrics (in years).
    
    Returns:
        list of dicts: One dictionary per fold with performance metrics.
    """
    print("\n=== Evaluating outer models ===\n", flush=True)
    print(f"Desired eval time grid: {time_grid}")
    
    performance = []
    
    for entry in outer_models:
        fold = entry["fold"]
        
        if entry["model"] is None:
            print(f"  Skipping fold {fold} (no model)", flush=True)
            continue
        
        model = entry["model"]
        test_idx = entry["test_idx"]
        train_idx = entry["train_idx"]
        
        model_name = model.named_steps[list(model.named_steps.keys())[-1]].__class__.__name__
        print(f"Evaluating fold {fold} ({model_name})...", flush=True)
        
        # Subset data
        if entry.get("features_after_filter2") is not None:
            features_to_use = entry["features_after_filter2"]
        elif entry.get("features_after_filter1") is not None:
            features_to_use = entry["features_after_filter1"]
        else:
            raise ValueError(f"Fold {fold}: No feature list found in entry.")
        
        X_test = X.iloc[test_idx][features_to_use]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Get max follow-up time in test set
        max_test_time = y_test['RFi_years'].max()
        max_train_time = y_train['RFi_years'].max()
        min_test_time = y_test['RFi_years'].min()
        min_train_time = y_train['RFi_years'].min()

        print(f"  Fold {fold} - test set time range: [{min_test_time:.3f}, {max_test_time:.3f}]", flush=True)
        print(f"  Fold {fold} - train set time range: [{min_train_time:.3f}, {max_train_time:.3f}]", flush=True)
        
        # Determine which time points are feasible for this fold
        # avoid boundary numerical errors
        safe_min_time = min_test_time + 0.001
        safe_max_time = max_test_time - 0.001

        fold_time_grid = time_grid[(time_grid >= safe_min_time) & (time_grid < safe_max_time)]        
        print(f"  Fold {fold} adapted time grid: {fold_time_grid}", flush=True)

        # Log what we're evaluating
        if len(fold_time_grid) < len(time_grid):
            excluded_times = time_grid[time_grid > safe_max_time]
            print(f"  Fold {fold}: Evaluating at {len(fold_time_grid)}/{len(time_grid)} time points (excluding {excluded_times})", flush=True)
        else:
            print(f"  Fold {fold}: Evaluating at all {len(time_grid)} time points", flush=True)
        
        # Predict risk scores and survival functions
        pred_scores = model.predict(X_test)
        surv_funcs = model.predict_survival_function(X_test)
        preds = np.row_stack([[fn(t) for t in fold_time_grid] for fn in surv_funcs])
        
        # Compute time-dependent AUC
        auc, mean_auc = cumulative_dynamic_auc(
            y_train, y_test, pred_scores, times=fold_time_grid
        )

        # Handle NaN in AUC values - just use nanmean if some are NaN
        if np.any(np.isnan(auc)):
            mean_auc = np.nanmean(auc)  # Recalculate using only valid values
            n_valid = (~np.isnan(auc)).sum()
            # Find which time points had NaN
            invalid_times = fold_time_grid[np.isnan(auc)]
                
            print(f"  Fold {fold}: {n_valid}/{len(auc)} time points had valid AUC. "
                  f"Invalid at: {invalid_times} years", flush=True)
            
        # Compute C-index
        cindex = concordance_index_censored(
            y_test["RFi_event"],
            y_test["RFi_years"],
            pred_scores
        )[0]
        
        # Compute Brier scores and IBS
        brier_scores = brier_score(y_train, y_test, preds, fold_time_grid)[1]
        ibs = integrated_brier_score(y_train, y_test, preds, fold_time_grid)
        
        # Create dictionaries mapping time -> metric
        # Use NaN for time points not evaluated in this fold
        auc_by_time = {t: np.nan for t in time_grid}
        brier_by_time = {t: np.nan for t in time_grid}
        
        for i, t in enumerate(fold_time_grid):
            auc_by_time[t] = auc[i]
            brier_by_time[t] = brier_scores[i]
        
        performance.append({
            "fold": fold,
            "auc_by_time": auc_by_time,
            "brier_by_time": brier_by_time,
            "mean_auc": mean_auc,
            "ibs": ibs,
            "cindex": cindex,
            "eval_times": fold_time_grid})
        
        print(f"  Fold {fold} - C-index: {cindex:.3f}, Mean AUC: {mean_auc:.3f}, IBS: {ibs:.3f}", flush=True)
    
    return performance


def aggregate_performance(performance, time_grid):
    """
    Aggregate performance across folds.
    For each time point, computes mean ± SEM across folds that had coverage.
    
    Args:
        performance: List of performance dicts from evaluate_outer_models
        time_grid: The time grid used for evaluation
    
    Returns:
        dict with aggregated metrics including n_folds per time point
    """
    print("\n=== Aggregating Performance Across Folds ===\n")
    
    # C-index (always available)
    cindices = [p['cindex'] for p in performance if not np.isnan(p['cindex'])]
    
    # Mean AUC (computed per fold)
    mean_aucs = [p['mean_auc'] for p in performance if not np.isnan(p['mean_auc'])]
    
    # IBS
    ibs_scores = [p['ibs'] for p in performance if not np.isnan(p['ibs'])]
    
    # Time-specific metrics
    auc_by_time = {}
    brier_by_time = {}
    
    for t in time_grid:
        # Collect AUC values at this time point (excluding NaN)
        auc_values = [p['auc_by_time'][t] for p in performance 
                      if not np.isnan(p['auc_by_time'][t])]
        
        brier_values = [p['brier_by_time'][t] for p in performance 
                        if not np.isnan(p['brier_by_time'][t])]
        
        auc_by_time[t] = {
            'mean': np.mean(auc_values) if auc_values else np.nan,
            'sem': np.std(auc_values) / np.sqrt(len(auc_values)) if len(auc_values) > 1 else np.nan,
            'std': np.std(auc_values) if auc_values else np.nan,  # Keep SD too for reference
            'n_folds': len(auc_values)
        }
        
        brier_by_time[t] = {
            'mean': np.mean(brier_values) if brier_values else np.nan,
            'sem': np.std(brier_values) / np.sqrt(len(brier_values)) if len(brier_values) > 1 else np.nan,
            'std': np.std(brier_values) if brier_values else np.nan,
            'n_folds': len(brier_values)
        }
    
    # store results
    results = {
        'cindex': {
            'mean': np.mean(cindices),
            'sem': np.std(cindices) / np.sqrt(len(cindices)) if len(cindices) > 1 else np.nan,
            'std': np.std(cindices),
            'n_folds': len(cindices)
        },
        'mean_auc': {
            'mean': np.mean(mean_aucs),
            'sem': np.std(mean_aucs) / np.sqrt(len(mean_aucs)) if len(mean_aucs) > 1 else np.nan,
            'std': np.std(mean_aucs),
            'n_folds': len(mean_aucs)
        },
        'ibs': {
            'mean': np.mean(ibs_scores),
            'sem': np.std(ibs_scores) / np.sqrt(len(ibs_scores)) if len(ibs_scores) > 1 else np.nan,
            'std': np.std(ibs_scores),
            'n_folds': len(ibs_scores)
        },
        'auc_by_time': auc_by_time,
        'brier_by_time': brier_by_time
    }
    
    # Print summary with SEM
    print(f"C-index: {results['cindex']['mean']:.3f} ± {results['cindex']['sem']:.3f} (n={results['cindex']['n_folds']} folds)")
    print(f"Mean AUC: {results['mean_auc']['mean']:.3f} ± {results['mean_auc']['sem']:.3f} (n={results['mean_auc']['n_folds']} folds)")
    print(f"IBS: {results['ibs']['mean']:.3f} ± {results['ibs']['sem']:.3f} (n={results['ibs']['n_folds']} folds)")
    
    print("\nAUC by time point:")
    for t in time_grid:
        n = auc_by_time[t]['n_folds']
        if n > 0:
            print(f"  {t:.1f} years: {auc_by_time[t]['mean']:.3f} ± {auc_by_time[t]['sem']:.3f} (n={n} folds)")
        else:
            print(f"  {t:.1f} years: Not available (n=0 folds)")
    
    print("\nBrier Score by time point:")
    for t in time_grid:
        n = brier_by_time[t]['n_folds']
        if n > 0:
            print(f"  {t:.1f} years: {brier_by_time[t]['mean']:.3f} ± {brier_by_time[t]['sem']:.3f} (n={n} folds)")
        else:
            print(f"  {t:.1f} years: Not available (n=0 folds)")
    
    return results
# ==============================================================================   

def _fit_univar_cox(col_series, y_time, y_event):
    """
    Fit a univariate Cox proportional hazards model using lifelines
    and return the p-value for that single variable.
    Returns np.nan if the fit fails or the variable is invalid.
    """

    # --- 1. Build a small dataframe expected by lifelines ---
    df = pd.DataFrame({
        "time": y_time,
        "event": y_event,
        "x": col_series
    }).dropna()

    # --- 2. Skip if no variance ---
    if df["x"].var() == 0.0:
        return np.nan

    try:
        # --- 3. Fit the one-variable Cox model ---
        cph = CoxPHFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence convergence warnings
            cph.fit(df, duration_col="time", event_col="event", show_progress=False)

        # --- 4. Extract p-value directly from the summary table ---
        # cph.summary has one row (index 'x') and a 'p' column with the p-value
        pval = float(cph.summary.loc["x", "p"])

        # --- 5. Return the p-value as a float ---
        return pval

    except Exception:
        # If the fit fails for any reason (singularity, overflow, etc.), return NaN
        return np.nan

# ==============================================================================   

def univariate_cox_filter(X, y, top_n=None, keep_vars=None, n_jobs=8):
    """
    Univariate Cox PH filter (API-compatible with your variance_filter).

    Args:
        X (pd.DataFrame): samples x features.
        y (structured array): survival array with fields 'RFi_time' and 'RFi_event'.
        top_n (int, optional): select this many features with smallest p-values.
        keep_vars (str or iterable, optional): columns to always keep.
        n_jobs (int): number of parallel jobs for feature-wise fitting (joblib).

    Returns:
        list: selected column names (keep_vars first, then selected features).
    """

    # normalize keep_vars to consistent list type
    if keep_vars is None:
        keep_list = []
    elif isinstance(keep_vars, str):
        keep_list = [keep_vars]
    else:
        keep_list = list(keep_vars)

    # check required fields in y
    try:
        y_time = y["RFi_years"]
        y_event = y["RFi_event"]
    except Exception as e:
        raise ValueError("univariate_cox_filter expects y to have 'RFi_time' and 'RFi_event' fields.") from e

    # features to test
    pool_cols = [c for c in X.columns if c not in keep_list]
    if len(pool_cols) == 0:
        print("\tNo features to test (all columns are in keep_vars).")
        return keep_list

    # Compute univariate Cox p-values in parallel.
    # - 'n_jobs' controls number of threads; -1 uses all cores. 
    # - 'prefer="threads"' avoids pickling large DataFrames.
    #results = Parallel(n_jobs=n_jobs, prefer="threads")(
    #    delayed(_fit_univar_cox)(
    #        X[col], y_time, y_event
    #    ) for col in pool_cols
    #)

    # Convert to list so tqdm can measure length
    pool_cols_list = list(pool_cols)
    # wrap the generator in tqdm for a progress bar
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_univar_cox)(X[col], y_time, y_event)
        for col in tqdm(pool_cols_list, desc="\tFitting univariate Cox models", mininterval=20.0)
    )

    pvals = pd.Series(results, index=pool_cols, name="pvalue")

    # drop NaNs (failed fits / zero variance)
    pvals = pvals.dropna()

    if pvals.empty:
        print("\tAll univariate Cox fits failed or had insufficient data; returning keep_vars only.")
        return keep_list

    # Selection logic:  top_n
    top_n = min(int(top_n), len(pvals)) # cant select more features than in pvals
    selected = pvals.sort_values().head(top_n).index.tolist()
    
    # final list: keep_vars first (preserve order), then selected (avoid duplicates)
    final_selected = keep_list + [c for c in selected if c not in keep_list]

    # Print info about selected features
    print(f"\t{len(final_selected)} features selected by univariate Cox "
        f"\t({len(keep_list)} kept, {len(selected)} from tests).")
    
    selected_pvals = pvals.loc[selected]
    print(f"\tSelected p-values: min={selected_pvals.min():.4g}, "
        f"\tmax={selected_pvals.max():.4g}, median={selected_pvals.median():.4g}")

    return final_selected
