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

import os
import math
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import datetimes_to_durations
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

def cox_filter(X, y, time_col='RFi_years', event_col='RFi_event',
               top_n=None, keep_vars=None, prefilter=None,
               output_dir=None, save_csv=True, save_plots=True,
               plot_prefix='coxfilter', show_plots=False):
    """
    Filter features based on univariate Cox regression and BH-adjusted p-values.
    Always keeps keep_vars.
    Supports y as a DataFrame or a structured array (e.g. from sksurv).

    Additional args:
      output_dir: folder to save results (CSV + PNGs). If None, no files are written.
      save_csv: whether to write a CSV summary (requires output_dir)
      save_plots: whether to write PNG plots (requires output_dir)
      plot_prefix: prefix for plot filenames
      show_plots: whether to call plt.show() after creating each plot (useful in notebooks)
    Returns:
      final_selected: list of kept + top features (same as before)
      summary_df: DataFrame with per-feature stats (p, bh_p, coef, HR, HR_CI_lower, HR_CI_upper)
    """

    # --- optional prefilter using a user function variance_filter (unchanged) ---
    if prefilter is not None:
        selected_cpgs = variance_filter(X, top_n=prefilter, keep_vars=keep_vars)
        X = X[selected_cpgs]

    # --- Convert structured array y (from sksurv) to DataFrame if needed ---
    if not isinstance(y, pd.DataFrame):
        try:
            y = pd.DataFrame({time_col: y[time_col], event_col: y[event_col]})
        except Exception:
            raise ValueError("y must be a DataFrame or a structured array with fields "
                             f"'{time_col}' and '{event_col}'")

    # --- Ensure keep_vars list ---
    if keep_vars is None:
        keep_list = []
    elif isinstance(keep_vars, str):
        keep_list = [keep_vars]
    else:
        keep_list = list(keep_vars)

    pool_cols = [c for c in X.columns if c not in keep_list]

    if top_n is None:
        top_n = len(pool_cols)

    if top_n == 0:
        print(f"Top_n=0, returning only keep_vars ({len(keep_list)} features).")
        summary_df = pd.DataFrame(columns=['feature', 'p', 'bh_p', 'coef', 'hr', 'hr_ci_lower', 'hr_ci_upper'])
        return keep_list, summary_df

    # --- Normalize y columns to numeric/boolean lifelines-friendly types ---
    y_clean = y[[time_col, event_col]].copy()

    # If time is datetime-like, convert to durations (float) and origin (we discard origin)
    if pd.api.types.is_datetime64_any_dtype(y_clean[time_col]) or \
       np.issubdtype(y_clean[time_col].dtype, np.datetime64) :
        durations, origins = datetimes_to_durations(y_clean[time_col], y_clean[event_col])
        y_clean[time_col] = durations
        # datetimes_to_durations returns boolean-ish events; ensure boolean
        y_clean[event_col] = y_clean[event_col].astype(bool)
    else:
        # coerce time to numeric (float). invalid -> NaN (will be dropped per-feature)
        y_clean[time_col] = pd.to_numeric(y_clean[time_col], errors='coerce')
        # coerce event to bool (or int 0/1)
        if y_clean[event_col].dtype == object:
            lower = y_clean[event_col].astype(str).str.lower()
            mapped = lower.map({'y': True, 'yes': True, 'true': True, '1': True,
                                'n': False, 'no': False, 'false': False, '0': False})
            if mapped.isna().any():
                y_clean[event_col] = pd.to_numeric(y_clean[event_col], errors='coerce').fillna(0).astype(bool)
            else:
                y_clean[event_col] = mapped.astype(bool)
        else:
            y_clean[event_col] = y_clean[event_col].astype(bool)

    # Storage for results
    results = []
    pvals = {}

    for feature in pool_cols:
        # Build per-feature df, coerce feature to numeric
        df = pd.concat([y_clean, X[[feature]]], axis=1).copy()
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
        df = df.dropna()
        if df.shape[0] < 3:
            # Not enough data to fit a model reliably; assign p=1.0 and NaNs for coef
            pvals[feature] = 1.0
            results.append({
                'feature': feature, 'p': 1.0, 'coef': np.nan, 'se_coef': np.nan,
                'hr': np.nan, 'hr_ci_lower': np.nan, 'hr_ci_upper': np.nan,
                'n_obs': df.shape[0]
            })
            continue

        cph = CoxPHFitter()
        try:
            cph.fit(df, duration_col=time_col, event_col=event_col, show_progress=False)
            # lifelines summary: 'coef' and 'se(coef)' usually present
            summary = cph.summary
            # attempt to read coef and se
            if feature in summary.index:
                coef = float(summary.loc[feature, 'coef'])
                # try various possible names for se column
                se = None
                for col_name in ['se(coef)', 'se(coef)']:  # repetitive but clear
                    if col_name in summary.columns:
                        se = float(summary.loc[feature, col_name])
                        break
                # sometimes lifelines names are 'se(coef)'
                if se is None:
                    # fallback: try 'var' or compute from confidence intervals if available
                    for alt in ['z', 'p']:
                        if alt in summary.columns:
                            # can't reliably get se from z without coef; skip se
                            se = np.nan
                pval = float(summary.loc[feature, 'p']) if 'p' in summary.columns else 1.0
            else:
                # If the index name does not match the feature, attempt to extract first row
                first_row = summary.iloc[0]
                coef = float(first_row.get('coef', np.nan))
                se = float(first_row.get('se(coef)', np.nan)) if 'se(coef)' in first_row.index else np.nan
                pval = float(first_row.get('p', 1.0))
        except Exception:
            coef = np.nan
            se = np.nan
            pval = 1.0

        # compute hazard ratio and CI from coef & se if available
        if (not np.isnan(coef)) and (not np.isnan(se)):
            hr = math.exp(coef)
            z = 1.96  # approximate 95% CI
            lower = math.exp(coef - z * se)
            upper = math.exp(coef + z * se)
        elif not np.isnan(coef):
            hr = math.exp(coef)
            lower = np.nan
            upper = np.nan
        else:
            hr = np.nan
            lower = np.nan
            upper = np.nan

        pvals[feature] = pval
        results.append({
            'feature': feature,
            'p': pval,
            'coef': coef,
            'se_coef': se,
            'hr': hr,
            'hr_ci_lower': lower,
            'hr_ci_upper': upper,
            'n_obs': df.shape[0]
        })

    # multiple testing correction
    features_list = [r['feature'] for r in results]
    raw_pvals = [r['p'] for r in results]
    # ensure length > 0
    if len(raw_pvals) == 0:
        raise RuntimeError("No features were tested.")

    _, bh_adj, _, _ = multipletests(raw_pvals, method='fdr_bh')

    # assemble summary dataframe
    summary_df = pd.DataFrame(results)
    summary_df['bh_p'] = bh_adj
    # order by bh_p ascending
    summary_df.sort_values('bh_p', inplace=True)

    # select top features
    top_features = summary_df.head(top_n)['feature'].tolist()
    final_selected = keep_list + top_features
    print(f"{len(final_selected)} features selected (including {len(keep_list)} keep_vars).")

    # write outputs if requested
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        if save_csv:
            csv_path = os.path.join(output_dir, f"{plot_prefix}_cox_summary.csv")
            summary_df.to_csv(csv_path, index=False)
            print(f"Saved CSV summary to: {csv_path}")

        if save_plots:
            # 1) histogram of raw p-values
            try:
                plt.figure(figsize=(6,4))
                plt.hist(summary_df['p'].dropna(), bins=50)
                plt.xlabel('Raw p-value')
                plt.ylabel('Count')
                plt.title('Histogram of raw p-values')
                p1 = os.path.join(output_dir, f"{plot_prefix}_pval_hist.png")
                plt.tight_layout()
                plt.savefig(p1, dpi=150)
                if show_plots:
                    plt.show()
                plt.close()
                print(f"Saved: {p1}")
            except Exception as e:
                print("Failed to create raw p-value histogram:", e)

            # 2) histogram of BH-adjusted p-values
            try:
                plt.figure(figsize=(6,4))
                plt.hist(summary_df['bh_p'].dropna(), bins=50)
                plt.xlabel('BH-adjusted p-value')
                plt.ylabel('Count')
                plt.title('Histogram of BH-adjusted p-values')
                p2 = os.path.join(output_dir, f"{plot_prefix}_bh_pval_hist.png")
                plt.tight_layout()
                plt.savefig(p2, dpi=150)
                if show_plots:
                    plt.show()
                plt.close()
                print(f"Saved: {p2}")
            except Exception as e:
                print("Failed to create BH p-value histogram:", e)

            # 3) histogram of hazard ratios (HR)
            try:
                plt.figure(figsize=(6,4))
                plt.hist(summary_df['hr'].dropna(), bins=50)
                plt.xlabel('Hazard ratio (HR)')
                plt.ylabel('Count')
                plt.title('Histogram of hazard ratios')
                p3 = os.path.join(output_dir, f"{plot_prefix}_hr_hist.png")
                plt.tight_layout()
                plt.savefig(p3, dpi=150)
                if show_plots:
                    plt.show()
                plt.close()
                print(f"Saved: {p3}")
            except Exception as e:
                print("Failed to create HR histogram:", e)

            # 4) volcano-style plot: log(HR) vs -log10(p)
            try:
                plot_df = summary_df.dropna(subset=['hr', 'p']).copy()
                plot_df['log_hr'] = np.log(plot_df['hr'].replace(0, np.nan))
                plot_df['minus_log10_p'] = -np.log10(plot_df['p'].replace(0, np.nextafter(0, 1)))
                plt.figure(figsize=(6,5))
                plt.scatter(plot_df['log_hr'], plot_df['minus_log10_p'], s=10)
                plt.axvline(0, color='k', linewidth=0.5)
                plt.xlabel('log(HR)')
                plt.ylabel('-log10(p)')
                plt.title('Volcano-style: log(HR) vs -log10(p)')
                p4 = os.path.join(output_dir, f"{plot_prefix}_volcano.png")
                plt.tight_layout()
                plt.savefig(p4, dpi=150)
                if show_plots:
                    plt.show()
                plt.close()
                print(f"Saved: {p4}")
            except Exception as e:
                print("Failed to create volcano plot:", e)

    return final_selected, summary_df

# ==============================================================================

from sklearn.compose import ColumnTransformer

def estimate_alpha_grid(X, y, l1_ratio, n_alphas, alpha_min_ratio='auto',
                        top_n_variance=5000, 
                        filter_func=None,
                        dont_filter_vars=None, dont_scale_vars=None,
                        dont_penalize_vars=None):
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
    
    # TEST THIS; ADD ARGUMENT HERE AND ALSO IN THE ESTIMATE ALPHA
    selected_cpgs = filter_func(X,y, top_n=top_n_variance, keep_vars=dont_filter_vars)
    
    #selected_cpgs = variance_filter(X, top_n=top_n_variance, keep_vars=dont_filter_vars)
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

        preproc = ColumnTransformer(transformers=transformers, 
                                    verbose_feature_names_out=False,
                                    remainder="drop")

        # Get feature order after transformation
        preproc.fit(X)  
        feature_names = preproc.get_feature_names_out()

    
    else:
        # All features scaled
        preproc = StandardScaler()
        #preproc.fit(X)  
        feature_names = X.columns.values  # order is preserved

    # ---------------------------
    # Build penalty factor vector
    # ---------------------------

    penalty_factor = np.ones(len(feature_names), dtype=float)
    if dont_penalize_vars is not None:
        for i, fname in enumerate(feature_names):
            # if feature name matches one to not penalize
            if fname in dont_penalize_vars:
                penalty_factor[i] = 0.0

    # ---------------------------
    # Construct inner CV pipeline
    # ---------------------------
    pipe = make_pipeline(preproc,
                         CoxnetSurvivalAnalysis(l1_ratio=l1_ratio,
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

import numpy as np
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
    auc5y = [p["auc_at_5y"] for p in performance if p["auc_at_5y"] is not None]
    mean_aucs = [p["mean_auc"] for p in performance]
    ibs_vals = [p["ibs"] for p in performance]

    mean, se, ci95 = mean_se_ci(cindexes)
    print(f"C-index: mean={mean:.3f} ± {se:.3f} (SE), 95% CI ±{ci95:.3f}")

    if auc5y:
        mean, se, ci95 = mean_se_ci(auc5y)
        print(f"AUC@5y: mean={mean:.3f} ± {se:.3f} (SE), 95% CI ±{ci95:.3f}")

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

def subset_methylation(mval_matrix,cpg_ids_file):
    with open(cpg_ids_file, 'r') as f:
        cpg_ids = [line.strip() for line in f if line.strip()] # empy line would be skipped
    print(f"Successfully loaded {len(cpg_ids)} CpG IDs for pre-filtering.")

    valid_cpgs = [cpg for cpg in cpg_ids if cpg in mval_matrix.columns]
    missing_cpgs = [cpg for cpg in cpg_ids if cpg not in mval_matrix.columns]
    
    if missing_cpgs:
        print(f"Warning: {len(missing_cpgs)} CpGs from the input file are not in the training data: {missing_cpgs}")

    if len(valid_cpgs) == 0:
        print("Error: No valid pre-filtered CpGs found in the current methylation data columns.")
        raise ValueError("No valid CpGs to proceed with.")

    mval_matrix_filtered = mval_matrix.loc[:, valid_cpgs].copy()
    print(f"Successfully subsetted methylation data to {mval_matrix_filtered.shape[1]} pre-filtered CpGs.")

    return mval_matrix_filtered

# ==============================================================================   