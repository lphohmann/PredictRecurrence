#!/usr/bin/env python
# Script: Functions for XGBoost pipeline
# Author: lennart hohmann

# ==============================================================================
# IMPORTS
# ==============================================================================

import numpy as np
from sksurv.metrics import (cumulative_dynamic_auc, concordance_index_censored,
                             brier_score, integrated_brier_score)
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from xgbse import XGBSEDebiasedBCE
from sksurv.metrics import as_concordance_index_ipcw_scorer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
# ==============================================================================
# FUNCTIONS
# ==============================================================================

# ==============================================================================

def define_param_grid(X=None, y=None):

    param_distributions = {
        "xgbregressor__n_estimators": randint(100, 801),        # 100..800
        "xgbregressor__max_depth": randint(3, 8),               # 3..7
        "xgbregressor__learning_rate": uniform(0.01, 0.19),     # 0.01..0.20
        "xgbregressor__min_child_weight": randint(1, 11),       # 1..10
        "xgbregressor__subsample": uniform(0.6, 0.4),           # 0.6..1.0
        "xgbregressor__colsample_bytree": uniform(0.6, 0.4),    # 0.6..1.0
        "xgbregressor__reg_alpha": uniform(0.0, 1.0),           # 0..1
        "xgbregressor__reg_lambda": uniform(1.0, 9.0),          # 1..10
    }

    return param_distributions

# ==============================================================================

from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import numpy as np

def run_nested_cv_xgb(X, y, param_grid, 
                      outer_cv_folds=5, inner_cv_folds=3, top_n_variance=5000, 
                      filter_func=None,
                      dont_filter_vars=None, dont_scale_vars=None,
                      dont_penalize_vars=None):
    """
    Run nested cross-validation for survival models using XGBoost (survival:cox).

    Args:
        X (pd.DataFrame): Feature matrix.
        y (structured array): Survival outcome (event, time).
        param_grid (dict): Hyperparameter grid.
        outer_cv_folds (int): Number of outer CV folds.
        inner_cv_folds (int): Number of inner CV folds.
        top_n_variance (int): Number of top variance features to keep per fold.
        filter_func (callable): Feature selection function.
        dont_filter_vars (list or None): Variables to keep during filtering.
        dont_scale_vars (list or None): Variables to exclude from scaling.
        dont_penalize_vars (list or None): Not used in XGBoost, kept for API compatibility.

    Returns:
        list: Results from each outer fold with trained models and metadata.
    """

    print(f"\n=== Running nested CV with {outer_cv_folds} outer folds "
          f"and {inner_cv_folds} inner folds ===\n", flush=True)

    event_labels = y["RFi_event"]
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=96)
    inner_cv = KFold(n_splits=inner_cv_folds, shuffle=True, random_state=96)

    outer_models = []

    for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X, event_labels)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"\nOuter fold {fold_num}: {sum(y_train['RFi_event'])} events in training set.", flush=True)

        preproc = None 

        try:
            selected_cpgs = filter_func(X_train, y_train, top_n=top_n_variance, keep_vars=dont_filter_vars)
            X_train, X_test = X_train[selected_cpgs], X_test[selected_cpgs]

            if dont_scale_vars is not None:
                print("Not scaling specified variables.", flush=True)
                scale_cols = [c for c in X_train.columns if c not in dont_scale_vars]
                transformers = []
                if len(scale_cols) > 0:
                    transformers.append(("scale", StandardScaler(), scale_cols))
                if len(dont_scale_vars) > 0:
                    transformers.append(("passthrough_encoded", "passthrough", dont_scale_vars))
                preproc = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False,
                                            remainder="drop")
                preproc.fit(X_train)
                feature_names = preproc.get_feature_names_out()
            else:
                preproc = StandardScaler()
                feature_names = X_train.columns.values

            # Encode target: negative time for censored samples
            y_train_xgb = np.where(y_train['RFi_event'] == 1, y_train['RFi_years'], -y_train['RFi_years'])

            pipe = make_pipeline(
                preproc,
                XGBRegressor(objective="survival:cox", eval_metric="cox-nloglik", 
                             tree_method="hist", n_jobs = max(1, (os.cpu_count() or 2) // 2),
                             verbosity=0)
            )
            #if you have more RAM and want full parallelism: set RandomizedSearchCV(n_jobs=-1) and set XGBRegressor(n_jobs=1)
            #scorer_pipe = as_concordance_index_ipcw_scorer(pipe)
            inner_model = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=param_grid,
                cv=inner_cv,
                error_score=0.5,
                n_jobs=1,#-1,
                refit=True
            )

            inner_model.fit(X_train, y_train_xgb)
            best_params = inner_model.best_params_
            print(f"\t--> Fold {fold_num}: Best params {best_params}", flush=True)

            best_xgb_params = {k.replace("estimator__xgbregressor__", ""): v
                               for k, v in best_params.items() if k.startswith("estimator__xgbregressor__")}

            refit_pipe = make_pipeline(
                preproc,
                XGBRegressor(objective="survival:cox", eval_metric="cox-nloglik", 
                             tree_method="hist", n_jobs = max(1, (os.cpu_count() or 2) // 2),
                             verbosity=0, 
                             **best_xgb_params)
            )
            refit_pipe.fit(X_train, y_train_xgb)

            outer_models.append({
                "fold": fold_num,
                "model": refit_pipe,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "cv_results": inner_model.cv_results_,
                "error": None,
                "selected_cpgs": feature_names
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
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
import numpy as np

def evaluate_outer_models_xgb(outer_models, X, y, time_grid):
    """
    Evaluate performance of XGBoost survival models from outer CV folds.

    For each fold, computes:
    - Concordance index (C-index)
    - Time-dependent AUC over the specified time grid
    - Mean AUC
    - AUC at 5 years (if available)
    - Brier score and IBS are set to None (to keep output consistent)

    Args:
        outer_models (list): List of dicts with trained models and fold metadata.
        X (pd.DataFrame): Feature matrix.
        y (structured array): Survival outcome.
        time_grid (np.ndarray): Time points to evaluate metrics.

    Returns:
        list of dicts: One dictionary per fold with performance metrics.
    """
    print("\n=== Evaluating XGBoost outer models ===\n", flush=True)
    performance = []

    for entry in outer_models:
        fold = entry["fold"]
        print(f"Evaluating fold {fold}...", flush=True)

        if entry["model"] is None:
            print(f"  Skipping fold {fold} (no model)", flush=True)
            continue

        model = entry["model"]
        test_idx = entry["test_idx"]
        train_idx = entry["train_idx"]

        # Subset data
        selected_features = entry.get("selected_cpgs", X.columns)
        X_train, X_test = X.iloc[train_idx][selected_features], X.iloc[test_idx][selected_features]
        y_train, y_test = y[train_idx], y[test_idx]

        # Predict risk scores (higher = higher hazard)
        pred_scores = model.predict(X_test)

        # Compute time-dependent AUC
        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, pred_scores, times=time_grid)

        # Compute C-index
        cindex = concordance_index_censored(y_test["RFi_event"], y_test["RFi_years"], pred_scores)[0]

        performance.append({
            "fold": fold,
            "mean_auc": mean_auc,
            "auc": auc,
            "auc_at_5y": auc[np.where(time_grid == 5.0)[0][0]] if 5.0 in time_grid else None,
            "brier_t": None,
            "ibs": None,
            "cindex": cindex
        })

        print(f"  Fold {fold} - C-index: {cindex:.3f}, Mean AUC: {mean_auc:.3f}", flush=True)

    return performance


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

    #mean, se, ci95 = mean_se_ci(ibs_vals)
    #print(f"Integrated Brier Score (IBS): mean={mean:.3f} ± {se:.3f} (SE), 95% CI ±{ci95:.3f}")