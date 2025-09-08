#!/usr/bin/env python
# Script: Functions for penalized Cox model
# Author: Lennart Hohmann

# ==============================================================================
# IMPORTS
# ==============================================================================

import numpy as np
import math
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored, brier_score, integrated_brier_score
import matplotlib.pyplot as plt
from sksurv.metrics import as_concordance_index_ipcw_scorer
from sklearn.preprocessing import RobustScaler
from src.utils import variance_filter #, estimate_alpha_grid

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def define_param_grid(grid_alphas, grid_l1ratio=[0.9]):
    """
    Define the final parameter grid for GridSearchCV using estimated alphas.

    Args:
        estimated_alphas (array-like): Array of alpha values.
        param_grid_l1ratio (list): List of l1_ratio values to try.

    Returns:
        dict: Parameter grid for GridSearchCV.
    """
    param_grid = {
        "estimator__coxnetsurvivalanalysis__alphas": [[alpha] for alpha in grid_alphas], # diff string if i dont use a pipe (remove estimator__)
        "estimator__coxnetsurvivalanalysis__l1_ratio": grid_l1ratio
    }
    print(f"\nDefined parameter grid:\n{param_grid}\n", flush=True)
    return param_grid


# ==============================================================================

def run_nested_cv_cox(X, y, param_grid, 
                  outer_cv_folds=5, inner_cv_folds=3):
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
    pipe = make_pipeline(RobustScaler(),CoxnetSurvivalAnalysis())  # do not pass external base_estimator

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
            selected_cpgs = variance_filter(X_train, top_n=50000)
            X_train, X_test = X_train[selected_cpgs], X_test[selected_cpgs]

            # Fit inner CV model
            inner_model.fit(X_train, y_train)
            best_model = inner_model.best_estimator_
            best_params = inner_model.best_params_

            print(f"\t--> Fold {fold_num}: Best params {best_params}", flush=True)

            # --- REFIT with baseline model for survival function prediction ---
            refit_best_model = make_pipeline(
                CoxnetSurvivalAnalysis(
                    alphas=[best_params["estimator__coxnetsurvivalanalysis__alphas"][0]],
                    l1_ratio=best_params["estimator__coxnetsurvivalanalysis__l1_ratio"],
                    fit_baseline_model=True)
            )
            refit_best_model.fit(X_train, y_train)

            outer_models.append({
                "fold": fold_num,
                "model": refit_best_model,
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
def evaluate_outer_models_coxnet(outer_models, X, y, time_grid):
    """
    Evaluate performance of Coxnet models from outer CV folds.

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

    print("\n=== Evaluating Coxnet outer models ===\n", flush=True)
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
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"  Fold {fold} - test set min time: {y_test['RFi_years'].min():.3f}, max time: {y_test['RFi_years'].max():.3f}", flush=True)
        print(f"  Fold {fold} - train set min time: {y_train['RFi_years'].min():.3f}, max time: {y_train['RFi_years'].max():.3f}", flush=True)

        # Subset X_test
        X_test = X_test[selected_features]

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
        cindex = concordance_index_censored(y_test["RFi_event"], y_test["RFi_years"], pred_scores)[0]

        # Compute Brier scores and IBS
        brier_scores = brier_score(y_train, y_test, preds, time_grid)[1]
        ibs = integrated_brier_score(y_train, y_test, preds, time_grid)

        performance.append({
            "fold": fold,
            "auc": auc,
            "mean_auc": mean_auc,
            "auc_at_5y": auc[np.where(time_grid == 5.0)[0][0]] if 5.0 in time_grid else None,
            "brier_t": brier_scores,
            "ibs": ibs,
            "cindex": cindex
        })

        print(f"  Fold {fold} - C-index: {cindex:.3f}, Mean AUC: {mean_auc:.3f}, IBS: {ibs:.3f}", flush=True)

    return performance

# ==============================================================================

def print_selected_cpgs_counts(outer_models):
    """
    Print the number and names of non-zero coefficient CpGs 
    for each outer fold Coxnet model.

    Args:
        outer_models (list): List of dicts with trained models and fold metadata.
                             Each entry must have 'selected_cpgs' and 'model'.
    """
    print("\n=== Selected CpGs per outer fold ===\n", flush=True)

    for entry in outer_models:
        fold = entry["fold"]
        model = entry["model"]
        selected_features = entry.get("selected_cpgs", None)

        if model is None:
            print(f"Fold {fold}: no model", flush=True)
            continue
        if selected_features is None:
            print(f"Fold {fold}: no selected features recorded", flush=True)
            continue

        # Extract Coxnet model from pipeline
        coxnet = model.named_steps["coxnetsurvivalanalysis"]

        # Get non-zero coefficients
        coefs = coxnet.coef_.flatten()
        nonzero_mask = coefs != 0

        # Map to feature names actually used in this fold
        selected_cpgs = np.array(selected_features)[nonzero_mask]

        print(f"Fold {fold}: {len(selected_cpgs)} CpGs selected", flush=True)
        print(f"  CpGs: {selected_cpgs.tolist()}", flush=True)

