#!/usr/bin/env python
# Script: Functions for XGBoost pipeline
# Author: lennart hohmann

# ==============================================================================
# IMPORTS
# ==============================================================================

import numpy as np
from sksurv.metrics import (cumulative_dynamic_auc, concordance_index_censored,
                             brier_score, integrated_brier_score)

import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold

from sksurv.metrics import as_concordance_index_ipcw_scorer


# ==============================================================================
# FUNCTIONS
# ==============================================================================

def define_param_grid(X=None, y=None):
    param_grid = {
        "estimator__xgb_model__max_depth": [3, 5, 7],
        "estimator__xgb_model__learning_rate": [0.01, 0.1],
        "estimator__xgb_model__n_estimators": [200, 500],
        "estimator__xgb_model__subsample": [0.8, 1.0],
        "estimator__xgb_model__colsample_bynode": [0.5, 1.0]
    }
    print(f"Defined XGBSE parameter grid: {param_grid}")
    return param_grid

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
import pandas as pd
from sksurv.metrics import cumulative_dynamic_auc, brier_score, integrated_brier_score, concordance_index_censored

def _extract_event_time_from_structured(y_struct):
    """
    Given a numpy structured array y (first field = event indicator, second = time),
    returns (events, times) as numpy arrays.
    Works even if field names are nonstandard (e.g. 'RFi_event'/'RFi_years' or 'c1'/'c2').
    """
    names = y_struct.dtype.names
    if names is None or len(names) < 2:
        raise ValueError("y must be a structured numpy array with at least two fields (event, time).")
    events = y_struct[names[0]].astype(bool)
    times = y_struct[names[1]].astype(float)
    return events, times

# ==============================================================================

def evaluate_outer_models(outer_models, X, y, time_grid):
    """
    Evaluate each trained model in outer_models on its test fold.

    Supports:
      - scikit-survival style estimators that expose `predict_survival_function(X)` and `predict(X)` (risk).
      - xgbse estimators whose `predict(X)` returns a pandas.DataFrame with columns == time windows.

    Returns: list of dicts with keys:
      "fold", "auc", "mean_auc", "auc_at_5y", "brier_t", "ibs", "cindex"
    """
    print("\n=== Evaluating outer models ===\n", flush=True)
    performance = []

    # robustly get global min/max time for informative printing
    try:
        _, global_times = _extract_event_time_from_structured(y)
        print(f"Global evaluation time grid: {time_grid}", flush=True)
        print(f"Global min time: {global_times.min()}, max time: {global_times.max()}\n", flush=True)
    except Exception:
        print("Warning: couldn't read min/max from y (unexpected format).", flush=True)

    for entry in outer_models:
        fold = entry["fold"]
        print(f"Evaluating fold {fold}...", flush=True)

        if entry.get("model") is None:
            print(f"  Skipping fold {fold} (no model).", flush=True)
            continue

        model = entry["model"]
        test_idx = entry["test_idx"]
        train_idx = entry["train_idx"]

        # Decide features for this fold
        selected_features = entry.get("selected_cpgs", X.columns)
        X_test = X.iloc[test_idx][selected_features]
        X_train = X.iloc[train_idx][selected_features]
        y_test = y[test_idx]
        y_train = y[train_idx]

        # sanity prints (use structured extraction so names don't matter)
        try:
            _, t_test = _extract_event_time_from_structured(y_test)
            _, t_train = _extract_event_time_from_structured(y_train)
            print(f"  Fold {fold} - test set min time: {t_test.min():.3f}, max time: {t_test.max():.3f}", flush=True)
            print(f"  Fold {fold} - train set min time: {t_train.min():.3f}, max time: {t_train.max():.3f}", flush=True)
        except Exception:
            print("  Warning: unable to print train/test min/max times (y format unexpected).", flush=True)

        # ---------------------------
        # Get survival curves (preds) and a single risk score (pred_scores)
        # ---------------------------
        try:
            # Case A: scikit-survival style estimator (RSF etc.)
            if hasattr(model, "predict_survival_function"):
                # risk scores (RSF.predict gives risk ranking used previously)
                pred_scores = model.predict(X_test)

                # survival functions: a list (or iterable) of callables/step-fns
                surv_funcs = model.predict_survival_function(X_test)
                # evaluate at time_grid -> matrix (n_samples, n_times)
                preds = np.vstack([[fn(t) for t in time_grid] for fn in surv_funcs])

            else:
                # Case B: xgbse-like estimator (predict returns DataFrame with time columns)
                raw_pred = model.predict(X_test)

                # If model.predict returned a pandas DataFrame -> columns are times
                if isinstance(raw_pred, pd.DataFrame):
                    surv_df = raw_pred.copy()
                    # convert column labels to numeric times
                    try:
                        model_times = pd.to_numeric(surv_df.columns).astype(float)
                    except Exception:
                        # fallback: use index positions (rare)
                        model_times = np.arange(surv_df.shape[1], dtype=float)

                    # ensure times are sorted and reorder columns accordingly
                    order = np.argsort(model_times)
                    model_times = model_times[order]
                    surv_vals = surv_df.values[:, order]

                elif isinstance(raw_pred, np.ndarray):
                    # If predict returned ndarray (n_samples, n_timepoints), try to find model times
                    surv_vals = raw_pred
                    model_times = None
                    # prefer attributes commonly used by xgbse-like models
                    for attr in ("time_bins_", "time_bins", "event_times_", "event_times"):
                        if hasattr(model, attr):
                            model_times = np.asarray(getattr(model, attr))
                            break
                    if model_times is None:
                        # as a last resort, assume the array columns correspond to requested time_grid
                        model_times = np.asarray(time_grid)
                        if surv_vals.shape[1] != len(model_times):
                            # try to broadcast (or raise explicit error)
                            raise ValueError("Model returned ndarray but we could not infer its time bins. "
                                             "Either return a DataFrame with time columns or set model.time_bins_.")
                else:
                    # Try to coerce into DataFrame (e.g. list of lists)
                    try:
                        surv_df = pd.DataFrame(raw_pred)
                        model_times = pd.to_numeric(surv_df.columns).astype(float)
                        order = np.argsort(model_times)
                        model_times = model_times[order]
                        surv_vals = surv_df.values[:, order]
                    except Exception as exc:
                        raise ValueError(f"Couldn't interpret model.predict() output (type={type(raw_pred)}).") from exc

                # Interpolate model survival values to requested time_grid
                # Do row-wise interpolation (handle different left/right values per row)
                preds = np.vstack([
                    np.interp(time_grid, model_times, row, left=row[0], right=row[-1])
                    for row in surv_vals
                ])
                # derive a single risk score per sample: negative area under the survival curve
                # (lower area -> lower survival -> higher risk; negative makes higher risk => larger score)
                pred_scores = -np.trapz(preds, x=time_grid, axis=1)

            # ---------------------------
            # Compute metrics (AUC, mean AUC, Brier, IBS, cindex)
            # ---------------------------
            # cumulative_dynamic_auc expects (structured y_train, y_test, score_vector, times=time_grid)
            auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, pred_scores, times=time_grid)

            # concordance_index_censored wants (event_indicator, event_time, pred_scores)
            ev_test, t_test = _extract_event_time_from_structured(y_test)
            cindex = concordance_index_censored(ev_test, t_test, pred_scores)[0]

            # Brier score expects preds shape (n_samples, n_times)
            brier_scores = brier_score(y_train, y_test, preds, time_grid)[1]
            ibs = integrated_brier_score(y_train, y_test, preds, time_grid)

            performance.append({
                "fold": fold,
                "auc": auc,
                "mean_auc": mean_auc,
                "auc_at_5y": (auc[np.where(time_grid == 5.0)[0][0]] if 5.0 in time_grid else None),
                "brier_t": brier_scores,
                "ibs": ibs,
                "cindex": cindex
            })
            print(f"  Fold {fold} - C-index: {cindex:.3f}, Mean AUC: {mean_auc:.3f}, IBS: {ibs:.3f}", flush=True)

        except Exception as e:
            # keep same behavior as before on errors: log and append a failed entry
            print(f"  Skipping fold {fold} due to error during evaluation: {e}", flush=True)
            performance.append({
                "fold": fold,
                "auc": None,
                "mean_auc": None,
                "auc_at_5y": None,
                "brier_t": None,
                "ibs": None,
                "cindex": None,
                "error": str(e)
            })

    return performance

