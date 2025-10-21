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
from xgbse import XGBSEDebiasedBCE
from sksurv.metrics import as_concordance_index_ipcw_scorer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ==============================================================================
# FUNCTIONS
# ==============================================================================

# Helper function already in your xgbse_functions
def _extract_event_time_from_structured(y_struct):
    names = y_struct.dtype.names
    events = y_struct[names[0]].astype(bool)
    times = y_struct[names[1]].astype(float)
    return events, times

# ==============================================================================

def define_param_grid(X=None, y=None):
    param_grid = {
        "estimator__xgbsedebiasedbce__xgb_params": [
            {
                "max_depth": 3, "learning_rate": 0.01, "n_estimators": 200, "subsample": 0.8, "colsample_bynode": 0.5
            },
            {
                "max_depth": 3, "learning_rate": 0.01, "subsample": 0.8, "colsample_bynode": 1.0
            },
            {
                "max_depth": 5, "learning_rate": 0.1, "n_estimators": 500, "subsample": 1.0, "colsample_bynode": 1.0
            }
        ]
    }
    return param_grid


# ==============================================================================

def run_nested_cv_xgbse(X, y, param_grid, 
                      outer_cv_folds=5, inner_cv_folds=3, top_n_variance=5000, 
                      filter_func=None,
                      dont_filter_vars=None, dont_scale_vars=None,
                      dont_penalize_vars=None):
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

    # ---------------------------
    # Setup: CV splitters and bookkeeping
    # ---------------------------
    print(f"\n=== Running nested CV with {outer_cv_folds} outer folds "
          f"and {inner_cv_folds} inner folds ===\n", flush=True)

    event_labels = y["RFi_event"]
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=96)
    inner_cv = KFold(n_splits=inner_cv_folds, shuffle=True, random_state=96)

    outer_models = []

    # ---------------------------
    # Outer CV loop
    # ---------------------------
    for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X, event_labels)):
        # Subset data for this outer fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"\nOuter fold {fold_num}: {sum(y_train['RFi_event'])} events in training set.", flush=True)

        # Ensure preproc is defined for later refit check (avoid stale variable leakage)
        preproc = None 

        try:
            # ---------------------------
            # Feature filtering (fold-specific)
            # ---------------------------
            
            # Keep top variance features in X_train but always include dont_filter_vars
            #selected_cpgs = variance_filter(X_train, top_n=top_n_variance, keep_vars=dont_filter_vars)
            
            # TEST THIS; ADD ARGUMENT HERE AND ALSO IN THE ESTIMATE ALPHA
            selected_cpgs = filter_func(X_train, y_train, top_n=top_n_variance, keep_vars=dont_filter_vars)

            # Subset both train and test to the selected features for this fold
            X_train, X_test = X_train[selected_cpgs], X_test[selected_cpgs]


            # ---------------------------
            # Build pipeline for inner CV (must be constructed per-fold because columns changed)
            # ---------------------------
            if dont_scale_vars is not None:
                # If some variables should NOT be scaled (e.g., encoded clinical dummies),
                # scale everything else and passthrough the dont_scale_vars.
                print("Not scaling specified variables.", flush=True)

                scale_cols = [c for c in X_train.columns if c not in dont_scale_vars]

                transformers = []
                if len(scale_cols) > 0:
                    transformers.append(("scale", StandardScaler(), scale_cols))
                if len(dont_scale_vars) > 0:
                    transformers.append(("passthrough_encoded", "passthrough", dont_scale_vars))

                preproc = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False,
                                            remainder="drop")

                # Get feature order after transformation
                preproc.fit(X_train)
                feature_names = preproc.get_feature_names_out()

                #print("Pipeline feature names:", feature_names)

            else:
                # All features scaled
                preproc = StandardScaler()
                feature_names = X_train.columns.values  # order is preserved
                
            # ---------------------------
            # Construct inner CV pipeline
            # ---------------------------
            #estimator_cls = CoxnetSurvivalAnalysis
            #estimator_kwargs = {"penalty_factor": penalty_factor}

            pipe = make_pipeline(
                preproc,
                XGBSEDebiasedBCE() #n_intervals=20
            )
            #print(pipe.get_params().keys())

            # Wrap with scorer for inner CV
            scorer_pipe = as_concordance_index_ipcw_scorer(pipe)
            inner_model = GridSearchCV(
                scorer_pipe,
                param_grid=param_grid,
                cv=inner_cv,
                error_score=0.5,
                n_jobs=-1,
                refit=True
            )

            # ---------------------------
            # Fit inner CV
            # ---------------------------
            inner_model.fit(X_train, y_train)
            best_params = inner_model.best_params_
            print(f"\t--> Fold {fold_num}: Best params {best_params}", flush=True)

            # ---------------------------
            # Refit final pipeline for this fold
            # ---------------------------
            # Extract the best parameters for your XGBSE estimator
            # Replace the Coxnet-specific keys with your XGBSE estimator name
            best_xgb_params = {k.replace("estimator__xgbsedebiasedbce__", ""): v
                            for k, v in best_params.items() if k.startswith("estimator__xgbsedebiasedbce__")}

            # Build pipeline with preprocessing + XGBSE using best params
            refit_pipe = make_pipeline(
                preproc,  # reuse exact preprocessing from inner CV
                XGBSEDebiasedBCE(**best_xgb_params)
            )
            refit_pipe.fit(X_train, y_train)

            # ---------------------------
            # Store outer fold results
            # ---------------------------
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
            # Handle errors 
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

    # ---------------------------
    # Done with outer CV
    # ---------------------------
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

import numpy as np
import pandas as pd
from sksurv.metrics import cumulative_dynamic_auc, brier_score, integrated_brier_score, concordance_index_censored

def _interp_preds_to_grid(preds_df, desired_times):
    """
    preds_df: pd.DataFrame with columns = time_bins (floats), rows = samples, values = survival probs
    desired_times: 1d array-like of times we want (must be within preds_df.columns range for interpolation)
    Returns: np.array shape (n_samples, len(desired_times))
    """
    # ensure float columns and sorted ascending
    cols = np.array([float(c) for c in preds_df.columns])
    sort_idx = np.argsort(cols)
    cols_sorted = cols[sort_idx]
    mat = preds_df.values[:, sort_idx]  # shape (n_samples, n_cols)

    # if desired_times exactly match columns, quick path
    desired_times = np.asarray(desired_times, dtype=float)
    if np.array_equal(cols_sorted, desired_times):
        return mat

    # interpolate each row over the columns to the desired times
    # np.interp requires x to be 1d increasing
    # If desired_times are outside the range, np.interp will extrapolate with edge values -> avoid by clipping
    t_min, t_max = cols_sorted[0], cols_sorted[-1]
    if (desired_times < t_min).any() or (desired_times > t_max).any():
        # Option A: clip desired_times into available range (safe but reduces requested extremes)
        desired_clipped = np.clip(desired_times, t_min, t_max)
    else:
        desired_clipped = desired_times

    # perform interpolation row-wise
    interp_mat = np.vstack([np.interp(desired_clipped, cols_sorted, row) for row in mat])
    return interp_mat


def evaluate_outer_models_xgbse(outer_models, X, y, time_grid):
    """
    Evaluate performance of XGBSE models from outer CV folds.

    Returns same dict structure as your Cox/RSF evaluator.
    """
    print("\n=== Evaluating XGBSE outer models ===\n", flush=True)
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

        # Extract event/time arrays robustly from structured arrays
        events_train, times_train = _extract_event_time_from_structured(y_train)
        events_test, times_test = _extract_event_time_from_structured(y_test)

        print(f"  Fold {fold} - test set min time: {times_test.min():.3f}, max time: {times_test.max():.3f}", flush=True)
        print(f"  Fold {fold} - train set min time: {times_train.min():.3f}, max time: {times_train.max():.3f}", flush=True)

        # Subset X_test
        X_test = X_test[selected_features]

        # ---------- Get survival matrix from XGBSE ----------
        # XGBSE .predict(X) returns a DataFrame: columns=time_bins, rows=samples, values=cumulative survival probs
        preds_df = model.predict(X_test)  # pd.DataFrame

        # Ensure columns are numeric (floats) and sorted
        try:
            preds_df.columns = [float(c) for c in preds_df.columns]
        except Exception:
            # if columns are multiindex or something odd, coerce to float safely
            preds_df.columns = [float(str(c)) for c in preds_df.columns]

        # Interpolate / reorder preds to match time_grid requested by caller
        preds = _interp_preds_to_grid(preds_df, time_grid)  # shape (n_samples, len(time_grid))

        # ---------- Build 1D risk score for ranking-based metrics ----------
        # A common and robust choice is negative area under the survival curve:
        # lower survival -> higher risk (negative area)
        # Integrate with trapezoid rule across time_grid
        # If time_grid is not strictly increasing or length 1 handle gracefully:
        if preds.shape[1] == 1:
            # single timepoint: risk = negative survival prob at that time
            risk = -preds[:, 0]
        else:
            # integrate survival curve from min to max of provided time_grid
            risk = -np.trapz(preds, x=np.asarray(time_grid), axis=1)

        # ---------- Compute time-dependent AUC ----------
        # cumulative_dynamic_auc accepts either 1D risk or 2D risk matrix; using 1D scalar risk is fine.
        try:
            auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, times=time_grid)
        except Exception as e:
            print(f"  Warning: cumulative_dynamic_auc failed for fold {fold}: {e}", flush=True)
            auc = np.full(len(time_grid), np.nan)
            mean_auc = np.nan

        # ---------- Concordance index ----------
        try:
            cindex = concordance_index_censored(events_test, times_test, risk)[0]
        except Exception as e:
            print(f"  Warning: concordance_index_censored failed for fold {fold}: {e}", flush=True)
            cindex = np.nan

        # ---------- Brier scores and IBS ----------
        # sksurv.metrics.brier_score / integrated_brier_score expect estimate shape (n_samples, n_times)
        try:
            brier_scores = brier_score(y_train, y_test, preds, time_grid)[1]  # returns (times, scores) tuple -> [1] is array of scores
            ibs = integrated_brier_score(y_train, y_test, preds, time_grid)
        except Exception as e:
            print(f"  Warning: Brier score computation failed for fold {fold}: {e}", flush=True)
            brier_scores = np.full(len(time_grid), np.nan)
            ibs = np.nan

        performance.append({
            "fold": fold,
            "auc": auc,
            "mean_auc": mean_auc,
            "auc_at_5y": auc[np.where(np.isclose(time_grid, 5.0))[0][0]] if 5.0 in time_grid else None,
            "brier_t": brier_scores,
            "ibs": ibs,
            "cindex": cindex
        })

        print(f"  Fold {fold} - C-index: {cindex:.3f}, Mean AUC: {mean_auc:.3f}, IBS: {ibs:.3f}", flush=True)

    return performance
