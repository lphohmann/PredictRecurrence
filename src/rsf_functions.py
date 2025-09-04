#!/usr/bin/env python
# Script: Functions for Random Survival Forest pipeline
# Author: lennart hohmann

# ==============================================================================
# IMPORTS
# ==============================================================================

import numpy as np
from sksurv.metrics import (cumulative_dynamic_auc, concordance_index_censored,
                             brier_score, integrated_brier_score)

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def define_param_grid(X=None, y=None):
    """
    Define a parameter grid for RandomSurvivalForest hyperparameters. 
    The grid values can be adjusted as needed.
    """
    # Example grid: tune number of trees, min_samples_split/leaf, max_features
    param_grid = {
        "estimator__randomsurvivalforest__n_estimators": [100, 300],
        "estimator__randomsurvivalforest__max_features": [0.2, 0.5, "sqrt", "log2"],
        "estimator__randomsurvivalforest__min_samples_split": [5, 10],
        "estimator__randomsurvivalforest__min_samples_leaf": [5, 10],
        "estimator__randomsurvivalforest__max_depth": [3, 5, None]  # added
    }

    print(f"Defined RSF parameter grid: {param_grid}")
    return param_grid

# ==============================================================================

def evaluate_outer_models(outer_models, X, y, time_grid):
    """
    Evaluate each trained RSF on its test fold, computing:
      - Concordance index (c-index)
      - Time-dependent AUC(t) and mean AUC
      - Brier score over time and Integrated Brier Score (IBS)
    """
    print("\n=== Evaluating outer models ===\n", flush=True)
    performance = []

    print(f"Global evaluation time grid: {time_grid}", flush=True)
    print(f"Global min time: {y['RFi_years'].min()}, max time: {y['RFi_years'].max()}\n", flush=True)


    for entry in outer_models:
        fold = entry['fold']
        print(f"Evaluating fold {fold}...", flush=True)
        if entry["model"] is None:
            print(f"  Skipping fold {fold} (no model).", flush=True)
            continue

        model = entry["model"]
        test_idx = entry["test_idx"]
        train_idx = entry["train_idx"]

        # Get the features used in training this fold
        selected_features = entry.get("selected_cpgs", X.columns)  # fallback to all columns if missing
        X_test, X_train = X.iloc[test_idx], X.iloc[train_idx]
        y_test, y_train = y[test_idx], y[train_idx]

        print(f"  Fold {fold} - test set min time: {y_test['RFi_years'].min():.3f}, max time: {y_test['RFi_years'].max():.3f}", flush=True)
        print(f"  Fold {fold} - train set min time: {y_train['RFi_years'].min():.3f}, max time: {y_train['RFi_years'].max():.3f}", flush=True)

        # Subset X_train and X_test
        #X_train = X_train[selected_features]
        X_test = X_test[selected_features]

        # Predict risk scores and survival functions on test set
        pred_scores = model.predict(X_test)  # RSF.predict gives risk (sum of cumulative hazard)
        surv_funcs = model.predict_survival_function(X_test)
        # Build array of survival probabilities for each patient x time grid
        preds = np.vstack([[fn(t) for t in time_grid] for fn in surv_funcs])

        # Compute time-dependent AUC and mean AUC
        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, pred_scores, times=time_grid)
        # Compute c-index (concordance) on test set
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
