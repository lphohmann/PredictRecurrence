#!/usr/bin/env python
# Script: Functions for GBM Survival pipeline
# Author: lennart hohmann 

# ==============================================================================
# IMPORTS
# ==============================================================================

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
# from sklearn.ensemble import RandomForestClassifier # Not needed for GBM, remove for clarity
from sksurv.ensemble import GradientBoostingSurvivalAnalysis # <--- CHANGE: Import GBM model
from sksurv.metrics import (cumulative_dynamic_auc, concordance_index_censored,
                             brier_score, integrated_brier_score,
                             as_concordance_index_ipcw_scorer,
                             as_cumulative_dynamic_auc_scorer)
import matplotlib.pyplot as plt
# from sklearn.inspection import permutation_importance # <--- Removed as per your request

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def define_param_grid():
    """
    Define a more compact and computationally feasible parameter grid for
    GradientBoostingSurvivalAnalysis hyperparameters for nested CV.
    Focuses on the most impactful parameters.
    """
    param_grid = {
        # n_estimators: Number of boosting stages (trees).
        # We need a range, but keep it manageable.
        "estimator__gradientboostingsurvivalanalysis__n_estimators": [100, 200], # Fewer options

        # learning_rate: Controls the step size. Crucial for speed and performance.
        # Often best to keep this relatively small and adjust n_estimators.
        "estimator__gradientboostingsurvivalanalysis__learning_rate": [0.05, 0.1], # Fewer options

        # max_depth: Limits the depth of individual trees. Small values prevent overfitting.
        # This is a very impactful parameter for model complexity.
        "estimator__gradientboostingsurvivalanalysis__max_depth": [3, 5], # Keep it shallow and focused

        # subsample: Fraction of samples for fitting each tree. Helps with regularization.
        # Common values are usually in this range.
        "estimator__gradientboostingsurvivalanalysis__subsample": [0.8, 1.0], # 1.0 is no subsampling

        # loss: For survival, 'coxph' is almost always what you want.
        # Fixing it reduces the search space without compromising much.
        "estimator__gradientboostingsurvivalanalysis__loss": ["coxph"]
    }

    print(f"Defined compact GBM parameter grid:\n{param_grid}")
    return param_grid

# ==============================================================================

def run_nested_cv(X, y, param_grid, outer_cv_folds, inner_cv_folds,
                  inner_scorer="concordance_index_ipcw", auc_scorer_times=None):
    """
    Run nested cross-validation for GBM survival model.
    Outer loop estimates performance, inner loop tunes hyperparameters.
    """

    # --- CHANGE START ---
    print(f"\n=== Running nested CV for GBM Survival with {outer_cv_folds} outer folds and {inner_cv_folds} inner folds (scorer: {inner_scorer}) ===\n", flush=True)
    # --- CHANGE END ---

    # Stratify outer folds by event indicator to maintain event proportion
    event_labels = y["RFi_event"]
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=21)
    inner_cv = KFold(n_splits=inner_cv_folds, shuffle=True, random_state=12) # KFold is fine for inner CV

    # --- CHANGE START ---
    # Base GBM pipeline
    # Note: GradientBoostingSurvivalAnalysis also takes a random_state
    pipe = make_pipeline(GradientBoostingSurvivalAnalysis(random_state=42))
    # --- CHANGE END ---
    print(pipe.get_params().keys())

    # Choose scoring method for inner CV
    # This section remains largely the same as the scorers in sksurv are model-agnostic
    if inner_scorer == "concordance_index_ipcw":
        scorer_pipe = as_concordance_index_ipcw_scorer(pipe)
    elif inner_scorer == "cumulative_dynamic_auc":
        if auc_scorer_times is None:
            raise ValueError("Specify auc_scorer_times when using cumulative_dynamic_auc scorer.")
        scorer_pipe = as_cumulative_dynamic_auc_scorer(pipe, times=auc_scorer_times)
    else:
        raise ValueError(f"Unsupported inner_scorer: {inner_scorer}")

    # Set up GridSearchCV on the scorer-wrapped pipeline
    inner_model = GridSearchCV(
        scorer_pipe,
        param_grid=param_grid,
        cv=inner_cv,
        error_score=0.5, # Important: Set error_score to a reasonable value for survival metrics
        n_jobs=-1,
        refit=True
    )

    outer_models = []
    for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X, event_labels)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"\nOuter fold {fold_num}: {sum(y_train['RFi_event'])} events in training set.", flush=True)

        try:
            # Inner CV for hyperparameter tuning
            inner_model.fit(X_train, y_train)
            best_model = inner_model.best_estimator_  # pipeline with GBM fitted on X_train
            best_params = inner_model.best_params_

            print(f"\n\t--> Fold {fold_num}: Best grid-search parameters {best_params}\n", flush=True)

            # --- CHANGE START ---
            # Refit best GBM with exact best hyperparameters (on full outer train set)
            # Extract and clean hyperparameters for GradientBoostingSurvivalAnalysis
            #gbm_params = {
            #    key.split("gradientboostingsurvivalanalysis__")[1]: value
            #    for key, value in best_params.items()
            #    if "gradientboostingsurvivalanalysis__" in key
            #}

            #refit_best_model = make_pipeline(
            #    GradientBoostingSurvivalAnalysis(
            #        **gbm_params,  # Unpack best hyperparameters
            #        random_state=42 # Ensure reproducibility
            #    )
            #)
            #refit_best_model.fit(X_train, y_train)
            # --- CHANGE END ---

            outer_models.append({
                "fold": fold_num,
                "model": best_model,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "cv_results": inner_model.cv_results_,
                "error": None
            })

        except Exception as e:
            print(f"\n\nSkipping fold {fold_num} due to error: {e}\n\n", flush=True)
            outer_models.append({
                "fold": fold_num,
                "model": None,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "cv_results": None,
                "error": str(e)
            })

    return outer_models

# ==============================================================================
# The following functions are largely model-agnostic and should work as-is.
# They operate on the outputs of the `run_nested_cv` and `evaluate_outer_models`.
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
              f"Test samples: {len(entry.get('test_idx', []))}")
        print(f"  Error: {entry.get('error')}")
    print()

# ==============================================================================

def evaluate_outer_models(outer_models, X, y, time_grid):
    """
    Evaluate each trained GBM on its test fold, computing:
      - Concordance index (c-index)
      - Time-dependent AUC(t) and mean AUC
      - Brier score over time and Integrated Brier Score (IBS)
    """
    print("\n=== Evaluating outer models ===\n", flush=True)
    performance = []

    for entry in outer_models:
        fold = entry['fold']
        print(f"Evaluating fold {fold}...", flush=True)
        if entry["model"] is None:
            print(f"  Skipping fold {fold} (no model).", flush=True)
            continue

        model = entry["model"]
        test_idx = entry["test_idx"]
        train_idx = entry["train_idx"]

        X_test, X_train = X.iloc[test_idx], X.iloc[train_idx]
        y_test, y_train = y[test_idx], y[train_idx]

        # Predict risk scores and survival functions on test set
        # `predict` method for GradientBoostingSurvivalAnalysis gives risk scores (log-hazard)
        pred_scores = model.predict(X_test)
        # `predict_survival_function` method is also available
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

# ==============================================================================

def plot_brier_scores(brier_array, ibs_array, folds, time_grid, outfile):
    """
    Plot time-dependent Brier scores for each fold and IBS per fold.
    """
    print("Plotting Brier scores...", flush=True)
    plt.style.use('seaborn-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                   gridspec_kw={'height_ratios': [3, 1]})
    # Brier curves
    for i, brier in enumerate(brier_array):
        ax1.plot(time_grid, brier, label=f'Fold {folds[i]}', alpha=0.6)
    ax1.plot(time_grid, np.mean(brier_array, axis=0), color='black', lw=3,
             label='Mean Brier Score')
    ax1.set_title("Time-dependent Brier Score")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Brier Score")
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 0.5)

    # IBS bar chart
    colors = plt.cm.Paired(np.linspace(0, 1, len(folds)))
    bars = ax2.bar(folds, ibs_array, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_title("Integrated Brier Score (IBS) per Fold")
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("IBS")
    ax2.set_ylim(0, max(ibs_array) * 1.1 if len(ibs_array) else 1)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.005, f'{yval:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Saved Brier plot to {outfile}", flush=True)

# ==============================================================================

def plot_auc_curves(performance, time_grid, outfile):
    """
    Plot time-dependent AUC(t) curves for all folds and their mean.
    """
    print("Plotting time-dependent AUC curves...", flush=True)
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 6))

    # Plot each fold's AUC curve
    for p in performance:
        plt.plot(time_grid, p["auc"], label=f'Fold {p["fold"]}', alpha=0.5)
    # Mean AUC curve
    mean_auc_curve = np.mean([p["auc"] for p in performance], axis=0)
    plt.plot(time_grid, mean_auc_curve, color='black', lw=2.5, label='Mean AUC')
    plt.title("Time-dependent AUC(t) per Fold")
    plt.xlabel("Time")
    plt.ylabel("AUC(t)")
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Saved AUC plot to {outfile}", flush=True)

# ==============================================================================

def summarize_performance(performance):
    """
    Print summary statistics (mean ± std) for evaluation metrics across folds.
    """
    # --- CHANGE START ---
    print("\n=== GBM Survival Evaluation Summary ===\n", flush=True)
    # --- CHANGE END ---
    cindexes = [p["cindex"] for p in performance]
    auc5y = [p["auc_at_5y"] for p in performance if p["auc_at_5y"] is not None]
    mean_aucs = [p["mean_auc"] for p in performance]
    ibs_vals = [p["ibs"] for p in performance]
    print(f"C-index: mean={np.mean(cindexes):.3f} ± {np.std(cindexes):.3f}")
    if auc5y:
        print(f"AUC@5y: mean={np.mean(auc5y):.3f} ± {np.std(auc5y):.3f}")
    print(f"Mean AUC over times: {np.mean(mean_aucs):.3f} ± {np.std(mean_aucs):.3f}")
    print(f"Integrated Brier Score (IBS): {np.mean(ibs_vals):.3f} ± {np.std(ibs_vals):.3f}")
    print()

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

# Permutation importance function removed as per your request.