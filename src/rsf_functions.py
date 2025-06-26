#!/usr/bin/env python
# Script: Functions for Random Survival Forest pipeline
# Author: lennart hohmann

# ==============================================================================
# IMPORTS
# ==============================================================================

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier  # for type hints (not directly used)
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (cumulative_dynamic_auc, concordance_index_censored,
                             brier_score, integrated_brier_score,
                             as_concordance_index_ipcw_scorer,
                             as_cumulative_dynamic_auc_scorer)
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

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

def run_nested_cv(X, y, param_grid, outer_cv_folds, inner_cv_folds, 
                  inner_scorer="concordance_index_ipcw", auc_scorer_times=None):
    """
    Run nested cross-validation for RSF. 
    Outer loop estimates performance, inner loop tunes hyperparameters.
    """
    
    print(f"\n=== Running nested CV for CoxNet with {outer_cv_folds} outer folds and {inner_cv_folds} inner folds (scorer: {inner_scorer}) ===\n", flush=True)

    # Stratify outer folds by event indicator to maintain event proportion
    event_labels = y["RFi_event"]
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=21)
    inner_cv = KFold(n_splits=inner_cv_folds, shuffle=True, random_state=12)

    # Base RSF pipeline (no one-hot or scaling since data is numeric features)
    pipe = make_pipeline(RandomSurvivalForest(random_state=42))
    print(pipe.get_params().keys())

    # Choose scoring method for inner CV
    if inner_scorer == "concordance_index_ipcw":
        scorer_pipe = as_concordance_index_ipcw_scorer(pipe)
    elif inner_scorer == "cumulative_dynamic_auc":
        if auc_scorer_times is None:
            raise ValueError("Specify auc_scorer_times when using cumulative_dynamic_auc scorer.")
        scorer_pipe = as_cumulative_dynamic_auc_scorer(pipe, times=auc_scorer_times)
    #elif inner_scorer is None:
    #    scorer_pipe = pipe  # use default scoring (Harrell’s C-index)
    else:
        raise ValueError(f"Unsupported inner_scorer: {inner_scorer}")

    # Set up GridSearchCV on the scorer-wrapped pipeline
    inner_model = GridSearchCV(
        scorer_pipe,
        param_grid=param_grid,
        cv=inner_cv,
        error_score=0.5,
        n_jobs=-1,
        refit=True #Refit an estimator using the best found parameters on the whole dataset
    )

    outer_models = []
    for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X, event_labels)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"\nOuter fold {fold_num}: {sum(y_train['RFi_event'])} events in training set.", flush=True)

        try:
            # Inner CV for hyperparameter tuning
            inner_model.fit(X_train, y_train)
            best_model = inner_model.best_estimator_  # pipeline with RSF fitted on X_train
            best_params = inner_model.best_params_

            print(f"\n\t--> Fold {fold_num}: Best grid-search parameters {best_params}\n", flush=True)

            # We can use best_model directly; RandomSurvivalForest is already fitted on X_train.
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
    Evaluate each trained RSF on its test fold, computing:
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
    print("\n=== RSF Evaluation Summary ===\n", flush=True)
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

def compute_permutation_importance(outer_models, X, y, n_repeats=15, random_state=42):
    """
    Compute permutation feature importance on the test set for each outer fold model.

    Args:
        outer_models (list): List of dicts with trained models and test indices from outer CV.
        X (DataFrame): Full feature matrix.
        y (structured array): Full survival outcome.
        n_repeats (int): Number of permutation repeats.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: Mapping from fold number to feature importances DataFrame (or None if failed).
    """
    print("\n=== Computing permutation feature importances for outer models ===\n", flush=True)

    importances_by_fold = {}

    for entry in outer_models:
        fold = entry['fold']
        model = entry.get('model')
        test_idx = entry.get('test_idx')

        if model is None:
            print(f"Skipping fold {fold} (no trained model).", flush=True)
            importances_by_fold[fold] = None
            continue

        X_test = X.iloc[test_idx]
        y_test = y[test_idx]

        print(f"Computing permutation importance for fold {fold}...", flush=True)

        try:
            result = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=-1
            )

            importances_df = pd.DataFrame({
                "importances_mean": result["importances_mean"],
                "importances_std": result["importances_std"]
            }, index=X.columns).sort_values(by="importances_mean", ascending=False)

            importances_by_fold[fold] = importances_df
            print(f"Fold {fold} - Top features:\n{importances_df.head()}\n", flush=True)

        except Exception as e:
            print(f"Error computing permutation importance for fold {fold}: {e}", flush=True)
            importances_by_fold[fold] = None

    return importances_by_fold