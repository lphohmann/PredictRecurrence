#!/usr/bin/env python
# Script: Functions for penalized Cox model
# Author: Lennart Hohmann

# ==============================================================================
# IMPORTS
# ==============================================================================

import pandas as pd
import numpy as np
import warnings
import math
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import FitFailedWarning
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored, brier_score, integrated_brier_score
import matplotlib.pyplot as plt
import joblib

# ==============================================================================
# FUNCTIONS
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

def variance_filter(df, min_variance=None, top_n=None):
    """
    Filter features (CpGs) based on variance. Extects CpGs as columns and patients as rows
    Args:
        X (pd.DataFrame): Feature matrix (samples x features).
        top_n (int, optional): Number of top features to select based on variance.
        min_variance (float, optional): Minimum variance threshold to retain features.
    Returns:
        pd.DataFrame: Filtered feature matrix.
    """
    variances = df.var(axis=0)
    
    if min_variance is not None:
        # Filter features with variance above threshold
        selected_features = variances[variances >= min_variance].index
    elif top_n is not None:
        # Select top_n features by variance
        selected_features = variances.sort_values(ascending=False).head(top_n).index
    else:
        raise ValueError("Either min_variance or top_n must be specified")
    
    return df.loc[:, selected_features]

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

def preprocess_data(beta_matrix, top_n_cpgs):
    """
    Converts beta values to M-values and retains most variable CpGs.

    Args:
        beta_matrix (DataFrame): Raw beta methylation matrix.
        top_n_cpgs (int): Number of most variable CpGs to retain.

    Returns:
        DataFrame: Preprocessed M-values matrix.
    """
    # Convert beta values to M-values with a threshold 
    mvals = beta2m(beta_matrix, beta_threshold=0.001)
    # Apply variance filtering to retain top N most variable CpGs
    mvals = variance_filter(mvals, top_n=top_n_cpgs)
    return mvals

# ==============================================================================

def define_param_grid(X, y, n_alphas=30):
    """
    Estimates a suitable grid of alpha values for Coxnet hyperparameter tuning.

    Args:
        X (DataFrame): Feature matrix.
        y (structured array): Survival labels (from sksurv).
        n_alphas (int): Number of alpha values to generate.

    Returns:
        dict: Parameter grid dictionary for GridSearchCV.
    """
    # fit a Coxnet model to estimate reasonable alpha values for grid search
    initial_pipe = make_pipeline(
        CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.1, n_alphas=n_alphas)
    )
    warnings.simplefilter("ignore", FitFailedWarning)
    warnings.simplefilter("ignore", UserWarning)

    # Fit on full data just to estimate alphas
    initial_pipe.fit(X, y)
    estimated_alphas = initial_pipe.named_steps["coxnetsurvivalanalysis"].alphas_

    # Set up full parameter grid for tuning
    param_grid = {
        "coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]#,
        #"coxnetsurvivalanalysis__l1_ratio": [0.1, 0.5, 0.9]  # Elastic Net mixing
    }
    return param_grid

# ==============================================================================

def run_nested_cv(X, y, param_grid, outer_cv_folds, inner_cv_folds):
    """
    Runs nested cross-validation to tune CoxNet model and assess generalization.

    Args:
        X (DataFrame): Feature matrix.
        y (structured array): Survival labels.
        param_grid (dict): Grid of hyperparameters for inner CV.
        outer_cv_folds (int): Number of outer CV folds.
        inner_cv_folds (int): Number of inner CV folds.

    Returns:
        list: Dictionary for each outer fold containing model, results, indices, and errors.
    """
    # stratified outer cv for performance est
    event_labels = y["RFi_event"]
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=21)

    # Inner CV for hyperparameter tuning
    inner_cv = KFold(n_splits=inner_cv_folds, shuffle=True, random_state=12)

    # Define the model and wrap in GridSearchCV (for the inner loop)
    inner_model = GridSearchCV(
        estimator=make_pipeline(CoxnetSurvivalAnalysis(l1_ratio=0.9)),
        param_grid=param_grid,
        cv=inner_cv,
        error_score=0.5,
        n_jobs=-1
    )

    # store best outer models
    outer_models = []

    # for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X)): # non strat cv
    for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X, event_labels)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"Fold {fold_num} has {sum(y_train['RFi_event'])} events.", flush=True)

        try:
            inner_model.fit(X_train, y_train)
            best_model = inner_model.best_estimator_
            best_alpha = best_model.named_steps["coxnetsurvivalanalysis"].alphas_[0]
            best_l1_ratio = best_model.named_steps["coxnetsurvivalanalysis"].l1_ratio

            # Refit best model with fit_baseline_model=True for later eval
            refit_best_model = make_pipeline(
                CoxnetSurvivalAnalysis(
                    alphas=[best_alpha],
                    l1_ratio=best_l1_ratio,
                    fit_baseline_model=True,
                    max_iter=100000
                )
            )
            refit_best_model.fit(X_train, y_train)

            outer_models.append({
                "fold": fold_num,
                "model": refit_best_model,
                "test_idx": test_idx,
                "train_idx": train_idx,
                "cv_results": inner_model.cv_results_,
                "error": None
            })

        except ArithmeticError as e:
            print(f"Skipping fold {fold_num} due to numerical error: {e}", flush=True)
            outer_models.append({
                "fold": fold_num,
                "model": None,
                "test_idx": test_idx,
                "train_idx": train_idx,
                "cv_results": None,
                "error": str(e)
            })

    return outer_models

# ==============================================================================

def summarize_outer_models(outer_models):
    """
    Prints a summary of each outer CV fold result, including model info,
    index sizes, CV scores, and any errors encountered.

    Args:
        outer_models (list): List of dictionaries, one per outer CV fold,
                             containing model, indices, cv_results, and error info.
    """
    for i, entry in enumerate(outer_models):
        print(f"\n--- Fold {i} ---")
        print(f"Fold number: {entry.get('fold')}")
        print(f"Model type: {type(entry.get('model'))}")
        print(f"Train indices: {len(entry.get('train_idx', []))} samples")
        print(f"Test indices: {len(entry.get('test_idx', []))} samples")
        
        error = entry.get("error")
        if error:
            print(f"Error: {error}")
        else:
            print("Error: None")

# ==============================================================================

def evaluate_outer_models(outer_models, X, y, time_grid):
    """
    Evaluate performance of each outer CV fold model.

    For each fold, computes:
    - AUC(t)
    - Mean AUC
    - Brier score at each time point
    - Integrated Brier score
    - Concordance index (C-index)
    - AUC at 5 years

    Skips folds where the model is missing or linear predictor values indicate overflow risk.

    Args:
        outer_models (list): List of dictionaries with trained models and fold metadata.
        X (pd.DataFrame): Feature matrix.
        y (structured array): Survival outcome.
        time_grid (np.ndarray): Time grid to evaluate metrics on.

    Returns:
        list of dicts: One dictionary per fold with performance metrics.
    """
    performance = []

    for entry in outer_models:
        print(f"current outer cv fold model: {entry['fold']}", flush=True)
        if entry["model"] is None:
            print(f"skipping: {entry['fold']}", flush=True)
            continue  # skip failed folds
        
        # for this fold
        model = entry["model"]
        test_idx = entry["test_idx"]
        train_idx = entry["train_idx"]

        X_test = X.iloc[test_idx]
        X_train = X.iloc[train_idx]
        y_test = y[test_idx]
        y_train = y[train_idx]

        # Compute linear predictor manually to check if this fold is stable
        coefs = model.named_steps["coxnetsurvivalanalysis"].coef_
        linear_predictor = X_test @ coefs
        # Detect dangerous folds
        if linear_predictor.max().item() > 700 or linear_predictor.min().item() < -700:
            print(f"⚠️ Fold {entry['fold']} skipped due to overflow risk.")
            continue
        
        # Predict risk scores and surv function for each pateint in fold test set
        pred_scores = model.predict(X_test)
        surv_funcs = model.predict_survival_function(X_test)
        # survival probabilities at each time point per patient
        preds = np.row_stack([
            [fn(t) for t in time_grid] for fn in surv_funcs
        ])

        # Compute AUC(t)
        auc, mean_auc = cumulative_dynamic_auc(
            y_train, y_test,
            pred_scores,
            times=time_grid
        )
        # AUC at 5 years
        auc_at_5y = auc[np.where(time_grid == 5.0)[0][0]]

        # Compute C-index
        cindex = concordance_index_censored(
            y_test["RFi_event"],
            y_test["RFi_years"],
            pred_scores
        )[0]

        # Brier score for each time point
        brier_scores = brier_score(y_train, y_test, preds, time_grid)[1]
        # area under the Brier score curve
        ibs = integrated_brier_score(y_train, y_test, preds, time_grid)

        performance.append({
            "fold": entry["fold"],
            "model": model,
            "auc": auc,
            "mean_auc": mean_auc,
            "brier_t": brier_scores,
            "ibs": ibs,
            "cindex": cindex,
            "auc_at_5y": auc_at_5y
        })

    return performance

# ==============================================================================

def plot_brier_scores(brier_array, ibs_array, folds, time_grid, outfile):
    """
    Plot time-dependent Brier Score(t) per fold and integrated Brier Score (IBS).
    """
    plt.style.use('seaborn-whitegrid')
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Plot Brier Score(t) curves
    for i, brier in enumerate(brier_array):
        axs[0].plot(time_grid, brier, label=f'Fold {folds[i]}', alpha=0.6)
    axs[0].plot(time_grid, brier_array.mean(axis=0), color='black', lw=3, label='Mean Brier Score(t)')

    axs[0].set_title("Time-dependent Brier Score", fontsize=16, fontweight='bold')
    axs[0].set_xlabel("Time", fontsize=14)
    axs[0].set_ylabel("Brier Score(t)", fontsize=14)
    axs[0].legend(title='Folds', loc='upper right', fontsize=10)
    axs[0].set_ylim(0, 0.5)
    axs[0].grid(True, linestyle='--', alpha=0.7)

    # Plot IBS as bar chart
    bar_colors = plt.cm.Paired(np.linspace(0, 1, len(folds)))
    bars = axs[1].bar(folds, ibs_array, color=bar_colors, edgecolor='black', alpha=0.85)

    axs[1].set_title("Integrated Brier Score (IBS) per Fold", fontsize=16, fontweight='bold')
    axs[1].set_xlabel("Fold", fontsize=14)
    axs[1].set_ylabel("IBS", fontsize=14)
    axs[1].set_ylim(0, max(ibs_array)*1.15)
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axs[1].annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================

def plot_auc_curves(performance, time_grid, outfile):
    """
    Plot time-dependent AUC(t) curves for all folds and the mean curve.
    """
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 6))

    # Extract AUC(t) curves
    auc_curves = np.array([p["auc"] for p in performance])
    assert all(len(p["auc"]) == len(time_grid) for p in performance), "Mismatch in auc curve lengths!"

    # Plot AUC(t) for each fold
    for p in performance:
        plt.plot(time_grid, p["auc"], label=f'Fold {p["fold"]}', alpha=0.5)

    # Mean AUC(t)
    mean_auc_curve = auc_curves.mean(axis=0)
    plt.plot(time_grid, mean_auc_curve, color='black', linewidth=2.5, label='Mean AUC(t)')

    plt.title("Time-dependent AUC(t) per Fold")
    plt.xlabel("Time")
    plt.ylabel("AUC(t)")
    plt.ylim(0, 1)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================

def summarize_performance(performance):
    """
    Compute and print summary statistics (mean ± std) for model evaluation metrics across folds.
    """
    mean_cindex = np.mean([p["cindex"] for p in performance])
    std_cindex = np.std([p["cindex"] for p in performance])
    mean_auc_5y = np.mean([p["auc_at_5y"] for p in performance])
    std_auc_5y = np.std([p["auc_at_5y"] for p in performance])
    mean_ibs = np.mean([p["ibs"] for p in performance])
    std_ibs = np.std([p["ibs"] for p in performance])

    valid_mean_aucs = [
        p["mean_auc"] for p in performance
        if p["mean_auc"] is not None and not math.isnan(p["mean_auc"])
    ]
    mean_auc_avg = np.mean(valid_mean_aucs)
    mean_auc_std = np.std(valid_mean_aucs)

    print("\n============================")
    print("Model Evaluation Summary")
    print("============================")
    print(f"Mean Concordance Index (C-index): {mean_cindex:.2f} (±{std_cindex:.2f})")
    print(f"Time-dependent AUC at 5 years:    {mean_auc_5y:.2f} (±{std_auc_5y:.2f})")
    print(f"Mean AUC over all times:           {mean_auc_avg:.2f} (±{mean_auc_std:.2f})")
    print(f"Integrated Brier Score (IBS):     {mean_ibs:.2f} (±{std_ibs:.2f})")

# ==============================================================================

def select_best_model(performance, outer_models, metric):
    """
    Select the best model based on a given performance metric.

    Parameters:
        performance (list of dict): List of performance dictionaries per fold.
        outer_models (list of dict): List of outer model fold entries (with 'fold', 'model', etc.).
        metric (str): Metric to select best model. Must be one of: "ibs", "mean_auc", "auc_at_5y".
        outfile (str): File path to save the best model (via joblib).

    Returns:
        dict: The best model's performance dictionary.
    """
    assert metric in {"ibs", "mean_auc", "auc_at_5y"}, f"Unsupported metric: {metric}"

    if metric == "ibs":
        best_model_perf = min(performance, key=lambda p: p[metric])
    elif metric in ["mean_auc", "auc_at_5y"]:
        best_model_perf = max(performance, key=lambda p: p[metric])

    print(f"\nBest model by {metric}:")
    print(f"  Fold: {best_model_perf['fold']}")
    print(f"  AUC at 5y: {best_model_perf['auc_at_5y']:.4f}")
    print(f"  IBS: {best_model_perf['ibs']:.4f}")
    print(f"  Mean AUC(t): {best_model_perf['mean_auc']:.4f}")

    # Find matching outer model entry by fold number
    fold_number = best_model_perf['fold']
    best_outer_fold = next((e for e in outer_models if e['fold'] == fold_number), None)

    return best_outer_fold
