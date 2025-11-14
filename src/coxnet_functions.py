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
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_censored, brier_score, integrated_brier_score
import matplotlib.pyplot as plt
from sksurv.metrics import as_concordance_index_ipcw_scorer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from src.utils import variance_filter #, estimate_alpha_grid
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import warnings
from sklearn.exceptions import FitFailedWarning
import pickle

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def estimate_alpha_grid(X, y, l1_ratio, n_alphas, alpha_min_ratio='auto',
                        filter_func_1=None,
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
    
    if filter_func_1 is not None:
        selected_cpgs = filter_func_1(X,y,keep_vars=dont_filter_vars)
        
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
# ==============================================================================

def run_nested_cv_coxnet(X, y, param_grid, 
                      outer_cv_folds=5, inner_cv_folds=3, 
                      filter_func_1=None,
                      dont_filter_vars=None, dont_scale_vars=None,
                      dont_penalize_vars=None,
                      output_fold_ids_file=None):
    """
    Run nested cross-validation for Coxnet.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (structured array): Survival outcome (event, time).
        param_grid (dict): Hyperparameter grid for inner CV.
        outer_cv_folds (int): Number of outer CV folds.
        inner_cv_folds (int): Number of inner CV folds.
        top_n_variance (int): Number of top variance features to keep per fold.
        dont_filter_vars (list or None): Feature names to always keep during filtering.
        dont_scale_vars (list or None): Feature names to NOT scale (passthrough in preproc).
        dont_penalize_vars (list or None): Feature names to NOT penalize
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
    fold_id_dict = {} # to save ids

    for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X, event_labels)):
        # Subset data for this outer fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"\nOuter fold {fold_num}: {sum(y_train['RFi_event'])} events in training set.", flush=True)
        # Ensure preproc is defined for later refit check (avoid stale variable leakage)
        preproc = None 

        # this is to save ids belaongi to each fold
        train_ids = X_train.index.values
        test_ids = X_test.index.values
        fold_key = f"fold{fold_num}"
        fold_id_dict[fold_key] = {
            "train_ids": train_ids,
            "test_ids": test_ids,
            "features_after_filter1": None,
            "features_after_filter2": None,
        }


        try:
            # ---------------------------
            # Feature filtering (fold-specific)
            # ---------------------------
            
            # filter 1
            if filter_func_1 is not None:
                selected_features_1 = filter_func_1(X_train, y_train, keep_vars=dont_filter_vars)
                X_train = X_train[selected_features_1]
                X_test  = X_test[selected_features_1]

                fold_id_dict[fold_key]["features_after_filter1"] = list(selected_features_1)
            else:
                fold_id_dict[fold_key]["features_after_filter1"] = list(X_train.columns)

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
            # Build penalty factor vector
            # ---------------------------

            penalty_factor = np.ones(len(feature_names), dtype=float)
            if dont_penalize_vars is not None:
                for i, fname in enumerate(feature_names):
                    # if feature name matches one to not penalize
                    if fname in dont_penalize_vars:
                        penalty_factor[i] = 0.0

            #print(penalty_factor)

            # ---------------------------
            # For case that only clinical vars use coxph
            # ---------------------------

            if np.all(penalty_factor == 0.0):
                penalty_factor[:] = 1e-8
                # Use Coxnet with very small ridge penalty, basically non-penalized for only clin vars
                print("All features unpenalized -> using near-unpenalized Coxnet for this fold.", flush=True)
                estimator_cls = CoxnetSurvivalAnalysis
                estimator_kwargs = {"penalty_factor": penalty_factor,
                                    "l1_ratio": 0.0,         # force pure ridge
                                    "fit_baseline_model": True
                                    }

            else:
                estimator_cls = CoxnetSurvivalAnalysis
                estimator_kwargs = {"penalty_factor": penalty_factor}

            # ---------------------------
            # Construct inner CV pipeline
            # ---------------------------

            pipe = make_pipeline(
                preproc,
                estimator_cls(**estimator_kwargs)
                #CoxnetSurvivalAnalysis(penalty_factor=penalty_factor)
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
            best_alphas = best_params["estimator__coxnetsurvivalanalysis__alphas"]
            best_l1 = best_params["estimator__coxnetsurvivalanalysis__l1_ratio"]
            alpha_to_use = best_alphas[0] if hasattr(best_alphas, "__len__") else best_alphas

            refit_pipe = make_pipeline(
                preproc,  # reuse the exact preprocessing
                CoxnetSurvivalAnalysis(
                    alphas=[alpha_to_use],
                    l1_ratio=best_l1,
                    fit_baseline_model=True,
                    penalty_factor=penalty_factor  # keep same unpenalized vars
                )
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

    # save ids dict
    if output_fold_ids_file is not None:
        with open(output_fold_ids_file, "wb") as f:
            pickle.dump(fold_id_dict, f)
        print(f"Saved outer fold IDs dictionary to {output_fold_ids_file}")

    return outer_models

# ==============================================================================
def print_selected_cpgs_counts_coxnet(outer_models):
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
        #print(coxnet.coef_)

        nonzero_mask = coefs != 0

        # Map to feature names actually used in this fold
        selected_cpgs = np.array(selected_features)[nonzero_mask]

        print(f"Fold {fold}: {len(selected_cpgs)} CpGs selected", flush=True)
        print(f"  CpGs: {selected_cpgs.tolist()}", flush=True)



