#!/usr/bin/env python
# Script: Functions for penalized Cox model
# Author: Lennart Hohmann

# ==============================================================================
# IMPORTS
# ==============================================================================

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import as_concordance_index_ipcw_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import warnings
from sklearn.exceptions import FitFailedWarning
import copy
from sklearn.base import clone

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
        #preproc.fit(X)  
        #feature_names = preproc.get_feature_names_out()

    else:
        # All features scaled
        preproc = StandardScaler()
        #preproc.fit(X)  
        #feature_names = X.columns.values  # order is preserved

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
                      dont_penalize_vars=None):
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
    #inner_cv = KFold(n_splits=inner_cv_folds, shuffle=True, random_state=96)

    outer_models = []

    # ---------------------------
    # Outer CV loop
    # ---------------------------

    for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X, event_labels)):
        # Subset data for this outer fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"\nOuter fold {fold_num}: "
              f"Train={sum(y_train['RFi_event'])} events, "
              f"Val={sum(y_test['RFi_event'])} events",
              flush=True)
        # Ensure preproc is defined for later refit check (avoid stale variable leakage)
        preproc = None 

        # this is to save ids belaongi to each fold
        train_ids = X_train.index.values
        test_ids = X_test.index.values

        # Initialize variables for error handling
        selected_features_1 = None
        #selected_features_2 = None
        feature_names = None

        try:
            # ---------------------------
            # Feature filtering (fold-specific)
            # ---------------------------
            
            # filter 1
            if filter_func_1 is not None:
                selected_features_1 = filter_func_1(X_train, y_train, keep_vars=dont_filter_vars)
                X_train = X_train[selected_features_1]
                X_test  = X_test[selected_features_1]

            else:
                selected_features_1 = list(X_train.columns)
                X_train = X_train[selected_features_1]
                X_test  = X_test[selected_features_1]

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
                preproc.fit(X_train)
                feature_names = X_train.columns.values  # Order preserved for StandardScaler

            # ---------------------------
            # Build penalty factor vector
            # ---------------------------

            # Build penalty factor with verification
            penalty_factor = np.ones(len(feature_names), dtype=float)
            if dont_penalize_vars is not None:
                matched_features = []
                for i, fname in enumerate(feature_names):
                    if fname in dont_penalize_vars:
                        penalty_factor[i] = 0.0
                        matched_features.append(fname)
                
                # Verify all dont_penalize_vars were found
                unmatched = set(dont_penalize_vars) - set(matched_features)
                if unmatched:
                    print(f"WARNING: These dont_penalize_vars were not found in transformed features: {unmatched}")
                    print(f"Available feature names after transformation: {list(feature_names)[:10]}...")  # Show first 10
            #print(penalty_factor)

            # ---------------------------
            # For case that only clinical vars use coxph
            # ---------------------------

            if np.all(penalty_factor == 0.0): # not possible to run completely unpenalized, force negligible penliaztion
                penalty_factor[:] = 1e-8
                # Use Coxnet with very small ridge penalty, basically non-penalized for only clin vars
                print("All features unpenalized -> using near-unpenalized Coxnet for this fold.", flush=True)
                estimator_cls = CoxnetSurvivalAnalysis
                estimator_kwargs = {"penalty_factor": penalty_factor,
                                    #"l1_ratio": 0.01,         # force pure ridge
                                    "fit_baseline_model": True
                                    }
                
                # Modify param_grid to force l1_ratio=0.0 for this fold
                #param_grid_fold = param_grid.copy()  # Make a copy to avoid modifying original
                #param_grid_fold["estimator__coxnetsurvivalanalysis__l1_ratio"] = [0.01]

                # 1. Create a deep copy of the original param_grid to avoid modification in later folds.
                # 2. Modify *each* dictionary in the list to enforce the fixed L1 ratio and a broad alpha range.
                param_grid_fold = copy.deepcopy(param_grid) 
                
                # Modify ALL dictionaries in the list
                for pdict in param_grid_fold:
                    # Enforce the L1 ratio to 0.01 (pure ridge with near-zero penalty)
                    pdict["estimator__coxnetsurvivalanalysis__l1_ratio"] = [0.001]
                    # Force small alphas to keep effective penalty truly negligible
                    pdict["estimator__coxnetsurvivalanalysis__alphas"] = [[1e-6]]

            else:
                estimator_cls = CoxnetSurvivalAnalysis
                estimator_kwargs = {"penalty_factor": penalty_factor}
                param_grid_fold = param_grid  # Use original grid

            # ---------------------------
            # Construct inner CV pipeline with stratification
            # ---------------------------

            # Extract event labels for stratification in inner CV
            train_events = y_train["RFi_event"]

            # Pre-compute stratified inner CV splits
            inner_cv_splits = list(StratifiedKFold(
                n_splits=inner_cv_folds, 
                shuffle=True, 
                random_state=96
            ).split(X_train, train_events))

            # Print event distribution for each inner fold
            print(f"\tFold {fold_num}: Created {len(inner_cv_splits)} stratified inner CV folds", flush=True)
            for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_cv_splits):
                inner_train_events = train_events[inner_train_idx].sum()
                inner_val_events = train_events[inner_val_idx].sum()
                
                print(f"\t  Inner fold {inner_fold_idx}: "
                    f"Train={inner_train_events} events, "
                    f"Val={inner_val_events} events", 
                    flush=True)
                
            pipe = make_pipeline(
                clone(preproc),
                estimator_cls(**estimator_kwargs)
                #CoxnetSurvivalAnalysis(penalty_factor=penalty_factor)
            )
            #print(pipe.get_params().keys())

            # Wrap with scorer for inner CV
            scorer_pipe = as_concordance_index_ipcw_scorer(pipe)

            inner_model = GridSearchCV(
                scorer_pipe,
                param_grid=param_grid_fold,
                cv=inner_cv_splits,
                error_score=0.5,
                n_jobs=-1,
                refit=True,
                verbose=1
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
                clone(preproc),  # reuse the exact preprocessing
                CoxnetSurvivalAnalysis(
                    alphas=[alpha_to_use],
                    l1_ratio=best_l1,
                    fit_baseline_model=True,
                    penalty_factor=penalty_factor  # keep same unpenalized vars
                )
            )
            refit_pipe.fit(X_train, y_train)

            # ---------------------------
            # Check selected features
            # ---------------------------
            # Extract Coxnet step from the fitted pipeline
            coxnet = refit_pipe.named_steps["coxnetsurvivalanalysis"]

            # Get non-zero coefficients
            coefs = coxnet.coef_.flatten()
            nonzero_mask = coefs != 0
            
            # Map to feature names actually used in this fold (after preprocessing)
            model_features = np.array(feature_names)[nonzero_mask].tolist()  # feature_names already set above

            # Check which unpenalized vars ended up with zero coefs
            #if dont_penalize_vars is not None:
            #    for var in dont_penalize_vars:
            #        if var in feature_names:
            #            idx = list(feature_names).index(var)
            #            print(f"\t{var}: coef={coefs[idx]:.4f}, penalty={penalty_factor[idx]}")

            # ---------------------------
            # Store outer fold results
            # ---------------------------
            outer_models.append({
                "fold": fold_num,
                "model": refit_pipe,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "train_ids": train_ids,
                "test_ids": test_ids, 
                "cv_results": inner_model.cv_results_,
                "features_after_filter1": selected_features_1,
                "features_after_filter2": None,
                "input_training_features": feature_names,
                "features_in_model": model_features,
                "error": None
            })

        except Exception as e:
            # Handle errors 
            print(f"Skipping fold {fold_num} due to error: {e}", flush=True)
            outer_models.append({
                "fold": fold_num,
                "model": None,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "train_ids": train_ids,
                "test_ids": test_ids, 
                "cv_results": None,
                "features_after_filter1": selected_features_1,
                "features_after_filter2": None,
                "input_training_features": feature_names,
                "features_in_model": None,
                "error": str(e)
            })

    # ---------------------------
    # Done with outer CV
    # ---------------------------

    return outer_models

# ==============================================================================

def train_final_aggregated_coxnet(X, y, outer_models, 
                                  filter_func_1=None,
                                  dont_filter_vars=None, 
                                  dont_scale_vars=None,
                                  dont_penalize_vars=None):
    """
    Train final model on full dataset using aggregated hyperparameters from CV folds.
    Returns a dict matching the outer_models structure.
    
    Args:
        X (pd.DataFrame): Full feature matrix.
        y (structured array): Full survival outcome (event, time).
        outer_models (list): Results from run_nested_cv_coxnet.
        filter_func_1 (callable): Feature filtering function (same as used in CV).
        dont_filter_vars (list): Feature names to always keep during filtering.
        dont_scale_vars (list): Feature names to NOT scale.
        dont_penalize_vars (list): Feature names to NOT penalize.
    
    Returns:
        dict: Final model in same structure as outer_models entries.
    """
    
    print(f"\n=== Training final aggregated model on full dataset ===\n", flush=True)
    
    # ---------------------------
    # Extract and aggregate hyperparameters
    # ---------------------------
    alphas_per_fold = []
    l1_ratios_per_fold = []
    
    for fold_result in outer_models:
        if fold_result['model'] is None or fold_result.get('error') is not None:
            continue
        
        coxnet = fold_result['model'].named_steps['coxnetsurvivalanalysis']
        alphas_per_fold.append(coxnet.alphas_[0])
        l1_ratios_per_fold.append(coxnet.l1_ratio)
    
    if len(alphas_per_fold) == 0:
        raise ValueError("No successful folds to aggregate from!")
    
    # Aggregate
    agg_alpha = np.median(alphas_per_fold)
    agg_l1 = np.median(l1_ratios_per_fold)

    print(f"Aggregated alpha: {agg_alpha:.6f} (median of {len(alphas_per_fold)} folds)")
    print(f"Aggregated l1_ratio: {agg_l1:.3f}")
    
    # ---------------------------
    # Apply same preprocessing as CV folds
    # ---------------------------
    X_train = X.copy()
    y_train = y.copy()
    
    # Apply filtering
    if filter_func_1 is not None:
        selected_features_1 = filter_func_1(X_train, y_train, keep_vars=dont_filter_vars)
        X_train = X_train[selected_features_1]
    else:
        selected_features_1 = list(X_train.columns)
    
    # Build preprocessing
    if dont_scale_vars is not None:
        scale_cols = [c for c in X_train.columns if c not in dont_scale_vars]
        
        transformers = []
        if len(scale_cols) > 0:
            transformers.append(("scale", StandardScaler(), scale_cols))
        if len(dont_scale_vars) > 0:
            transformers.append(("passthrough_encoded", "passthrough", dont_scale_vars))
        
        preproc = ColumnTransformer(transformers=transformers,
                                        verbose_feature_names_out=False,
                                        remainder="drop")
        
        preproc.fit(X_train)
        feature_names = preproc.get_feature_names_out()
    else:
        preproc = StandardScaler()
        preproc.fit(X_train)
        feature_names = X_train.columns.values
    
    # Build penalty factor
    penalty_factor = np.ones(len(feature_names), dtype=float)
    if dont_penalize_vars is not None:
        for i, fname in enumerate(feature_names):
            if fname in dont_penalize_vars:
                penalty_factor[i] = 0.0
    
    if np.all(penalty_factor == 0.0):
        penalty_factor[:] = 1e-8
    
    # Build and fit final model
    final_model = make_pipeline(
        clone(preproc),
        CoxnetSurvivalAnalysis(
            alphas=[agg_alpha],
            l1_ratio=agg_l1,
            fit_baseline_model=True,
            penalty_factor=penalty_factor
        )
    )
    
    final_model.fit(X_train, y_train)
    print("Final model training complete!")
    
    # Extract features
    coxnet = final_model.named_steps['coxnetsurvivalanalysis']
    coefs = coxnet.coef_.flatten()
    nonzero_mask = coefs != 0
    model_features = np.array(feature_names)[nonzero_mask].tolist()
    
    print(f"Selected {len(model_features)}/{len(feature_names)} features\n")
    
    # ---------------------------
    # Return in same structure as outer_models
    # ---------------------------
    return {
        "fold": "final_aggregated",
        "model": final_model,
        "train_idx": np.arange(len(X)),
        "test_idx": None,
        "train_ids": X.index.values,
        "test_ids": None,
        "cv_results": {
            'aggregated_alpha': agg_alpha,
            'aggregated_l1_ratio': agg_l1,
            'individual_alphas': alphas_per_fold,
            'individual_l1_ratios': l1_ratios_per_fold
        },
        "features_after_filter1": selected_features_1,
        "features_after_filter2": None,
        "input_training_features": feature_names,
        "features_in_model": model_features,
        "error": None
    }

# ==============================================================================

def train_final_coxnet_model(X_train, y_train, param_grid, 
                             filter_func_1=None, filter_func_2=None,
                             dont_filter_vars=None, dont_scale_vars=None,
                             dont_penalize_vars=None):
    """
    Train final CoxNet model on full training data using inner CV for hyperparameter selection.
    Uses the same preprocessing pipeline as nested CV.
    
    Args:
        X_train: Feature matrix
        y_train: Structured array with survival outcome
        param_grid: Hyperparameter grid for CoxNet (list of dicts)
        filter_func_1: First feature filtering function
        filter_func_2: Second feature filtering function
        dont_filter_vars: Variables to keep regardless of filtering
        dont_scale_vars: Variables to not scale (e.g., dummy encoded)
        dont_penalize_vars: Variables to not penalize in CoxNet
    
    Returns:
        dict: Final model in same structure as outer_models entries
    """
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.base import clone
    import copy
    
    print("\n=== Training Final COXNET Model on Full Training Data ===\n", flush=True)
    
    # Initialize variables
    preproc = None
    selected_features_1 = None
    selected_features_2 = None
    feature_names = None
    
    # ---------------------------
    # Apply same filtering as during nested CV
    # ---------------------------
    X_filtered = X_train.copy()
    
    if filter_func_1 is not None:
        selected_features_1 = filter_func_1(X_filtered, y_train, keep_vars=dont_filter_vars)
        X_filtered = X_filtered[selected_features_1]
        print(f"After filter 1: {X_filtered.shape[1]} features", flush=True)
    else:
        selected_features_1 = list(X_filtered.columns)
        X_filtered = X_filtered[selected_features_1]
    
    if filter_func_2 is not None:
        selected_features_2 = filter_func_2(X_filtered, y_train, keep_vars=dont_filter_vars)
        X_filtered = X_filtered[selected_features_2]
        print(f"After filter 2: {X_filtered.shape[1]} features", flush=True)
    
    # ---------------------------
    # Build preprocessing (same as nested CV)
    # ---------------------------
    if dont_scale_vars is not None:
        print("Not scaling specified variables.", flush=True)
        
        scale_cols = [c for c in X_filtered.columns if c not in dont_scale_vars]
        
        transformers = []
        if len(scale_cols) > 0:
            transformers.append(("scale", StandardScaler(), scale_cols))
        if len(dont_scale_vars) > 0:
            transformers.append(("passthrough_encoded", "passthrough", dont_scale_vars))
        
        preproc = ColumnTransformer(
            transformers=transformers,
            verbose_feature_names_out=False,
            remainder="drop"
        )
        
        # Get feature order after transformation
        preproc.fit(X_filtered)
        feature_names = preproc.get_feature_names_out()
    else:
        # All features scaled
        preproc = StandardScaler()
        preproc.fit(X_filtered)
        feature_names = X_filtered.columns.values
    
    # ---------------------------
    # Build penalty factor vector
    # ---------------------------
    penalty_factor = np.ones(len(feature_names), dtype=float)
    if dont_penalize_vars is not None:
        matched_features = []
        for i, fname in enumerate(feature_names):
            if fname in dont_penalize_vars:
                penalty_factor[i] = 0.0
                matched_features.append(fname)
        
        # Verify all dont_penalize_vars were found
        unmatched = set(dont_penalize_vars) - set(matched_features)
        if unmatched:
            print(f"WARNING: These dont_penalize_vars were not found in transformed features: {unmatched}")
            print(f"Available feature names after transformation: {list(feature_names)[:10]}...")
    
    # ---------------------------
    # Handle all-unpenalized case
    # ---------------------------
    if np.all(penalty_factor == 0.0):
        penalty_factor[:] = 1e-8
        print("All features unpenalized -> using near-unpenalized Coxnet.", flush=True)
        
        estimator_cls = CoxnetSurvivalAnalysis
        estimator_kwargs = {
            "penalty_factor": penalty_factor,
            "fit_baseline_model": True
        }
        
        # Modify param_grid for this special case
        param_grid_fold = copy.deepcopy(param_grid)
        for pdict in param_grid_fold:
            pdict["estimator__coxnetsurvivalanalysis__l1_ratio"] = [0.001]
            pdict["estimator__coxnetsurvivalanalysis__alphas"] = [[1e-6]]
    else:
        estimator_cls = CoxnetSurvivalAnalysis
        estimator_kwargs = {
            "penalty_factor": penalty_factor,
            "fit_baseline_model": True
        }
        param_grid_fold = param_grid
    
    # ---------------------------
    # Create inner CV splits (stratified on events)
    # ---------------------------
    train_events = y_train["RFi_event"]
    
    inner_cv_splits = list(StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=96
    ).split(X_filtered, train_events))
    
    print(f"Created {len(inner_cv_splits)} stratified inner CV folds for hyperparameter tuning", flush=True)
    
    # Print event distribution for each inner fold
    for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_cv_splits):
        inner_train_events = train_events[inner_train_idx].sum()
        inner_val_events = train_events[inner_val_idx].sum()
        
        print(f"  Inner fold {inner_fold_idx}: "
              f"Train={inner_train_events} events, "
              f"Val={inner_val_events} events", 
              flush=True)
    
    # ---------------------------
    # Build pipeline for hyperparameter search
    # ---------------------------
    pipe = make_pipeline(
        clone(preproc),
        estimator_cls(**estimator_kwargs)
    )
    
    # Wrap with scorer for inner CV
    scorer_pipe = as_concordance_index_ipcw_scorer(pipe)
    
    # ---------------------------
    # Hyperparameter search (using GridSearchCV like your nested CV)
    # ---------------------------
    inner_model = GridSearchCV(
        scorer_pipe,
        param_grid=param_grid_fold,
        cv=inner_cv_splits,
        error_score=0.5,
        n_jobs=-1,
        refit=True,
        verbose=1
    )
    
    print("\nStarting hyperparameter search...", flush=True)
    inner_model.fit(X_filtered, y_train)
    
    best_params = inner_model.best_params_
    print(f"\n{'='*60}")
    print(f"Best params: {best_params}")
    print(f"Best CV C-index: {inner_model.best_score_:.3f}")
    print(f"{'='*60}\n", flush=True)
    
    # ---------------------------
    # Refit final pipeline (same logic as your nested CV)
    # ---------------------------
    best_alphas = best_params["estimator__coxnetsurvivalanalysis__alphas"]
    best_l1 = best_params["estimator__coxnetsurvivalanalysis__l1_ratio"]
    alpha_to_use = best_alphas[0] if hasattr(best_alphas, "__len__") else best_alphas
    
    refit_pipe = make_pipeline(
        clone(preproc),
        CoxnetSurvivalAnalysis(
            alphas=[alpha_to_use],
            l1_ratio=best_l1,
            fit_baseline_model=True,
            penalty_factor=penalty_factor
        )
    )
    
    print("Training final model with best hyperparameters...", flush=True)
    refit_pipe.fit(X_filtered, y_train)
    
    # ---------------------------
    # Extract selected features (non-zero coefficients)
    # ---------------------------
    coxnet = refit_pipe.named_steps["coxnetsurvivalanalysis"]
    coefs = coxnet.coef_.flatten()
    nonzero_mask = coefs != 0
    model_features = np.array(feature_names)[nonzero_mask].tolist()
    
    print(f" Final CoxNet model trained on {len(X_filtered)} samples")
    print(f" Selected {len(model_features)}/{len(feature_names)} features (non-zero coefficients)\n", flush=True)
    
    # ---------------------------
    # Return in same structure as outer_models
    # ---------------------------
    return {
        "fold": "final",
        "model": refit_pipe,
        "train_idx": np.arange(len(X_train)),
        "test_idx": None,
        "train_ids": X_train.index.values,
        "test_ids": None,
        "cv_results": inner_model.cv_results_,
        "features_after_filter1": selected_features_1,
        "features_after_filter2": selected_features_2,
        "input_training_features": feature_names.tolist(),
        "features_in_model": model_features,
        "error": None
    }

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
        features_in_model = entry["features_in_model"]
        selected_features = entry.get("input_training_features", None)

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
        #print(f"Vanity check features_in_model: {len(features_in_model)}; features: {features_in_model}")