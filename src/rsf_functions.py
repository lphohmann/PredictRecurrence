#!/usr/bin/env python
# Script: Functions for RSF pipeline
# Author: lennart hohmann 

# ==============================================================================
# IMPORTS
# ==============================================================================

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sksurv.metrics import as_concordance_index_ipcw_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sksurv.ensemble import RandomSurvivalForest
import pickle
from sklearn.model_selection import RandomizedSearchCV
import copy

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def run_nested_cv_rsf(X, y, param_grid, 
                      outer_cv_folds=5, 
                      inner_cv_folds=3, #top_n_variance=5000, 
                      filter_func_1=None,
                      filter_func_2=None,
                      dont_filter_vars=None, 
                      dont_scale_vars=None):
    """
    Run nested cross-validation for RSF

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
        selected_features_2 = None
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

            # filter 2
            if filter_func_2 is not None:
                selected_features_2 = filter_func_2(X_train, y_train, keep_vars=dont_filter_vars)
                X_train = X_train[selected_features_2]
                X_test  = X_test[selected_features_2]

            else:
                selected_features_2 = list(X_train.columns)
                X_train = X_train[selected_features_2]
                X_test  = X_test[selected_features_2]

            # ---------------------------
            # Build pipeline for inner CV (must be constructed per-fold because columns changed)
            # ---------------------------

            if dont_scale_vars is not None:
                # If some variables should NOT be scaled (e.g., encoded clinical dummies),
                # scale everything else and passthrough the dont_scale_vars.
                print("\tNot scaling specified variables.", flush=True)

                scale_cols = [c for c in X_train.columns if c not in dont_scale_vars]

                transformers = []
                if len(scale_cols) > 0:
                    transformers.append(("scale", StandardScaler(), scale_cols))
                if len(dont_scale_vars) > 0:
                    transformers.append(("passthrough_encoded", "passthrough", dont_scale_vars))

                preproc = ColumnTransformer(transformers=transformers, 
                                            verbose_feature_names_out=False,
                                            remainder="drop")

                # Get feature order after transformation
                preproc.fit(X_train)
                feature_names = preproc.get_feature_names_out()

                #print("Pipeline feature names:", feature_names)

            else:
                # All features scaled
                preproc = StandardScaler()
                preproc.fit(X_train)
                feature_names = X_train.columns.values  # order is preserved

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
                preproc,
                RandomSurvivalForest(n_jobs=1, n_estimators=400)
            )

            scorer_pipe = as_concordance_index_ipcw_scorer(pipe)

            inner_model = RandomizedSearchCV(
                scorer_pipe,
                param_distributions=param_grid, 
                cv=inner_cv_splits,  # Use pre-computed stratified splits
                error_score=0.5,
                n_jobs=-1,
                refit=True,
                n_iter=50,
                random_state=42,
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

            # 1. Extract only the estimator's parameters
            estimator_prefix = [k for k in best_params.keys() if k.startswith("estimator__")]
            
            # 2. Strip the full pipeline prefix to get clean RSF params (e.g., 'n_estimators')
            estimator_params = {
                k.replace("estimator__randomsurvivalforest__", ""): best_params[k] 
                for k in estimator_prefix
            }

            # After finding best params
            estimator_params['n_estimators'] = 1000  # Use more for final model
            estimator_params['n_jobs'] = -1 # Use all cores for final training
            
            # 3. Manually reconstruct and fit the final pipe
            refit_pipe = make_pipeline(
                preproc,
                RandomSurvivalForest(**estimator_params)
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
                "train_ids": train_ids,
                "test_ids": test_ids, 
                "cv_results": inner_model.cv_results_,
                "features_after_filter1": selected_features_1,
                "features_after_filter2": selected_features_2,
                "input_training_features": feature_names,
                "features_in_model": feature_names,
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
                "features_after_filter2": selected_features_2,
                "input_training_features": feature_names,
                "features_in_model": None,
                "error": str(e)
            })

    # ---------------------------
    # Done with outer CV
    # ---------------------------
    return outer_models

# ==============================================================================

def train_final_rsf_model(X_train, y_train, param_grid, 
                          filter_func_1=None, filter_func_2=None,
                          dont_filter_vars=None, dont_scale_vars=None):
    """
    Train final Random Survival Forest model on full training data using inner CV 
    for hyperparameter selection. Uses the same preprocessing pipeline as nested CV.
    
    Args:
        X_train: Feature matrix
        y_train: Structured array with survival outcome
        param_grid: Hyperparameter grid for RSF (dict for RandomizedSearchCV)
        filter_func_1: First feature filtering function
        filter_func_2: Second feature filtering function
        dont_filter_vars: Variables to keep regardless of filtering
        dont_scale_vars: Variables to not scale (e.g., dummy encoded)
    
    Returns:
        dict: Final model in same structure as outer_models entries
    """
    from sksurv.ensemble import RandomSurvivalForest
    from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.base import clone
    import numpy as np
    
    print("\n=== Training Final RSF Model on Full Training Data ===\n", flush=True)
    
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
    else:
        selected_features_2 = list(X_filtered.columns)
        X_filtered = X_filtered[selected_features_2]
    
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
        preproc,
        RandomSurvivalForest(n_jobs=1, n_estimators=400)
    )
    
    scorer_pipe = as_concordance_index_ipcw_scorer(pipe)
    
    # ---------------------------
    # Hyperparameter search (using RandomizedSearchCV like your nested CV)
    # ---------------------------
    inner_model = RandomizedSearchCV(
        scorer_pipe,
        param_distributions=param_grid,
        cv=inner_cv_splits,
        error_score=0.5,
        n_jobs=-1,
        refit=True,
        n_iter=50,
        random_state=42,
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
    # Extract only the estimator's parameters
    estimator_prefix = [k for k in best_params.keys() if k.startswith("estimator__")]
    
    # Strip the full pipeline prefix to get clean RSF params
    estimator_params = {
        k.replace("estimator__randomsurvivalforest__", ""): best_params[k] 
        for k in estimator_prefix
    }
    
    # Use more trees for final model
    estimator_params['n_estimators'] = 1000
    estimator_params['n_jobs'] = -1
    
    # Manually reconstruct and fit the final pipe
    refit_pipe = make_pipeline(
        preproc,
        RandomSurvivalForest(**estimator_params)
    )
    
    print("Training final model with best hyperparameters (n_estimators=1000)...", flush=True)
    refit_pipe.fit(X_filtered, y_train)
    
    print(f" Final RSF model trained on {len(X_filtered)} samples")
    print(f" Using {X_filtered.shape[1]} features\n", flush=True)
    
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
        "features_in_model": feature_names.tolist(),  # RSF uses all features
        "error": None
    }