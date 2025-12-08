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
                print("Not scaling specified variables.", flush=True)

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
            # Construct inner CV pipeline
            # ---------------------------

            pipe = make_pipeline(
                preproc,
                RandomSurvivalForest(n_jobs=1,n_estimators=400) # set here for training, later override nestimators
            )
            #print(pipe.get_params().keys())

            # Wrap with scorer for inner CV
            scorer_pipe = as_concordance_index_ipcw_scorer(pipe)

            #print("Estimator params:", sorted(scorer_pipe.get_params().keys()))

            inner_model = RandomizedSearchCV(
                scorer_pipe,
                param_distributions=param_grid, 
                cv=inner_cv,
                error_score=0.5,
                n_jobs=-1, # parallize here
                refit=True,
                n_iter=50,  # The number of parameter settings that are sampled.
                random_state=42,
                verbose=1
            )

            #inner_model = GridSearchCV(
            #    scorer_pipe,
            #    param_grid=param_grid,
            #    cv=inner_cv,
            #    error_score=0.5,
            #    n_jobs=-1,
            #    refit=True,
            #    verbose=1
            #)

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
                "error": None
            })

    # ---------------------------
    # Done with outer CV
    # ---------------------------
    return outer_models

# ==============================================================================
