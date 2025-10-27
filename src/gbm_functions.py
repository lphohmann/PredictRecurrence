#!/usr/bin/env python
# Script: Functions for GBM Survival pipeline
# Author: lennart hohmann 

# ==============================================================================
# IMPORTS
# ==============================================================================

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sksurv.metrics import as_concordance_index_ipcw_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def run_nested_cv_gbm(X, y, param_grid, 
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

            pipe = make_pipeline(
                preproc,
                GradientBoostingSurvivalAnalysis()
            )
            #print(pipe.get_params().keys())

            # Wrap with scorer for inner CV
            scorer_pipe = as_concordance_index_ipcw_scorer(pipe)

            #print("Estimator params:", sorted(scorer_pipe.get_params().keys()))

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
            # Extract only the estimator's parameters
            estimator_prefix = [k for k in best_params.keys() if k.startswith("estimator__")]

            # Strip the "estimator__" prefix for refit
            estimator_params = {k.replace("estimator__gradientboostingsurvivalanalysis__", ""): best_params[k] for k in estimator_prefix}

            # Refit using the right estimator
            refit_pipe = make_pipeline(
                preproc,
                GradientBoostingSurvivalAnalysis(**estimator_params)
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
