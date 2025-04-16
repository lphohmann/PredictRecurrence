#!/usr/bin/env python3
################################################################################
# Script: Cox proportional-hazards LASSO model based on patient DNA methylation and outcome data
# Author: Lennart Hohmann
# Date: 11.04.2025
################################################################################

# import
import os
import sys
import pandas as pd
import numpy as np
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
from src.utils import beta2m, m2beta, create_surv, variance_filter, unicox_filter, preprocess, train_cox_lasso

# set wd
os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))

################################################################################
################################################################################

# input paths
infile_1 = "./data/raw/tnbc235.csv" # replace with PC dat later
infile_2 = "./data/raw/tnbc_anno.csv" # replace with tnbc dat

# output paths
#outfile_1 = 

################################################################################
# load data
################################################################################

# rows are patient IDs and columns are features (CpGs)
clinical_data_df = pd.read_csv(infile_2)
clinical_data_df = clinical_data_df.set_index("PD_ID")
beta_matrix_df = pd.read_csv(infile_1,index_col=0).T

# align dataframes by shared samples
shared_samples = clinical_data_df.index.intersection(beta_matrix_df.index)
print(f"Shared samples: {len(shared_samples)}")
clinical_data_df = clinical_data_df.loc[shared_samples]
beta_matrix_df = beta_matrix_df.loc[shared_samples]
clinical_data_df.head()

# create Survival Object
y = create_surv(clinical_data_df, time="RFI", event="RFIbin")

# 3. Preprocess Feature Data (X)
X_preprocessed = preprocess(
    beta_matrix=beta_matrix_df,
    filter_method="variance", # or "unicox", or None
    top_n=200000, # Adjust as needed
    to_mval=False
)
# If using 'unicox' filter, you would pass clinical_data_df to preprocess,
# and inside preprocess, you'd pass clinical_data_df to unicox_filter.
# If you adapt unicox_filter to take 'y' directly, you can pass 'y' here instead.


# 4. Train Cox LASSO Model (using train_cox_lasso - GridSearchCV for alpha)
best_lasso_model, best_alpha, lasso_cv_results, lasso_scaler = train_cox_lasso(
    X=X_preprocessed,
    y=y,
    alphas=np.logspace(-4, 1, 10), # Define alpha grid
    scale=True,
    cv=5,
    scoring="neg_log_loss"
)

print(f"Best LASSO model: {best_lasso_model}")
print(f"Best Alpha: {best_alpha}")


# 5. Or, Train Cox LASSO using the more general train_model function (GridSearchCV for alpha)
cox_lasso_model = CoxnetSurvivalAnalysis(l1_ratio=1.0, fit_baseline_model=True) # Create model instance
param_grid_lasso = {
    'alphas': np.logspace(-4, 1, 10) # Define the hyperparameter grid for alpha
}

best_model_gen, best_params_gen, cv_results_gen, scaler_gen = train_model(
    model=cox_lasso_model, # Pass the model instance
    X=X_preprocessed,
    y=y,
    param_grid=param_grid_lasso, # Pass the hyperparameter grid
    scale=True,
    cv=5,
    scoring="neg_log_loss"
)

print(f"Best model (general train_model): {best_model_gen}")
print(f"Best Hyperparameters (general train_model): {best_params_gen}")


# 6. Evaluation (Concordance Index - C-index)
from sksurv.metrics import concordance_index_censored

if lasso_scaler: # Use scaler if scaling was applied during training
    X_scaled = lasso_scaler.transform(X_preprocessed)
else:
    X_scaled = X_preprocessed

c_index = concordance_index_censored(
    event_indicator=y['event'],
    event_time=y['time'],
    estimate=best_lasso_model.predict(X_scaled) # Or best_model_gen.predict(X_scaled) if using train_model
)
print(f"C-index on training data: {c_index[0]:.4f}")

# 7. (Optional) Prediction on new data
# If you have new data (X_new), you would preprocess it in the same way as your training data
# (using the SAME scaler if you used scaling during training!) and then use:
# predictions_new = best_lasso_model.predict(X_scaler.transform(X_new)) # If scaled
# or
# predictions_new = best_lasso_model.predict(X_new) # If not scaled
