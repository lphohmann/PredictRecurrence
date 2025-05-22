################################################################################
# Script: Fitting a penalized Cox model: Elastic Net
# apporach from: https://scikit-survival.readthedocs.io/en/stable/user_guide/coxnet.html
# Author: Lennart Hohmann /usr/bin/env python
# Date: 22.05.2025
################################################################################

# Standard library imports
import os
import sys
import time
import importlib
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.exceptions import FitFailedWarning

#sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
sys.path.append("C:\\Users\\lhohmann\\PredictRecurrence")
sys.path.append("C:\\Users\\lhohmann\\PredictRecurrence\\src")
#print("sys.path =", sys.path)
import src.utils
importlib.reload(src.utils)
from src.utils import beta2m, variance_filter, custom_scorer

# set wd
#os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))
os.chdir(os.path.expanduser("C:\\Users\\lhohmann\\PredictRecurrence"))
os.makedirs("output", exist_ok=True)

start_time = time.time()  # Record start time

print(f"Script started at: {time.ctime(start_time)}")

################################################################################
# load data
################################################################################

# input paths
infile_0 = r"./data/train/train_subcohorts/TNBC_train_ids.csv" 
infile_1 = r"./data/train/train_methylation_adjusted.csv"
infile_2 = r"./data/train/train_clinical.csv"

# output paths
outfile_model = r"output/best_cox_model.pkl"
outfile_coefs = r"output/non_zero_coefs.csv"
outfile_cv = r"output/cv_results.csv"
outfile_plot_model = r"output/best_model.png"
outfile_plot_cv = r"output/nested_cv_concordance.png"
outfile_plot_coef_dist = r"output/nonzero_coef_distribution.png"

################################################################################
# format data
################################################################################

# sample training set
train_ids = pd.read_csv(infile_0, header=None).iloc[:, 0].tolist()

# load clin data
clinical_data = pd.read_csv(infile_2)
clinical_data = clinical_data.set_index("Sample")
clinical_data = clinical_data.loc[train_ids]

# load beta values
beta_matrix = pd.read_csv(infile_1,index_col=0).T
# align dataframes
beta_matrix = beta_matrix.loc[train_ids]

################################################################################
# preprocessing steps
################################################################################

# 1. Convert beta values to M-values with a threshold 
mval_matrix = beta2m(beta_matrix,beta_threshold=0.001)

# 2. Apply variance filtering to retain top N most variable CpGs
mval_matrix = variance_filter(mval_matrix, top_n=200) #200000

################################################################################
# create Survival Object
################################################################################

y = Surv.from_dataframe("RFi_event", "RFi_years", clinical_data)
X = mval_matrix

################################################################################
# Finding the optimal alpha using cross-validation
# Step 1: Estimate alpha path from initial model
################################################################################
# Fit a Coxnet model to extract reasonable alpha values for grid search
# Note: Only need one value of l1_ratio here to get the alpha path
initial_pipe = make_pipeline(
    StandardScaler(),
    CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, n_alphas=30, max_iter=1000)
)

# Suppress convergence warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FitFailedWarning)

# Fit on full data just to estimate alphas (NOT for model evaluation!)
initial_pipe.fit(X, y)
estimated_alphas = initial_pipe.named_steps["coxnetsurvivalanalysis"].alphas_

################################################################################
# Step 2: Set up full parameter grid for tuning
################################################################################

# Define l1_ratios and use alpha values from the initial fit
param_grid = {
    "coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]#,
    #"coxnetsurvivalanalysis__l1_ratio": [0.1, 0.5, 0.9]  # Elastic Net mixing
}

################################################################################
# Step 3: Set up nested cross-validation
################################################################################

# Outer CV for performance estimation
outer_cv = KFold(n_splits=2, shuffle=True, random_state=42) #10

# Inner CV for hyperparameter tuning
inner_cv = KFold(n_splits=2, shuffle=True, random_state=1) #5

# Define the model and wrap in GridSearchCV (for the inner loop)
inner_model = GridSearchCV(
    estimator=make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=0.9)),
    param_grid=param_grid,
    cv=inner_cv,
    error_score=0.5,
    n_jobs=-1
)

################################################################################
# Step 4: Run nested CV to evaluate performance of best inner model
################################################################################
from sklearn.metrics import make_scorer
def dummy_scorer(estimator, X, y):
    print("Dummy scorer called!")
    return 0.5
dummy_custom_scorer = make_scorer(dummy_scorer, greater_is_better=True)

nested_scores = cross_val_score(inner_model, X, y, cv=outer_cv, scoring=dummy_custom_scorer)



# Uses the inner model (which includes its own CV) for model selection in each fold#
#nested_scores = cross_val_score(inner_model, X, y, cv=outer_cv, scoring=custom_scorer)


print(f"Nested CV Concordance Index: {np.mean(nested_scores):.3f} ± {np.std(nested_scores):.3f}")

################################################################################
# Visualize nested CV performance across folds
################################################################################

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(nested_scores) + 1), nested_scores, marker="o")
plt.axhline(np.mean(nested_scores), color="r", linestyle="--", label="Mean CI")
plt.title("Nested CV Concordance Index per Fold")
plt.xlabel("Outer CV Fold")
plt.ylabel("Concordance Index")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(outfile_plot_cv, dpi=300)

################################################################################
# Best model (refit on full data after nested CV)
################################################################################

# Refit inner model on full data to extract best model
inner_model.fit(X, y)
best_model = inner_model.best_estimator_.named_steps["coxnetsurvivalanalysis"]

# Extract non-zero coefficients
best_coefs = pd.DataFrame(best_model.coef_, index=X.columns, columns=["coefficient"])
non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
print(f"Number of non-zero coefficients: {non_zero}")

best_alpha = best_model.alphas_[0]
print(f"Best alpha selected: {best_alpha:.6f}")

non_zero_coefs = best_coefs.query("coefficient != 0")
coef_order = non_zero_coefs.abs().sort_values("coefficient").index

# Plot non-zero coefficients
_, ax = plt.subplots(figsize=(6, 8))
non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
ax.set_xlabel("coefficient")
ax.grid(True)
plt.savefig(outfile_plot_model, dpi=300, bbox_inches="tight")

# Histogram of non-zero coefficient magnitudes
plt.figure(figsize=(6, 4))
non_zero_coefs["coefficient"].abs().hist(bins=30)
plt.xlabel("Absolute coefficient value")
plt.ylabel("Frequency")
plt.title("Distribution of Non-zero Coefficients")
plt.grid(True)
plt.tight_layout()
plt.savefig(outfile_plot_coef_dist, dpi=300)

################################################################################
# Save outputs: Best Model, Coefficients, CV Results
################################################################################

# 1. Save best trained model
joblib.dump(inner_model.best_estimator_, outfile_model)
print(f"Saved best model to: {outfile_model}")

# 2. Save non-zero coefficients
non_zero_coefs.to_csv(outfile_coefs)
print(f"Saved non-zero coefficients to: {outfile_coefs}")

# 3. Save full inner CV results (from GridSearchCV)
cv_results = pd.DataFrame(inner_model.cv_results_)
cv_results.to_csv(outfile_cv, index=False)
print(f"Saved CV results to: {outfile_cv}")

end_time = time.time()
print(f"Script ended at: {time.ctime(end_time)}")
print(f"Script executed in {end_time - start_time:.2f} seconds.")