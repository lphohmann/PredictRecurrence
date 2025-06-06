
################################################################################
# Script: Fitting a penalized Cox model: Elastic Net
# apporach from: https://scikit-survival.readthedocs.io/en/stable/user_guide/coxnet.html
# Author: Lennart Hohmann #/usr/bin/env python
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
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.exceptions import FitFailedWarning
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score, brier_score

sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
import src.utils
importlib.reload(src.utils)
from src.utils import beta2m, variance_filter, cindex_scorer_sksurv

# set wd
os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))
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
mval_matrix = variance_filter(mval_matrix, top_n=100) #200,000

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
    CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.1, n_alphas=30, max_iter=1000)
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
print(estimated_alphas)
# Define l1_ratios and use alpha values from the initial fit
param_grid = {
    "coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]#,
    #"coxnetsurvivalanalysis__l1_ratio": [0.1, 0.5, 0.9]  # Elastic Net mixing
}

################################################################################
# Step 3: Set up nested cross-validation
################################################################################

# Outer CV for performance estimation
outer_cv = KFold(n_splits=3, shuffle=True, random_state=21) #10

# Inner CV for hyperparameter tuning
inner_cv = KFold(n_splits=2, shuffle=True, random_state=12) #5

# Define the model and wrap in GridSearchCV (for the inner loop)
inner_model = GridSearchCV(
    estimator=make_pipeline(CoxnetSurvivalAnalysis(l1_ratio=0.9, max_iter=100000)),
    param_grid=param_grid,
    cv=inner_cv,
    scoring=cindex_scorer_sksurv, # use this or another scorer
    error_score=0.5,
    n_jobs=-1
)

################################################################################
# Step 4: Run nested CV manually to evaluate performance
################################################################################
outer_models = []

for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
    print(f"current outer cv fold: {fold_num}")    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    try:
        inner_model.fit(X_train, y_train)
        best_model = inner_model.best_estimator_
    
        # Refit best model with fit_baseline_model=True for later eval
        best_alpha = best_model.named_steps["coxnetsurvivalanalysis"].alphas_[0]
        best_l1_ratio = best_model.named_steps["coxnetsurvivalanalysis"].l1_ratio
        refit_model = make_pipeline(
            CoxnetSurvivalAnalysis(
                alphas=[best_alpha],
                l1_ratio=best_l1_ratio,
                fit_baseline_model=True,
                max_iter=100000
            )
        )
        refit_model.fit(X_train, y_train)

        outer_models.append({
            "fold": fold_num,
            "model": refit_model,  # Save the refitted model
            "test_idx": test_idx,
            "train_idx": train_idx,
            "cv_results": inner_model.cv_results_,
            "error": None
        })

    except ArithmeticError as e:
        print(f"Skipping fold {fold_num} due to numerical error: {e}")
        outer_models.append({
            "fold": fold_num,
            "model": None,
            "test_idx": test_idx,
            "train_idx": train_idx,
            "cv_results": None,
            "error": str(e)
        })

################################################################################
# define time grid for evaluation
################################################################################

# Define evaluation times
max_test_times = []
min_test_times = []
for train_idx, test_idx in outer_cv.split(X):
    y_test_fold = y[test_idx]
    max_test_times.append(y_test_fold['RFi_years'].max())
    min_test_times.append(y_test_fold['RFi_years'].min())
global_min_time = max(min_test_times)  # max of mins ensures all test sets have that minimum time
global_max_time = min(max_test_times)  # min of maxes ensures all test sets have that max time
# Create time grid with 10 points between global_min_time and global_max_time
# Round to nearest 0.5
start = np.ceil(global_min_time * 2) / 2  
end = np.floor(global_max_time * 2) / 2
time_grid = np.arange(start, end + 0.1, 0.5)  # add small epsilon to include 'end'
print("Consistent time grid:")
print(time_grid)

################################################################################
# Assess performance of each model across folds
################################################################################

# evalulation
performance = []
#entry = outer_models[0]

for entry in outer_models:
    if entry["model"] is None:
        continue  # skip failed folds
    model = entry["model"]
    test_idx = entry["test_idx"]
    train_idx = entry["train_idx"]

    X_test = X.iloc[test_idx]
    X_train = X.iloc[train_idx]
    y_test = y[test_idx]
    y_train = y[train_idx]

    # Compute AUC(t)
    auc, mean_auc = cumulative_dynamic_auc( # this stuff reutnr not an array for auc which iswrong
        y_train, y_test,
        model.predict(X_test),
        times=time_grid
    )

    # Compute Brier score and integrated Brier score
    surv_funcs = model.predict_survival_function(X_test)
    preds = np.row_stack([
        [fn(t) for t in time_grid] for fn in surv_funcs
    ])

    brier_scores = brier_score(y_train, y_test, preds, time_grid)[1]
    ibs = integrated_brier_score(y_train, y_test, preds, time_grid)

    performance.append({
        "fold": entry["fold"],
        "model": model,
        "auc": auc,      
        "mean_auc": mean_auc,     
        "brier_t": brier_scores,
        "ibs": ibs
    })

################################################################################
# Visualize Brier score for each best model (1 per outer fold)
################################################################################

# Extract data from performance
folds = [p["fold"] for p in performance]
brier_array = np.array([p["brier_t"] for p in performance])  # (n_folds, n_times)
ibs_array = np.array([p["ibs"] for p in performance])        # (n_folds,)

plt.style.use('seaborn-whitegrid')

fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

# 1) Plot Brier Score(t) curves per fold + mean
for i, brier in enumerate(brier_array):
    axs[0].plot(time_grid, brier, label=f'Fold {folds[i]}', alpha=0.6)
axs[0].plot(time_grid, brier_array.mean(axis=0), color='black', lw=3, label='Mean Brier Score(t)')

axs[0].set_title("Time-dependent Brier Score", fontsize=16, fontweight='bold')
axs[0].set_xlabel("Time", fontsize=14)
axs[0].set_ylabel("Brier Score(t)", fontsize=14)
axs[0].legend(title='Folds', loc='upper right', fontsize=10)
axs[0].set_ylim(0, 0.5)
axs[0].grid(True, linestyle='--', alpha=0.7)

# 2) Plot Integrated Brier Score (IBS) per fold (bar plot)
bar_colors = plt.cm.Paired(np.linspace(0, 1, len(folds)))
bars = axs[1].bar(folds, ibs_array, color=bar_colors, edgecolor='black', alpha=0.85)

axs[1].set_title("Integrated Brier Score (IBS) per Fold", fontsize=16, fontweight='bold')
axs[1].set_xlabel("Fold", fontsize=14)
axs[1].set_ylabel("IBS", fontsize=14)
axs[1].set_ylim(0, max(ibs_array)*1.15)
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of each bar for clarity
for bar in bars:
    height = bar.get_height()
    axs[1].annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()


################################################################################
# Visualize AUC
################################################################################

# Collect all auc curves from folds into an array (shape: n_folds x n_times)
auc_curves = np.array([p["auc"] for p in performance])

# time_grid should be the same for all folds
# if needed, verify length matches auc curve length:
assert all(len(p["auc"]) == len(time_grid) for p in performance), "Mismatch in auc curve lengths!"

# Plotting setup
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10, 6))

# Plot each fold's AUC(t) curve
for i, p in enumerate(performance):
    plt.plot(time_grid, p["auc"], label=f'Fold {p["fold"]}', alpha=0.5)

# Plot mean AUC(t) across folds
mean_auc_curve = auc_curves.mean(axis=0)
plt.plot(time_grid, mean_auc_curve, color='black', linewidth=2.5, label='Mean AUC(t)')

# Labels and legend
plt.title("Time-dependent AUC(t) per Fold")
plt.xlabel("Time")
plt.ylabel("AUC(t)")
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

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