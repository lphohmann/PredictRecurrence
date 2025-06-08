#!/usr/bin/env python

################################################################################
# Script: Fitting a penalized Cox model: Elastic Net
# apporach from: https://scikit-survival.readthedocs.io/en/stable/user_guide/coxnet.html
# Author: Lennart Hohmann #/usr/bin/env python
# Date: 22.05.2025
################################################################################

################################################################################
# SET UP
################################################################################

import os
import sys
import time
import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score, brier_score

sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
import src.utils
importlib.reload(src.utils)
from src.utils import beta2m, variance_filter

# set wd
os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))
os.makedirs("output", exist_ok=True)

start_time = time.time()  # Record start time

print(f"Script started at: {time.ctime(start_time)}",flush=True)

################################################################################
# PARAMS
################################################################################

top_n_cpgs = 100000 # has to match n used in training

################################################################################
# SET FILE PATHS
################################################################################

# input paths
infile_0 = r"./data/train/train_subcohorts/TNBC_train_ids.csv" # subcohort ids
infile_1 = r"./data/train/train_methylation_adjusted.csv"
infile_2 = r"./data/train/train_clinical.csv"
infile_3 = r"./output/CoxNet/outer_cv_models.pkl"

# output paths
outfile_brier_plot = r"./output/CoxNet/brier_scores.png"
outfile_auc_plot = r"./output/CoxNet/auc_curves.png"
outfile_perf_csv = r"./output/CoxNet/performance_summary.csv"
outfile_best_model = r"./output/CoxNet/best_model.pkl"

################################################################################
# LOAD AND SUBSET TRAINING DATA
################################################################################

# load outer models
outer_models = joblib.load(infile_3)

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
# PREPROCESSING
################################################################################

# 1. Convert beta values to M-values with a threshold 
mval_matrix = beta2m(beta_matrix,beta_threshold=0.001)

# 2. Apply variance filtering to retain top N most variable CpGs
mval_matrix = variance_filter(mval_matrix, top_n=top_n_cpgs) #200,000

################################################################################
# CREATE SURVIVAL OBJECT
################################################################################

y = Surv.from_dataframe("RFi_event", "RFi_years", clinical_data)
X = mval_matrix

################################################################################
# Define consistent evaluation time grid based on test sets across outer folds
################################################################################

max_test_times = []
min_test_times = []

# Loop through test indices in outer folds
for entry in outer_models:
    if entry["model"] is None:
        continue  # skip failed folds
    test_idx = entry["test_idx"]
    y_test_fold = y[test_idx]
    max_test_times.append(y_test_fold["RFi_years"].max())
    min_test_times.append(y_test_fold["RFi_years"].min())

# Safe time grid boundaries
global_min_time = max(min_test_times)  # max of mins: ensures coverage
global_max_time = min(max_test_times)  # min of maxes: avoids extrapolation

# Round to nearest 0.5 and create grid
start = np.ceil(global_min_time * 2) / 2  
end = np.floor(global_max_time * 2) / 2
time_grid = np.arange(start, end + 0.1, 0.5)  # add epsilon to include end

print("Consistent evaluation time grid across folds:",flush=True)
print(time_grid,flush=True)

################################################################################
# print outer fold model info to output
################################################################################

# evalulation
performance = []
#entry = outer_models[0]

for i, entry in enumerate(outer_models):
    print(f"\n--- Fold {i} ---")
    print(f"Fold number: {entry.get('fold')}")
    print(f"Model type: {type(entry.get('model'))}")
    print(f"Train indices: {len(entry.get('train_idx', []))} samples")
    print(f"Test indices: {len(entry.get('test_idx', []))} samples")
    
    cv_results = entry.get("cv_results")
    if cv_results:
        print(f"CV results keys: {list(cv_results.keys())[:5]}...")  # show a few keys
        print(f"Mean test score (first 3): {cv_results['mean_test_score'][:3]}")
    
    error = entry.get("error")
    if error:
        print(f"Error: {error}")
    else:
        print("Error: None")

################################################################################
# Assess performance of each model across folds
################################################################################

#entry = outer_models[3]

for entry in outer_models:

    print(f"current outer cv fold model: {entry['fold']}", flush=True)

    if entry["model"] is None:
        print(f"skipping: {entry['fold']}", flush=True)
        continue  # skip failed folds

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

    # Compute AUC(t)
    auc, mean_auc = cumulative_dynamic_auc(
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
    
#print("eval done")

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
plt.savefig(outfile_brier_plot, dpi=300, bbox_inches='tight')
plt.close()

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
plt.savefig(outfile_auc_plot, dpi=300, bbox_inches='tight')
plt.close()


# Save summary table (AUC, IBS per fold)
df_perf = pd.DataFrame({
    "fold": [p["fold"] for p in performance],
    "mean_auc": [p["mean_auc"] for p in performance],
    "ibs": [p["ibs"] for p in performance]
})
df_perf.to_csv(outfile_perf_csv, index=False)

################################################################################
# Select Best Model Based on Lowest IBS or Highest Mean AUC
################################################################################

# Strategy 1: Select by lowest Integrated Brier Score (IBS)
best_by_ibs = min(performance, key=lambda p: p["ibs"])

# Strategy 2: Select by highest mean AUC(t)
best_by_auc = max(performance, key=lambda p: p["mean_auc"])

# Report
print("\nBest model by lowest IBS:")
print(f"  Fold: {best_by_ibs['fold']}")
print(f"  IBS: {best_by_ibs['ibs']:.4f}")
print(f"  Mean AUC(t): {best_by_ibs['mean_auc']:.4f}")

print("\nBest model by highest mean AUC(t):")
print(f"  Fold: {best_by_auc['fold']}")
print(f"  Mean AUC(t): {best_by_auc['mean_auc']:.4f}")
print(f"  IBS: {best_by_auc['ibs']:.4f}")

# Save best model to file (choose which strategy)
best_model = best_by_auc["model"] # best_by_ibs["model"]

joblib.dump(best_model, outfile_best_model)
print(f"\nBest model saved to: {outfile_best_model}")
