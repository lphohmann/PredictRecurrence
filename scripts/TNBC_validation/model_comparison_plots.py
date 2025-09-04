#!/usr/bin/env python

################################################################################
# Script: Compare model performances across multiple types
# Author: Lennart Hohmann
################################################################################

################################################################################
# IMPORTS
################################################################################

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

# Add project src directory to path for imports (adjust as needed)
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
from src.utils import log

# Set working directory
os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))

################################################################################
# INPUT FILES & PARAMS
################################################################################

# Dictionary of model names -> saved performance files
MODELS = {
    "RSF": "./output/RSF/model_performances_rsf.joblib",
    "CoxPH": "./output/CoxPH/model_performances_coxph.joblib",
    "DeepSurv": "./output/DeepSurv/model_performances_deepsurv.joblib"
}

EVAL_TIME_GRID = np.linspace(0, 10, 100)  # adjust based on dataset

################################################################################
# OUTPUT FILES
################################################################################

OUTPUT_DIR = "./output/Model_validation/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTFILE_AUC = os.path.join(OUTPUT_DIR, "auc_comparison.png")
OUTFILE_BRIER = os.path.join(OUTPUT_DIR, "brier_comparison.png")

################################################################################
# FUNCTIONS
################################################################################

def plot_auc_curves_multi(performances_dict, time_grid, outfile):
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 6))

    for model_name, performance in performances_dict.items():
        mean_auc_curve = np.mean([p["auc"] for p in performance], axis=0)
        plt.plot(time_grid, mean_auc_curve, lw=2, label=f'{model_name} Mean AUC')

    plt.title("Time-dependent AUC(t) Comparison")
    plt.xlabel("Time")
    plt.ylabel("AUC(t)")
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    log(f"Saved AUC comparison plot to {outfile}")


def plot_brier_scores_multi(performances_dict, time_grid, outfile):
    plt.style.use('seaborn-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios':[3,1]})

    colors = plt.cm.tab10(np.arange(len(performances_dict)))

    # Time-dependent Brier scores
    for i, (model_name, perf) in enumerate(performances_dict.items()):
        mean_brier = np.mean([p["brier_t"] for p in perf], axis=0)
        ax1.plot(time_grid, mean_brier, lw=2, color=colors[i], label=f'{model_name} Mean Brier')
    
    ax1.set_title("Time-dependent Brier Score Comparison")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Brier Score")
    ax1.set_ylim(0, 0.5)
    ax1.legend(loc='upper right')

    # IBS bar chart
    model_names = list(performances_dict.keys())
    ibs_values = [np.mean([p["ibs"] for p in performances_dict[m]]) for m in model_names]
    bars = ax2.bar(model_names, ibs_values, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_title("Integrated Brier Score (IBS) per Model")
    ax2.set_ylabel("IBS")
    ax2.set_ylim(0, max(ibs_values) * 1.1)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    log(f"Saved Brier comparison plot to {outfile}")

################################################################################
# MAIN CODE
################################################################################

# Load all model performances
performances = {name: load(file) for name, file in MODELS.items()}

log("Generating comparison plots.")
plot_auc_curves_multi(performances, EVAL_TIME_GRID, OUTFILE_AUC)
plot_brier_scores_multi(performances, EVAL_TIME_GRID, OUTFILE_BRIER)
