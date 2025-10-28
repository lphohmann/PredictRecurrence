#!/usr/bin/env python

################################################################################
# Script: Compare performances across multiple approaches
# Author: Lennart Hohmann
################################################################################

################################################################################
# IMPORTS
################################################################################

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import argparse

# Add project src directory to path for imports (adjust as needed)
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
from src.utils import log

# Set working directory
os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))

################################################################################
# ARGUMENTS
################################################################################

parser = argparse.ArgumentParser(description="Compare model performances")
parser.add_argument(
    "--cohort",
    choices=["TNBC", "ERpHER2n", "All"]
)
parser.add_argument(
    "--data_type",
    choices=["Clinical", "Combined", "Methylation"]
)
args = parser.parse_args()

################################################################################
# INPUT FILES & PARAMS
################################################################################

cohort = args.cohort 
data_type = args.data_type

# Dictionary of model names -> saved performance files
MODELS = {
    f"CoxNet_{cohort}_{data_type}": f"./output/CoxNet/{cohort}/{data_type}/None/outer_cv_performance.pkl",
    f"RSF_{cohort}_{data_type}": f"./output/RSF/{cohort}/{data_type}/Unadjusted/outer_cv_performance.pkl",
    f"GBM_{cohort}_{data_type}": f"./output/GBM/{cohort}/{data_type}/Unadjusted/outer_cv_performance.pkl",
    }

if cohort == "TNBC":
    EVAL_TIME_GRID = np.arange(1.5, 5.1, 0.5)  # time points for metrics
else:
    EVAL_TIME_GRID = np.arange(1.5, 10.1, 0.5)  # time points for metrics

################################################################################
# OUTPUT FILES
################################################################################

OUTPUT_DIR = f"./output/Model_validation/Across_approaches/{data_type}/{cohort}/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTFILE_AUC = os.path.join(OUTPUT_DIR, f"auc_comparison_{cohort}_{data_type}.pdf")
OUTFILE_BRIER = os.path.join(OUTPUT_DIR, f"brier_comparison_{cohort}_{data_type}.pdf")

################################################################################
# FUNCTIONS
################################################################################

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def plot_auc_curves_multi(performances_dict, time_grid, outfile):
    plt.style.use('seaborn-whitegrid')
    # then override fonts
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['font.size'] = 20

    plt.figure(figsize=(10, 6))

    for model_name, performance in performances_dict.items():
        mean_auc_curve = np.mean([p["auc"] for p in performance], axis=0)
        plt.plot(time_grid, mean_auc_curve, lw=2, label=f'{model_name}')

    #plt.title(f"{cohort}")
    plt.xlabel("Time")
    plt.ylabel("mean AUC(t)")
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(outfile, format="pdf", dpi=300)
    plt.close()
    log(f"Saved AUC comparison plot to {outfile}")


def plot_brier_scores_multi(performances_dict, time_grid, outfile):
    plt.style.use('seaborn-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios':[3,1]})

    colors = plt.cm.tab10(np.arange(len(performances_dict)))

    # Time-dependent Brier scores
    for i, (model_name, perf) in enumerate(performances_dict.items()):
        mean_brier = np.mean([p["brier_t"] for p in perf], axis=0)
        ax1.plot(time_grid, mean_brier, lw=2, color=colors[i], label=f'{model_name}')
    
    #ax1.set_title("Time-dependent Brier Score Comparison")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Brier Score")
    ax1.set_ylim(0, 0.5)
    ax1.legend(loc='upper right')

    # IBS bar chart
    model_names = list(performances_dict.keys())
    ibs_values = [np.mean([p["ibs"] for p in performances_dict[m]]) for m in model_names]
    bars = ax2.bar(model_names, ibs_values, color=colors, edgecolor='black', alpha=0.85)
    #ax2.set_title("Integrated Brier Score (IBS) per Model")
    ax2.set_ylabel("IBS")
    ax2.set_ylim(0, max(ibs_values) * 1.1)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.3f}', ha='center')
    
    #plt.title(f"{cohort}")
    plt.tight_layout()
    plt.savefig(outfile, format="pdf", dpi=300)  # <-- save as PDF
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
