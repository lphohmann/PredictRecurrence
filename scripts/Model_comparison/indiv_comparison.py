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
    "--model_type",
    choices=["CoxNet", "GBM", "RSF", "XGBoost"]
)
args = parser.parse_args()

################################################################################
# INPUT FILES & PARAMS
################################################################################

cohort = args.cohort 
model_type = args.model_type

# Dictionary of model names -> saved performance files
MODELS = {
    "Clinical": f"./output/{model_type}/{cohort}/Clinical/None/outer_cv_performance.pkl",
    "Clinical+Methylation": f"./output/{model_type}/{cohort}/Combined/Unadjusted/outer_cv_performance.pkl",
    #"C+M(adj)": f"./output/{model_type}/{cohort}/Combined/Adjusted/outer_cv_performance.pkl",
    "Methylation": f"./output/{model_type}/{cohort}/Methylation/Unadjusted/outer_cv_performance.pkl",
    #"M(adj)": f"./output/{model_type}/{cohort}/Methylation/Adjusted/outer_cv_performance.pkl"
    }

EVAL_TIME_GRID = np.arange(1.5, 5.1, 0.5)  # time points for metrics

################################################################################
# OUTPUT FILES
################################################################################

OUTPUT_DIR = f"./output/Model_validation/Within_approaches/{model_type}/{cohort}/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTFILE_AUC = os.path.join(OUTPUT_DIR, f"auc_comparison_{model_type}_{cohort}.pdf")
OUTFILE_BRIER = os.path.join(OUTPUT_DIR, f"brier_comparison_{model_type}_{cohort}.pdf")

################################################################################
# FUNCTIONS
################################################################################

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
def _time_vector_from_perf(p, fallback_time_grid):
    """Return a time vector for a performance entry.
    If the entry stores a time vector under common keys, use it.
    Otherwise infer a linear time vector spanning fallback_time_grid with the same length as the saved curve.
    """
    for key in ("time_grid", "times", "eval_times", "t_eval"):
        if key in p:
            return np.asarray(p[key], dtype=float)
    # infer from length of available curve
    if "auc" in p:
        n = len(p["auc"])
    elif "brier_t" in p:
        n = len(p["brier_t"])
    else:
        return np.asarray(fallback_time_grid, dtype=float)
    return np.linspace(np.min(fallback_time_grid), np.max(fallback_time_grid), n)


def plot_auc_curves_multi(performances_dict, time_grid, outfile):
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 18
    plt.rcParams['font.size'] = 20

    plt.figure(figsize=(10, 6))

    for model_name, performance in performances_dict.items():
        interp_curves = []
        for p in performance:
            if "auc" not in p:
                raise KeyError(f"Missing 'auc' in performance entry for {model_name}")

            y = np.asarray(p["auc"], dtype=float)
            t = _time_vector_from_perf(p, time_grid)

            # ensure times sorted
            if np.any(np.diff(t) < 0):
                order = np.argsort(t)
                t = t[order]
                y = y[order]

            # mask valid region for this entry
            valid_mask = time_grid <= np.max(t)
            if valid_mask.any():
                y_sub = np.interp(time_grid[valid_mask], t, y)
            else:
                y_sub = np.array([])

            y_full = np.full_like(time_grid, np.nan, dtype=float)
            y_full[valid_mask] = y_sub
            interp_curves.append(y_full)

        # mean ignoring NaNs so curves stop where data ends
        mean_auc_curve = np.nanmean(np.vstack(interp_curves), axis=0)
        plt.plot(time_grid, mean_auc_curve, lw=2, label=f'{model_name}')

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

    # Time-dependent Brier
    for i, (model_name, perf) in enumerate(performances_dict.items()):
        interp_curves = []
        for p in perf:
            if "brier_t" not in p:
                raise KeyError(f"Missing 'brier_t' in performance entry for {model_name}")

            y = np.asarray(p["brier_t"], dtype=float)
            t = _time_vector_from_perf(p, time_grid)

            if np.any(np.diff(t) < 0):
                order = np.argsort(t)
                t = t[order]
                y = y[order]

            valid_mask = time_grid <= np.max(t)
            if valid_mask.any():
                y_sub = np.interp(time_grid[valid_mask], t, y)
            else:
                y_sub = np.array([])

            y_full = np.full_like(time_grid, np.nan, dtype=float)
            y_full[valid_mask] = y_sub
            interp_curves.append(y_full)

        mean_brier = np.nanmean(np.vstack(interp_curves), axis=0)
        ax1.plot(time_grid, mean_brier, lw=2, color=colors[i], label=f'{model_name}')

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Brier Score")
    ax1.set_ylim(0, 0.5)
    ax1.legend(loc='upper right')

    # IBS bar chart
    model_names = list(performances_dict.keys())
    ibs_values = []
    for m in model_names:
        vals = [p["ibs"] for p in performances_dict[m] if "ibs" in p]
        if not vals:
            raise KeyError(f"No 'ibs' values found for model {m}")
        ibs_values.append(np.mean(vals))

    bars = ax2.bar(model_names, ibs_values, color=colors, edgecolor='black', alpha=0.85)
    #ax2.set_title("Integrated Brier Score (IBS) per Model")
    ax2.set_ylabel("IBS")
    ax2.set_ylim(0, max(ibs_values) * 1.1)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig(outfile, format="pdf", dpi=300)
    plt.close()
    log(f"Saved Brier comparison plot to {outfile}")

################################################################################
# MAIN CODE
################################################################################

# Load all model performances
performances = {name: load(file) for name, file in MODELS.items()}

log("Generating comparison plots.")
if model_type == "XGBoost":
    plot_auc_curves_multi(performances, EVAL_TIME_GRID, OUTFILE_AUC)
else:
    plot_auc_curves_multi(performances, EVAL_TIME_GRID, OUTFILE_AUC)
    plot_brier_scores_multi(performances, EVAL_TIME_GRID, OUTFILE_BRIER)
