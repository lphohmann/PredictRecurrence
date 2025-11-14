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
# INPUT FILES & PARAMS
################################################################################

# Dictionary of model names -> saved performance files
MODELS = {
    "Clinical": f"./output/CoxNet/TNBC/Clinical/None/outer_cv_performance.pkl",
    "Methylation": f"./output/CoxNet/TNBC/Methylation/Unadjusted/outer_cv_performance.pkl"
}

EVAL_TIME_GRID = np.arange(1, 5.1, 0.5)  # time points for metrics

################################################################################
# OUTPUT FILES
################################################################################

OUTPUT_DIR = f"./output/Model_validation/CoxNet_CvsM/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        plt.plot(time_grid, mean_auc_curve, lw=2, label=f'{model_name} Mean AUC')

    #plt.title("Time-dependent AUC(t) Comparison")
    plt.xlabel("Time")
    plt.ylabel("mean AUC(t)")
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(outfile, format="pdf", dpi=300)
    plt.close()
    log(f"Saved AUC comparison plot to {outfile}")


################################################################################
# MAIN CODE
################################################################################

if __name__ == "__main__":

    # Load all model performances
    performances = {name: load(file) for name, file in MODELS.items()}

    log("Generating comparison plots.")
    plot_auc_curves_multi(performances, EVAL_TIME_GRID, OUTFILE_AUC)
