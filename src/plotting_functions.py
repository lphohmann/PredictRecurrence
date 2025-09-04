#!/usr/bin/env python
# Script: Functions for Random Survival Forest pipeline
# Author: lennart hohmann

# ==============================================================================
# IMPORTS
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import os

# ==============================================================================
# FUNCTIONS
# ==============================================================================

# ==============================================================================

def plot_brier_scores(brier_array, ibs_array, folds, time_grid, outfile):
    """
    Plot time-dependent Brier scores for each fold and IBS per fold.
    """
    print("Plotting Brier scores...", flush=True)
    plt.style.use('seaborn-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    # Brier curves
    for i, brier in enumerate(brier_array):
        ax1.plot(time_grid, brier, label=f'Fold {folds[i]}', alpha=0.6)
    ax1.plot(time_grid, np.mean(brier_array, axis=0), color='black', lw=3, 
             label='Mean Brier Score')
    ax1.set_title("Time-dependent Brier Score")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Brier Score")
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 0.5)

    # IBS bar chart
    colors = plt.cm.Paired(np.linspace(0, 1, len(folds)))
    bars = ax2.bar(folds, ibs_array, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_title("Integrated Brier Score (IBS) per Fold")
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("IBS")
    ax2.set_ylim(0, max(ibs_array) * 1.1 if len(ibs_array) else 1)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.005, f'{yval:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Saved Brier plot to {outfile}", flush=True)

# ==============================================================================

def plot_auc_curves(performance, time_grid, outfile):
    """
    Plot time-dependent AUC(t) curves for all folds and their mean.
    """
    print("Plotting time-dependent AUC curves...", flush=True)
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 6))

    # Plot each fold's AUC curve
    for p in performance:
        plt.plot(time_grid, p["auc"], label=f'Fold {p["fold"]}', alpha=0.5)
    # Mean AUC curve
    mean_auc_curve = np.mean([p["auc"] for p in performance], axis=0)
    plt.plot(time_grid, mean_auc_curve, color='black', lw=2.5, label='Mean AUC')
    plt.title("Time-dependent AUC(t) per Fold")
    plt.xlabel("Time")
    plt.ylabel("AUC(t)")
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Saved AUC plot to {outfile}", flush=True)



# ==============================================================================

def plot_histogram(data: pd.Series, title: str, xlabel: str, outfile: str, bins='auto', xlim=None):
    """
    Plots a histogram of the given data.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(data.dropna(), bins=bins, kde=True, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of CpGs")
    if xlim:
        plt.xlim(xlim)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    log(f"Saved {title} histogram to {outfile}")

# ==============================================================================

def plot_pvalue_histograms(df_results: pd.DataFrame, output_dir: str):
    """
    Plots histograms for raw and adjusted p-values.
    """
    log("Generating p-value histograms...")
    # Raw p-values
    plot_histogram(df_results["pval"],
                   title="Distribution of Raw Univariate Cox P-values",
                   xlabel="P-value",
                   outfile=os.path.join(output_dir, "raw_pvalue_histogram.png"),
                   bins=50, xlim=(0, 1))

    # Adjusted p-values
    plot_histogram(df_results["padj"],
                   title="Distribution of Adjusted Univariate Cox P-values (FDR)",
                   xlabel="Adjusted P-value",
                   outfile=os.path.join(output_dir, "adjusted_pvalue_histogram.png"),
                   bins=50, xlim=(0, 1))
    log("P-value histograms generated.")

# ==============================================================================

def plot_hr_histogram(df_results: pd.DataFrame, output_dir: str):
    """
    Plots a histogram for Hazard Ratios.
    """
    log("Generating Hazard Ratio histogram...")
    plot_histogram(df_results["HR"],
                   title="Distribution of Univariate Cox Hazard Ratios",
                   xlabel="Hazard Ratio (HR)",
                   outfile=os.path.join(output_dir, "hr_histogram.png"),
                   bins=50) # Bins can be 'auto' or specified
    log("Hazard Ratio histogram generated.")

# ==============================================================================

def plot_coefficients_histogram(coefficients: pd.Series, title: str, xlabel: str, outfile: str, bins='auto'):
    """
    Plots a histogram of the given coefficients, focusing on non-zero values.
    """
    plt.figure(figsize=(10, 7))
    # Filter for non-zero coefficients for a more meaningful plot
    non_zero_coefs = coefficients[coefficients != 0].dropna()
    if non_zero_coefs.empty:
        log(f"No non-zero coefficients to plot for {title}.")
        plt.text(0.5, 0.5, "No non-zero coefficients", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    else:
        sns.histplot(non_zero_coefs, bins=bins, kde=True, color='purple', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of CpGs")
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    log(f"Saved {title} histogram to {outfile}")