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
from src.utils import log

# ==============================================================================
# FUNCTIONS
# ==============================================================================
def plot_brier_scores(performance, time_grid, outfile):
    """
    Plot time-dependent Brier scores for each fold and IBS per fold.
    Handles folds with different time coverage gracefully.
    
    Args:
        performance: List of dicts from evaluate_outer_models
        time_grid: Array of time points
        outfile: Path to save the plot
    """
    print("Plotting Brier scores...", flush=True)
    plt.style.use('seaborn-v0_8-whitegrid')  # Updated style name
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Extract Brier scores and IBS for valid folds
    valid_folds = []
    brier_curves = []
    ibs_values = []
    fold_numbers = []
    
    for p in performance:
        if p.get('brier_by_time') is not None and not np.isnan(p.get('ibs', np.nan)):
            # Extract Brier scores at each time point
            brier_at_times = [p['brier_by_time'][t] for t in time_grid]
            
            # Only include if this fold has at least some valid time points
            if not all(np.isnan(brier_at_times)):
                brier_curves.append(brier_at_times)
                ibs_values.append(p['ibs'])
                fold_numbers.append(p['fold'])
                valid_folds.append(p)
    
    if len(valid_folds) == 0:
        print("Warning: No valid folds to plot!", flush=True)
        return
    
    brier_array = np.array(brier_curves)
    
    # Plot Brier curves for each fold
    for i, fold_num in enumerate(fold_numbers):
        # Only plot non-NaN values
        valid_mask = ~np.isnan(brier_array[i])
        if valid_mask.any():
            ax1.plot(time_grid[valid_mask], brier_array[i][valid_mask], 
                    label=f'Fold {fold_num}', alpha=0.6, marker='o', markersize=4)
    
    # Calculate and plot mean Brier score (using nanmean to handle missing values)
    mean_brier = np.nanmean(brier_array, axis=0)
    valid_mean_mask = ~np.isnan(mean_brier)
    ax1.plot(time_grid[valid_mean_mask], mean_brier[valid_mean_mask], 
            color='black', lw=3, label='Mean Brier Score', marker='s', markersize=6)
    
    ax1.set_title("Time-dependent Brier Score", fontsize=14)
    ax1.set_xlabel("Time (years)", fontsize=12)
    ax1.set_ylabel("Brier Score", fontsize=12)
    ax1.legend(loc='best')
    ax1.set_ylim(0, 0.5)
    ax1.grid(True, alpha=0.3)
    
    # IBS bar chart
    colors = plt.cm.Paired(np.linspace(0, 1, len(fold_numbers)))
    bars = ax2.bar(fold_numbers, ibs_values, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_title("Integrated Brier Score (IBS) per Fold", fontsize=14)
    ax2.set_xlabel("Fold", fontsize=12)
    ax2.set_ylabel("IBS", fontsize=12)
    ax2.set_ylim(0, max(ibs_values) * 1.1 if ibs_values else 1)
    ax2.set_xticks(fold_numbers)
    
    # Add value labels on bars
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.005, 
                f'{yval:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Brier plot to {outfile}", flush=True)


def plot_auc_curves(performance, time_grid, outfile):
    """
    Plot time-dependent AUC(t) curves for all folds and their mean.
    Handles folds with different time coverage gracefully.
    
    Args:
        performance: List of dicts from evaluate_outer_models
        time_grid: Array of time points
        outfile: Path to save the plot
    """
    print("Plotting time-dependent AUC curves...", flush=True)
    plt.style.use('seaborn-v0_8-whitegrid')  # Updated style name
    plt.figure(figsize=(10, 6))
    
    # Extract AUC curves for valid folds
    valid_folds = []
    auc_curves = []
    fold_numbers = []
    
    for p in performance:
        if p.get('auc_by_time') is not None:
            # Extract AUC scores at each time point
            auc_at_times = [p['auc_by_time'][t] for t in time_grid]
            
            # Only include if this fold has at least some valid time points
            if not all(np.isnan(auc_at_times)):
                auc_curves.append(auc_at_times)
                fold_numbers.append(p['fold'])
                valid_folds.append(p)
    
    if len(valid_folds) == 0:
        print("Warning: No valid folds to plot!", flush=True)
        return
    
    auc_array = np.array(auc_curves)
    
    # Plot each fold's AUC curve
    for i, fold_num in enumerate(fold_numbers):
        # Only plot non-NaN values
        valid_mask = ~np.isnan(auc_array[i])
        if valid_mask.any():
            plt.plot(time_grid[valid_mask], auc_array[i][valid_mask], 
                    label=f'Fold {fold_num}', alpha=0.5, marker='o', markersize=4)
    
    # Calculate and plot mean AUC curve (using nanmean to handle missing values)
    mean_auc_curve = np.nanmean(auc_array, axis=0)
    valid_mean_mask = ~np.isnan(mean_auc_curve)
    plt.plot(time_grid[valid_mean_mask], mean_auc_curve[valid_mean_mask], 
            color='black', lw=2.5, label='Mean AUC', marker='s', markersize=6)
    
    plt.title("Time-dependent AUC(t) per Fold", fontsize=14)
    plt.xlabel("Time (years)", fontsize=12)
    plt.ylabel("AUC(t)", fontsize=12)
    plt.ylim(0.5, 1.0)  # AUC range
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved AUC plot to {outfile}", flush=True)

def plot_auc_with_sem(performance, time_grid, outfile):
    """
    Plot mean AUC curve with SEM error bands.
    
    Args:
        performance: List of dicts from evaluate_outer_models
        time_grid: Array of time points
        outfile: Path to save the plot
    """
    print("Plotting AUC with SEM bands...", flush=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    # Collect AUC values at each time point
    auc_at_each_time = {t: [] for t in time_grid}
    
    for p in performance:
        if p.get('auc_by_time') is not None:
            for t in time_grid:
                if not np.isnan(p['auc_by_time'][t]):
                    auc_at_each_time[t].append(p['auc_by_time'][t])
    
    # Calculate mean and SEM at each time point
    means = []
    sems = []
    valid_times = []
    n_folds = []
    
    for t in time_grid:
        values = auc_at_each_time[t]
        if len(values) > 0:
            means.append(np.mean(values))
            sems.append(np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0)
            valid_times.append(t)
            n_folds.append(len(values))
    
    means = np.array(means)
    sems = np.array(sems)
    valid_times = np.array(valid_times)
    
    # Plot mean with error band
    plt.plot(valid_times, means, color='blue', lw=2.5, label='Mean AUC')
    plt.fill_between(valid_times, means - sems, means + sems, 
                     alpha=0.3, color='blue', label='±1 SEM')
    
    # Add markers showing number of folds at each time point
    for i, (t, n) in enumerate(zip(valid_times, n_folds)):
        plt.text(t, means[i] + sems[i] + 0.02, f'n={n}', 
                ha='center', va='bottom', fontsize=8, color='gray')
    
    plt.title("Time-dependent AUC with Standard Error", fontsize=14)
    plt.xlabel("Time (years)", fontsize=12)
    plt.ylabel("AUC(t)", fontsize=12)
    plt.ylim(0.5, 1.0)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved AUC plot with SEM to {outfile}", flush=True)

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