#!/usr/bin/env python
# Script: CpG set annotation functions
# Author: Lennart Hohmann

# ==============================================================================
# IMPORTS
# ==============================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def plot_cpg_correlation(data: pd.DataFrame, 
                         method: str = 'spearman', 
                         output_path: str = 'CpG_Correlation_Matrix.png') -> None:
    """
    Calculate and plot a correlation matrix for CpG data.
    
    Parameters:
    - data: pd.DataFrame
        Rows are CpGs, columns are patients.
    - method: str
        Correlation method: 'pearson' or 'spearman'.
    - output_path: str
        Path to save the heatmap image.
    """
    # Transpose so rows are patients, columns are CpGs
    data_t = data.T

    # Compute correlation matrix
    cor_matrix = data_t.corr(method=method)

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cor_matrix, 
                cmap='coolwarm', 
                center=0, 
                square=True,
                xticklabels=False, 
                yticklabels=False,
                cbar_kws={"label": f"{method.capitalize()} Correlation"},
                linewidths=0.5)

    plt.title("CpG Correlation Matrix")
    plt.tight_layout()

    # Save plot
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

# ==============================================================================
