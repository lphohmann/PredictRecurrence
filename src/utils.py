#!/usr/bin/env python
# Script: General functions
# Author: Lennart Hohmann

# ==============================================================================
# IMPORTS
# ==============================================================================

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests
import warnings
from sklearn.exceptions import ConvergenceWarning

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def log(msg):
    print(f"\n=== {msg} ===\n", flush=True)

# ==============================================================================

def beta2m(beta, beta_threshold=1e-3):
    """
    Convert beta-values to M-values safely.

    Args:
        beta (float, array-like, or pd.DataFrame): Beta-values.
        beta_threshold (float): Lower and upper bound to avoid logit instability. 
                                Clips beta values to [beta_threshold, 1 - beta_threshold].

    Returns:
        Same shape as input: M-values.
    """
    beta = np.clip(beta, beta_threshold, 1 - beta_threshold)
    return np.log2(beta / (1 - beta))

# ==============================================================================

def m2beta(m):
    """
    Convert M-values to beta-values.

    Args:
        m (float, array-like, or pd.DataFrame): M-values.

    Returns:
        Same shape as input: Beta-values.
    """
    return 2**m / (2**m + 1)

# ==============================================================================

def variance_filter(df, min_variance=None, top_n=None):
    """
    Filter features (CpGs) based on variance. Extects CpGs as columns and patients as rows
    Args:
        X (pd.DataFrame): Feature matrix (samples x features).
        top_n (int, optional): Number of top features to select based on variance.
        min_variance (float, optional): Minimum variance threshold to retain features.
    Returns:
        pd.DataFrame: Filtered feature matrix.
    """
    variances = df.var(axis=0)
    
    if min_variance is not None:
        # Filter features with variance above threshold
        selected_features = variances[variances >= min_variance].index
    elif top_n is not None:
        # Select top_n features by variance
        selected_features = variances.sort_values(ascending=False).head(top_n).index
    else:
        raise ValueError("Either min_variance or top_n must be specified")
    
    return df.loc[:, selected_features]

# ==============================================================================

def load_training_data(train_ids, beta_path, clinical_path):
    """
    Loads and aligns clinical and beta methylation data using provided sample IDs.

    Args:
        train_ids (list): Sample IDs to subset data.
        beta_path (str): File path to methylation beta matrix.
        clinical_path (str): File path to clinical data.

    Returns:
        tuple: (beta_matrix, clinical_data) aligned to the same samples.
    """
    # load clin data
    clinical_data = pd.read_csv(clinical_path)
    clinical_data = clinical_data.set_index("Sample")
    clinical_data = clinical_data.loc[train_ids]

    # load beta values
    beta_matrix = pd.read_csv(beta_path,index_col=0).T

    # align dataframes
    beta_matrix = beta_matrix.loc[train_ids]

    return beta_matrix, clinical_data

# ==============================================================================

def preprocess_data(beta_matrix, top_n_cpgs):
    """
    Converts beta values to M-values and retains most variable CpGs.

    Args:
        beta_matrix (DataFrame): Raw beta methylation matrix.
        top_n_cpgs (int): Number of most variable CpGs to retain.

    Returns:
        DataFrame: Preprocessed M-values matrix.
    """

    print(f"\n=== Preprocessing: Converting to M-values and selecting top {top_n_cpgs} most variable CpGs ===\n", flush=True)

    # Convert beta values to M-values with a threshold 
    mvals = beta2m(beta_matrix, beta_threshold=0.001)
    # Apply variance filtering to retain top N most variable CpGs
    mvals = variance_filter(mvals, top_n=top_n_cpgs)
    return mvals

# ==============================================================================

def run_univariate_cox_for_cpgs(mval_matrix: pd.DataFrame,
                                 clin_data: pd.DataFrame,
                                 time_col: str,
                                 event_col: str):
    """
    Run univariate Cox regression for each CpG site, using penalized likelihood
    to handle complete separation issues.

    Parameters:
    - mval_matrix: DataFrame [patients x CpGs] with M-values.
    - clin_data: DataFrame with clinical info (must contain time_col and event_col).
    - time_col: Name of the column with time to event.
    - event_col: Name of the column with event occurrence (1=event, 0=censored).

    Returns:
    - DataFrame with columns: CpG_ID, HR, CI_lower, CI_upper, pval, padj
    """
    results = []

    # Suppress lifelines warnings for individual CpGs that are handled by penalization
    # We will only print our custom warning if fitting truly fails.
    # You can comment this out if you want to see all warnings, but it will be verbose.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning) # Catch numpy runtime warnings too

        for cpg in mval_matrix.columns:
            df = pd.concat([clin_data[[time_col, event_col]].copy(), mval_matrix[[cpg]]], axis=1).dropna()
            df.columns = ["time", "event", "cpg_value"]

            # Check for constant or near-constant values *after* dropping NaNs
            # A more robust check for very low variance might be beneficial here,
            # but the penalizer should handle most cases.
            if df["cpg_value"].nunique() <= 1:
                results.append({
                    "CpG_ID": cpg,
                    "HR": np.nan,
                    "CI_lower": np.nan,
                    "CI_upper": np.nan,
                    "pval": np.nan
                })
                continue

            try:
                # --- CHANGE START ---
                # Add a penalizer to handle complete separation
                # A small value like 0.01 or 0.1 is often a good starting point.
                # This adds a Firth-type penalty.
                cph = CoxPHFitter(penalizer=0.01) # <--- ADDED PENALIZER
                # --- CHANGE END ---

                cph.fit(df, duration_col="time", event_col="event")
                summary = cph.summary.loc["cpg_value"]

                results.append({
                    "CpG_ID": cpg,
                    "HR": summary["exp(coef)"],
                    "CI_lower": summary["exp(coef) lower 95%"],
                    "CI_upper": summary["exp(coef) upper 95%"],
                    "pval": summary["p"]
                })
            except Exception as e:
                # Only print warning if it's a true fitting error not related to convergence/separation
                # that the penalizer should handle.
                print(f"Warning: Failed for {cpg}: {e}", flush=True)
                results.append({
                    "CpG_ID": cpg,
                    "HR": np.nan,
                    "CI_lower": np.nan,
                    "CI_upper": np.nan,
                    "pval": np.nan
                })

    df_results = pd.DataFrame(results)

    # Adjust p-values using Benjamini-Hochberg (FDR)
    pvals = df_results["pval"].values
    mask = ~pd.isna(pvals)
    padj = np.full_like(pvals, np.nan, dtype=np.float64)
    padj[mask] = multipletests(pvals[mask], method='fdr_bh')[1]
    df_results["padj"] = padj

    return df_results