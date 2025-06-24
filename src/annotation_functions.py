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
import mygene
import gseapy as gp
from matplotlib.backends.backend_pdf import PdfPages

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def plot_cpg_correlation(data: pd.DataFrame, 
                         output_path: str,
                         method: str = 'spearman') -> None:
    """
    Calculate and plot a clustered correlation matrix for CpG data.

    Parameters:
    - data: pd.DataFrame
        Rows are CpGs, columns are patients.
    - method: str
        Correlation method: 'pearson' or 'spearman'.
    - output_path: str
        Path to save the heatmap image.
    """

    # Compute correlation matrix (CpGs x CpGs)
    cor_matrix = data.corr(method=method)

    # Create clustered heatmap (hierarchical clustering of rows and columns)
    clustergrid = sns.clustermap(cor_matrix,
                                 cmap='coolwarm',
                                 center=0,
                                 figsize=(12, 10),
                                 linewidths=0.5,
                                 cbar_kws={"label": f"{method.capitalize()} Correlation"},
                                 xticklabels=True,
                                 yticklabels=True)

    plt.title("CpG Correlation Matrix", pad=120)

    # Save plot
    clustergrid.savefig(output_path, dpi=300)
    plt.close()


# ==============================================================================

def plot_beta_histograms(beta_matrix, cpg_list, outfile, cox_results_df=None, clinical_df=None):
    """
    Plot and save histograms of beta values for each CpG in a single multi-page PDF.
    
    Assumes beta_matrix has samples as rows and CpGs as columns.

    Parameters:
    - beta_matrix: pd.DataFrame with shape [samples x CpGs].
    - cpg_list: list of CpG names to plot.
    - outfile: path to save the PDF file.
    - cox_results_df: optional DataFrame with Cox regression results.
      Must contain: ['CpG_ID', 'HR', 'CI_lower', 'CI_upper', 'pval'].
    - clinical_df: optional DataFrame with clinical data.
      Must contain 'RFi_event' column with 1 (event) or 0 (no event).
    """
    with PdfPages(outfile) as pdf:
        for cpg in cpg_list:
            if cpg not in beta_matrix.columns:
                print(f"Warning: {cpg} not found in beta matrix.", flush=True)
                continue

            plt.figure(figsize=(6, 4))

            values = beta_matrix[cpg].dropna()

            if clinical_df is not None and 'RFi_event' in clinical_df.columns:
                # Align samples between beta_matrix and clinical_df
                common_samples = beta_matrix.index.intersection(clinical_df.index)
                cpg_vals = beta_matrix.loc[common_samples, cpg]
                events = clinical_df.loc[common_samples, 'RFi_event']

                vals_event = cpg_vals[events == 1]
                vals_noevent = cpg_vals[events == 0]

                plt.hist(vals_noevent.dropna(), bins=np.linspace(0, 1, 51) , color='grey', label='No Event', alpha=0.7)
                plt.hist(vals_event.dropna(), bins=np.linspace(0, 1, 51) , color='red', label='Event', alpha=0.7)
                plt.legend()
            else:
                # Default single histogram
                plt.hist(values, bins=np.linspace(0, 1, 51) , color='skyblue', edgecolor='black')

            # Title
            plt.title(f"Beta Value Distribution: {cpg}")

            # Add Cox regression stats if available
            if cox_results_df is not None:
                row = cox_results_df[cox_results_df['CpG_ID'] == cpg]
                if not row.empty:
                    hr = row['HR'].values[0]
                    ci_lower = row['CI_lower'].values[0]
                    ci_upper = row['CI_upper'].values[0]
                    padj = row['padj'].values[0]
                    subtitle = f"HR={hr:.3f} (CI: {ci_lower:.3f}-{ci_upper:.3f}), padj={padj:.3g}"
                    plt.suptitle(subtitle, y=0.92, fontsize=9)

            plt.xlabel("Beta Value")
            plt.ylabel("Frequency")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"Saved all beta histograms to: {outfile}", flush=True)



# ==============================================================================

def enrich_and_plot(cpg_anno: pd.DataFrame,
                    outfile: str,
                    top_n: int = 10) -> None:
    """
    Performs KEGG and GO enrichment using GSEApy and saves bar plots.

    Parameters:
    - cpg_anno: DF with annotation of CpGs of interest.
    - outfile: Output path to save the KEGG plot or combined plot.
    - top_n: Number of top terms to plot for each.
    """

    # 1. Extract gene names
    gene_names = set(cpg_anno.loc[cpg_anno["hasUCSCknownGeneOverlap"] == 1, "nameUCSCknownGeneOverlap"].to_list())
    print(f"Gene names ({len(gene_names)}): {sorted(gene_names)}", flush=True)

    # 2. Convert to Entrez IDs
    mg = mygene.MyGeneInfo()
    query = mg.querymany(gene_names, scopes='symbol', fields='entrezgene', species='human', as_dataframe=True)
    entrez_ids = query['entrezgene'].dropna().astype(int).astype(str).tolist()

    # 3. Run enrichment
    go_results = gp.enrichr(gene_list=entrez_ids, gene_sets='GO_Biological_Process_2025', organism='Human', cutoff=0.05)
    kegg_results = gp.enrichr(gene_list=entrez_ids, gene_sets='KEGG_2021_Human', organism='Human', cutoff=0.05)

    if go_results.results.empty or kegg_results.results.empty:
        print("No enrichment terms found for both or one of GO/KEGG. Exiting.")
        return

    # 4. Plot and save
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    gp.plot.barplot(kegg_results.results.head(top_n), title='KEGG Enrichment', ax=axs[0])
    gp.plot.barplot(go_results.results.head(top_n), title='GO BP Enrichment', ax=axs[1])
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Enrichment plots saved to: {outfile}", flush=True)

# ==============================================================================

from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests

def run_univariate_cox_for_cpgs(mval_matrix: pd.DataFrame,
                                 clin_data: pd.DataFrame,
                                 time_col: str,
                                 event_col: str):
    """
    Run univariate Cox regression for each CpG site.

    Parameters:
    - mval_matrix: DataFrame [patients x CpGs] with M-values.
    - clin_data: DataFrame with clinical info (must contain time_col and event_col).
    - time_col: Name of the column with time to event.
    - event_col: Name of the column with event occurrence (1=event, 0=censored).

    Returns:
    - DataFrame with columns: CpG_ID, HR, CI_lower, CI_upper, pval, padj
    """
    results = []

    for cpg in mval_matrix.columns:
        df = pd.concat([clin_data[[time_col, event_col]].copy(), mval_matrix[[cpg]]], axis=1).dropna()
        df.columns = ["time", "event", "cpg_value"]

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
            cph = CoxPHFitter()
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

