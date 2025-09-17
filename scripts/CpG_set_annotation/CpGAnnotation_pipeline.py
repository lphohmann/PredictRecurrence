#!/usr/bin/env python

################################################################################
# Script: CoxNet pipeline
# Author: Lennart Hohmann
################################################################################

################################################################################
# SET UP
################################################################################

import os
import sys
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# Import custom CoxNet functions
sys.path.append("/Users/le7524ho/PhD_Workspace/PredictRecurrence/src/")
from src.utils import (
    log,
    load_training_data,
    beta2m
)
from src.annotation_functions import (
    plot_cpg_correlation,
    enrich_and_plot,
    plot_beta_histograms,
    run_univariate_cox_for_cpgs
)

# Set working directory
os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))

################################################################################
# INPUT FILES
################################################################################

# Input files
infile_cpg_set = "./output/CoxLasso/TNBC/Unadjusted/Selected_model/selected_cpgs.txt"# ⚠️ ADAPT
infile_cpg_anno = "./data/raw/EPIC_probeAnnoObj.csv"

infile_train_ids = "./data/train/train_subcohorts/TNBC_train_ids.csv" # sample ids of training cohort
infile_betavalues = "./data/train/train_methylation_unadjusted.csv" # ⚠️ ADAPT
infile_clinical = "./data/train/train_clinical.csv"

################################################################################
# PARAMS
################################################################################

# Output directory and files
output_dir = "./output/CoxLasso/TNBC/Unadjusted/Selected_model/" # ⚠️ ADAPT
os.makedirs(output_dir, exist_ok=True)
outfile_correlation = os.path.join(output_dir, "correlation_matrix.png")
outfile_posbarplot = os.path.join(output_dir, "position_barplot.pdf")
outfile_enrich= os.path.join(output_dir, "enrichment_barplot.png")
outfile_betahist= os.path.join(output_dir, "beta_histograms.pdf")
outfile_heatmap = os.path.join(output_dir, "heatmap_selected_cpgs.pdf")

# log file
#path_logfile = os.path.join(output_dir, "_run.log")
#logfile = open(path_logfile, "w")
#sys.stdout = logfile
#sys.stderr = logfile

#sys.stdout = sys.__stdout__
#sys.stderr = sys.__stderr__

################################################################################
# load data
################################################################################

start_time = time.time()
log(f"Script started at: {time.ctime(start_time)}")

# Load and prepare data for correlation plot
train_ids = pd.read_csv(infile_train_ids, header=None).iloc[:, 0].tolist()
beta_matrix, clinical_data = load_training_data(train_ids, infile_betavalues, infile_clinical)
mval_matrix = beta2m(beta_matrix, beta_threshold=0.001)

# load cpgs set to annotate
with open(infile_cpg_set) as f:
    cpg_list = [line.strip() for line in f]

# load cpg annotation file
cpg_anno = pd.read_csv(infile_cpg_anno)
cpg_anno = cpg_anno[cpg_anno['illuminaID'].isin(cpg_list)]

#df=cpg_anno
#df.shape
#df.columns.tolist()
#df.head()
#df.dtypes
#df.isnull().sum()
#df.describe()
#df.describe(include='object')

################################################################################
# correlation matrix plot
################################################################################

plot_cpg_correlation(mval_matrix.loc[:,cpg_list], output_path = outfile_correlation,method = 'spearman')
#plot_cpg_correlation(beta_matrix.loc[:,cpg_list], output_path = outfile_correlation,method = 'spearman')
print(f"Correlation matrix saved to: {outfile_correlation}", flush=True)


################################################################################
# heatmap of selected CpGs
################################################################################

# Subset beta values for selected CpGs and training samples
heatmap_data = beta_matrix.loc[:, cpg_list]

# Transpose so that samples are columns
heatmap_data_T = heatmap_data.T  # now rows = CpGs, cols = samples

# Get clinical annotation for relapse
relapse_status = clinical_data.loc[heatmap_data.index, "RFi_event"]

# Create a color map for relapse annotation (0 = no relapse, 1 = relapse)
relapse_palette = {0: "lightgrey", 1: "red"}
col_colors = relapse_status.map(relapse_palette)  # now used as col_colors

# Plot clustered heatmap
sns.set_theme(style="white")
g = sns.clustermap(
    heatmap_data_T,
    col_colors=col_colors,  # samples are columns
    cmap="viridis",
    xticklabels=False,
    yticklabels=True,  # CpG names on rows
    figsize=(12, 8),
    metric="euclidean",
    method="average"
)

# Add legend for relapse annotation
for label in relapse_palette:
    g.ax_col_dendrogram.bar(0, 0, color=relapse_palette[label],
                            label=f"RFi_event={label}", linewidth=0)
g.ax_col_dendrogram.legend(loc="center", ncol=2)

# Save
g.savefig(outfile_heatmap, format="pdf", dpi=300, bbox_inches="tight")
plt.close()
print(f"Heatmap saved to: {outfile_heatmap}", flush=True)

################################################################################
# univar cox
################################################################################

cox_df = run_univariate_cox_for_cpgs(mval_matrix.loc[:,cpg_list], 
                                 clinical_data,
                                 time_col="RFi_years",
                                 event_col="RFi_event")

################################################################################
# histogram of beta values
################################################################################

plot_beta_histograms(beta_matrix, cpg_list, outfile_betahist,cox_results_df=cox_df,clinical_df=clinical_data)

################################################################################
# position barplot
################################################################################

counts = cpg_anno["featureClass"].value_counts()
plt.figure(figsize=(6, 4))  # Adjust size as needed
counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.xlabel("Feature Class")
plt.ylabel("Count")
plt.title("CpG Annotation Feature Classes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(outfile_posbarplot, dpi=300, format="pdf", bbox_inches="tight")
plt.close()
print(f"Position bar plot saved to: {outfile_posbarplot}", flush=True)

################################################################################
# gsea
################################################################################

enrich_and_plot(cpg_anno,
                    outfile=outfile_enrich,
                    top_n = 10)

################################################################################
# THE END
################################################################################

end_time = time.time()  # Record end time
log(f"Script ended at: {time.ctime(end_time)}")
log(f"Script executed in {end_time - start_time:.2f} seconds.")
#logfile.close()