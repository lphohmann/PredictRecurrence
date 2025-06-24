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
import matplotlib.pyplot as plt

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
infile_cpg_set = "./output/CoxNet_unadjusted/Selected_model/selected_cpgs.txt"# ⚠️ ADAPT
infile_cpg_anno = "./data/raw/EPIC_probeAnnoObj.csv"

infile_train_ids = "./data/train/train_subcohorts/TNBC_train_ids.csv" # sample ids of training cohort
infile_betavalues = "./data/train/train_methylation_unadjusted.csv" # ⚠️ ADAPT
infile_clinical = "./data/train/train_clinical.csv"

################################################################################
# PARAMS
################################################################################

# Output directory and files
output_dir = "output/CpG_set_annotation/CoxNet_unadjusted" # ⚠️ ADAPT
os.makedirs(output_dir, exist_ok=True)
outfile_correlation = os.path.join(output_dir, "correlation_matrix.png")
outfile_posbarplot = os.path.join(output_dir, "position_barplot.png")
outfile_enrich= os.path.join(output_dir, "enrichment_barplot.png")
outfile_betahist= os.path.join(output_dir, "beta_histograms.pdf")

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
cpg_anno.head()

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

cpg_anno = cpg_anno[cpg_anno['illuminaID'].isin(cpg_list)]

# Get counts
counts = cpg_anno["featureClass"].value_counts()

# Create new figure
plt.figure(figsize=(6, 4))  # Adjust size as needed

# Bar plot
counts.plot(kind='bar', color='skyblue', edgecolor='black')

# Labels and title
plt.xlabel("Feature Class")
plt.ylabel("Count")
plt.title("CpG Annotation Feature Classes")
plt.xticks(rotation=45)
plt.tight_layout()

# Save figure
plt.savefig(outfile_posbarplot, dpi=300, bbox_inches="tight")
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