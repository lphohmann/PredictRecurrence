#!/usr/bin/env python3
################################################################################
# Script: Overview of the SCAN-B Follow-up cohort (2010-2018)
# Author: Lennart Hohmann
# Date: 11.03.2025
################################################################################

# import 
import os
import sys
import missingno as msno
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
from venn import venn
from matplotlib.backends.backend_pdf import PdfPages
import src.utils as my
my.test_func()

# set wd
os.chdir(os.path.expanduser("~/PhD_Workspace/PredictRecurrence/"))

################################################################################
################################################################################

# input paths
infile_1 = "./data/standardized/SCANB_sample_modalities.csv"
infile_2 = "./data/standardized/SCANB_clinical.csv"
infile_3 = "./data/standardized/SCANB_RNAseq_expression.csv"
infile_4 = "./data/standardized/SCANB_RNAseq_mutations.csv"
infile_5 = "./data/standardized/SCANB_DNAmethylation.csv"

# set subgroup to run the analyses
#clin_group = "All" # "All" or "ER+HER2-"
clin_group = "ER+HER2-"

fig_size = (8.27/2, 11.69/2)

# output paths
outfile_1 = f"./output/SCANB_Modalities_Surv_{clin_group}.pdf"

# plot file
pdf = PdfPages(outfile_1)

################################################################################
# load data
################################################################################

# read in data
sample_modalities = pd.read_csv(infile_1)
clinical = pd.read_csv(infile_2)
RNAseq_expr = pd.read_csv(infile_3)
RNAseq_mut = pd.read_csv(infile_4)
DNAmethyl = pd.read_csv(infile_5) # until the real data is loaded
DNAmethyl.columns = ["Sample"] # until the real data is loaded

# subgroup data
if clin_group == "All":
    sub_sample_modalities = sample_modalities
    sub_clinical = clinical
    sub_RNAseq_expr = RNAseq_expr
    sub_RNAseq_mut = RNAseq_mut
    sub_DNAmethyl = DNAmethyl
else:
    sub_clinical = clinical[clinical['Group']==clin_group]
    sub_sample_modalities = sample_modalities[sample_modalities['Sample'].isin(sub_clinical['Sample'])]
    sub_RNAseq_expr = RNAseq_expr.loc[:,
                                      RNAseq_expr.columns.isin(['Gene'] + list(sub_clinical["Sample"]))]
    sub_RNAseq_mut = RNAseq_mut.loc[:,
                                      RNAseq_mut.columns.isin(['Gene'] + list(sub_clinical["Sample"]))]
    sub_DNAmethyl = DNAmethyl.isin(sub_clinical['Sample'])


################################################################################
# Piechart of ER and HER2 status in SCAN-B
################################################################################

group_order = ["ER+HER2-", "ER+HER2+", "ER-HER2+", "TNBC", "Other"]
group_counts = clinical['Group'].value_counts()[group_order]
# rename the count column as 'Count'
group_counts_df = group_counts.reset_index(name='Count')
# Add a new column 'Percentage' and calculate the percentage for each group
group_counts_df = group_counts_df.assign(
    Percentage=lambda x: (x['Count'] / x['Count'].sum()) * 100
)
group_counts_df = group_counts_df.round({'Percentage': 0})
# Rename the 'index' column to 'Group'
group_counts_df = group_counts_df.rename(columns={'index': 'Group'})
# Plotting the pie chart
plt.figure(figsize=fig_size)
plt.pie(group_counts_df['Count'], 
        labels=group_counts_df['Group'] + '\n' + group_counts_df['Count'].astype(str) + ' (' + group_counts_df['Percentage'].astype(str) + '%)', 
        autopct='%1.0f%%', 
        colors=['#abd9e9', '#f1b6da', '#d01c8b', '#d7191c', '#bababa'], 
        startangle=90, 
        wedgeprops={'edgecolor': 'black'})
plt.title('ER & HER2 status')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
pdf.savefig() # save to pdf
#plt.show()
plt.close()

################################################################################
# Available data modalities
################################################################################

# Create a list of sample groups based on conditions in the DataFrame
venn_dict = {
    "DNAmethyl": set(sub_sample_modalities[sub_sample_modalities['DNAmethylation'] == 1]['Sample']),
    "RNAseq_mut": set(sub_sample_modalities[sub_sample_modalities['RNAseq_mutations'] == 1]['Sample']),
    "RNAseq_gex": set(sub_sample_modalities[sub_sample_modalities['RNAseq_expression'] == 1]['Sample'])
}
plt.figure(figsize=fig_size)
venn(venn_dict)
plt.title(f"SCAN-B; {clin_group}; n={len(sub_sample_modalities['Sample'])}")
pdf.savefig() # save to pdf
#plt.show()
plt.close()

################################################################################
# In-depth look: Clinicopathological variables
################################################################################

summary = sub_clinical.describe(include='all')  # Summary of all columns (numerical and categorical)
#print(summary)

# Missing data
plt.figure()#figsize=fig_size[::-1])
msno.matrix(sub_clinical)
plt.title('Nullity Matrix')
plt.savefig(f"./output/SCANB_NullityMatrix_{clin_group}.png", pad_inches=0.1)
#pdf.savefig(pad_inches=0.1)  # Save to PDF
plt.close()

plt.figure()#figsize=fig_size[::-1])
msno.bar(sub_clinical)
plt.title('Nullity by Column')
pdf.savefig(pad_inches=0.1)  # Save to PDF
plt.close()

# number of events per outcome measure
# OS vs RFI
cross_tab = pd.crosstab(sub_clinical['OS_event'], sub_clinical['RFi_event'])
#print(cross_tab)
formatted_string = "; ".join([
    f"OS_event 0 n= {cross_tab.iloc[0,0]}",
    f"OS_event 1 n= {cross_tab.iloc[1,0]}",
    f"RFi_event 0 n= {cross_tab.iloc[0,1]}",
    f"RFi_event 1 n= {cross_tab.iloc[1,1]}"
])

#print(f"OS vs. RFI events: {formatted_string}.")
# Count the events
event_counts = sub_clinical[["OS_event", "RFi_event","DRFi_event"]].apply(pd.Series.value_counts).T
# plot bars in stack manner
ax = event_counts.plot(kind='bar', stacked=True, figsize=fig_size, 
                       rot=0, xlabel='Class', ylabel='Count')
for c in ax.containers:

    # Optional: if the segment is small or 0, customize the labels
    labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
    
    # remove the labels parameter if it's not needed for customized labels
    ax.bar_label(c, labels=list(map(int, labels)), label_type='center')

plt.title(f"OS vs. RFI events: {formatted_string}.",fontsize=8)
plt.tight_layout()
pdf.savefig()  # Save to PDF
plt.close()

# treatments
treatment_counts = sub_clinical[["Endo","Chemo"]]
# Create a list of sample groups based on conditions in the DataFrame
venn_dict = {
    "Endo": set(sub_clinical[sub_clinical['Endo'] == 1]['Sample']),
    "Chemo": set(sub_clinical[sub_clinical['Chemo'] == 1]['Sample']),
    "Immu": set(sub_clinical[sub_clinical['Immu'] == 1]['Sample'])
}

plt.figure(figsize=fig_size)
venn(venn_dict)
plt.title(f"SCAN-B; {clin_group}; n={len(sub_clinical['Sample'])}")
pdf.savefig() # save to pdf
#plt.show()
plt.close()

pdf.close()


################################################################################
# Survival analyses and Censoring distribution
################################################################################
