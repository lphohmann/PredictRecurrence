#!/usr/bin/env Rscript
# Script: overview data modalities SCAN-B FU methylation cohort
# Author: Lennart Hohmann
# Date: 19.03.2025
#-------------------
# empty environment
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence/")
#-------------------
# packages
#source("./scripts/src/")
if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, ggVennDiagram, gridExtra, data.table, naniar)
#-------------------
# set/create output directories
output_path <- "./output/"
dir.create(output_path)
#-------------------
# input paths
infile_1 <- "./data/standardized/SCANB_sample_modalities.csv"
infile_2 <- "./data/standardized/SCANB_clinical.csv"
infile_3 <- "./data/standardized/SCANB_RNAseq_expression.csv"
infile_4 <- "./data/standardized/SCANB_RNAseq_mutations.csv"
infile_5 <- "./data/standardized/SCANB_DNAmethylation.csv"
#-------------------
# which clin group to run for
clin_group <- "ER+HER2-" # All ER+HER2-
#-------------------
# output paths
outfile_1 <- paste0(output_path,"SCANB_FUmethyl_Modalities_",clin_group,".pdf")
outfile_2 <- paste0(output_path,"SCANB_FUmethyl_Modalities_",clin_group,".txt") 
#-------------------
# storing objects 
#plot.list <- list() # object to store plots
txt.out <- c() # object to store text output, ggf. use capture.output()

#######################################################################
# load data
#######################################################################

sample_modalities <- read.csv(infile_1)
clinical <- read.csv(infile_2)
#RNAseq_expr <- read.csv(infile_3) # takes too long
RNAseq_expr <- data.table::fread(infile_3)
RNAseq_expr <- as.data.frame(RNAseq_expr)
RNAseq_mut <- read.csv(infile_4)
DNAmethyl <- read.csv(infile_5) # until the real data is loaded
clinical$NHG <- as.character(clinical$NHG)

# only include methylaiton cohort samples
sample_modalities <- sample_modalities[sample_modalities$DNAmethylation==1,]
clinical <- clinical[clinical$Sample %in% sample_modalities$Sample,]
RNAseq_expr <- RNAseq_expr[, c("Gene", intersect(colnames(RNAseq_expr), sample_modalities$Sample))]
RNAseq_mut <- RNAseq_mut[, c("Gene", intersect(colnames(RNAseq_mut), sample_modalities$Sample))]
DNAmethyl <- DNAmethyl[ , c("ID_REF", intersect(colnames(DNAmethyl), sample_modalities$Sample))]

# Subgroup data
if (clin_group == "All") {
  sub_sample_modalities <- sample_modalities
  sub_clinical <- clinical
  sub_RNAseq_expr <- RNAseq_expr
  sub_RNAseq_mut <- RNAseq_mut
  sub_DNAmethyl <- DNAmethyl
} else {
  sub_clinical <- subset(clinical, Group == clin_group)
  sub_sample_modalities <- subset(sample_modalities, Sample %in% sub_clinical$Sample)
  sub_RNAseq_expr <- RNAseq_expr[, c("Gene", intersect(colnames(RNAseq_expr), sub_clinical$Sample))]
  sub_RNAseq_mut <- RNAseq_mut[, c("Gene", intersect(colnames(RNAseq_mut), sub_clinical$Sample))]
  sub_DNAmethyl <- DNAmethyl[c("ID_REF", intersect(colnames(DNAmethyl), sub_clinical$Sample))]
}

#######################################################################
# Pie chart FU cohort ClinGroups
#######################################################################

group_counts <- table(clinical$Group)[c("ER+HER2-","ER+HER2+","ER-HER2+","TNBC","Other")]
group_counts <- as.data.frame(group_counts)
group_counts$Percentage <- round((group_counts$Freq / sum(group_counts$Freq)) * 100,0)
# Plot the pie chart using ggplot2
plot_1 <- ggplot(data = group_counts, aes(x = "", y = Freq, fill = Var1)) +
  geom_bar(stat = "identity", width = 1, show.legend = TRUE) + 
  coord_polar(theta = "y") +  # This makes the chart circular
  labs(title = "ER & HER2 status in FU-methylation cohort") +
  theme_void() +  # Remove the background grid
  theme(legend.position = "none",
    axis.text.x = element_blank(),  # Remove x-axis text
        axis.ticks = element_blank(),  # Remove axis ticks
        legend.title = element_blank()) +  # Remove legend title
  scale_fill_manual(values = c(
    "ER+HER2-" = "#abd9e9", 
    "ER+HER2+" = "#f1b6da", 
    "ER-HER2+" = "#d01c8b", 
    "TNBC" = "#d7191c",
    "Other" ="#bababa"
  )) +
  geom_text(aes(label = paste(Var1, "\n", Freq, " (", Percentage, "%)", sep = "")), 
            position = position_stack(vjust = 0.5), size = 4)

#######################################################################
# Modalities Venn FU-methyl cohort
#######################################################################

# Prepare list of sample groups
venn_list <- list(
  "DNAmethyl" = sample_modalities$Sample[sample_modalities$DNAmethyl == 1],
  "RNAseq.mut" = sample_modalities$Sample[sample_modalities$RNAseq_mutations == 1],
  "RNAseq.gex" = sample_modalities$Sample[sample_modalities$RNAseq_expression == 1]
)

# Create the Venn diagram
plot_2 <- ggVennDiagram(venn_list, label_alpha = 0) +
  scale_fill_gradientn(
    colors = c("#FFFFFF", "#FFFFFF", "#FFFFFF"),
    values = c(0, 0.2, 1)
  ) +
  theme_void() +
  theme(legend.position = "none") +
  labs(
    title = paste0("SCAN-B FU-methyl cohort; n=", length(clinical$Sample)) # ,
    # subtitle = "With variants from RNAseq"
  )

# Prepare list of sample groups
venn_list <- list(
  "DNAmethyl" = sample_modalities$Sample[sample_modalities$DNAmethyl == 1],
  "Clinical" = sample_modalities$Sample[sample_modalities$clinical == 1],
  "RNAseq.gex" = sample_modalities$Sample[sample_modalities$RNAseq_expression == 1]
)

# Create the Venn diagram
plot_3 <- ggVennDiagram(venn_list, label_alpha = 0) + 
  scale_fill_gradientn(colors = c("#FFFFFF","#FFFFFF","#FFFFFF"), 
                       values = c(0, 0.2, 1)) +
  theme_void() +  
  theme(legend.position = "none") +
  labs(
    title = paste0("SCAN-B FU-methyl cohort; n=",length(clinical$Sample))#,
    #subtitle = "With variants from RNAseq"
  )

#######################################################################
# Clinical data in-depth
#######################################################################

# Missing #############################################################
# Plot Nullity Matrix
sub_clinical$TreatGroup[sub_clinical$TreatGroup == "Missing"] <- NA
plot_4 <- gg_miss_upset(sub_clinical[c("Age","PR","LN","NHG","Size.mm","OS_event","OS_years","RFi_event","RFi_years","DRFi_event","DRFi_years")]) 
plot_5 <- vis_miss(sub_clinical[c("ER","HER2","PR","Age","PR","LN","NHG","Size.mm","TreatGroup","RFi_event","RFi_years","OS_event","OS_years","DRFi_event","DRFi_years")]) +
scale_fill_manual(values = c("black", "red")) +
theme(
    axis.text.x = element_text(size = 14, angle = 45, hjust = 0.5),
    axis.title = element_text(size = 16),
    axis.text.y = element_text(size = 14))  # Angle and adjust )
#gg_miss_var(sub_clinical)

# Events #############################################################
events <- lapply(list(
  sub_clinical$OS_event,
  sub_clinical$RFi_event, 
  sub_clinical$DRFi_event),
  table)
events <- data.frame(
  Outcome = c("OS","OS","RFi","RFi","DRFi","DRFi"),
  Count = c(as.vector(events[[1]]),as.vector(events[[2]]),as.vector(events[[3]])), Event = c(0,1,0,1,0,1))
plot_6 <- ggplot(events, aes(x = Outcome, y = Count, fill = as.factor(Event))) +
  geom_bar(stat = 'identity', position = 'stack') +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5)) +
  labs(x = 'Outcome', y = 'Count', title = "OS vs. RFI vs. DRFi Events") +
  geom_text(aes(label = Count), position = position_stack(vjust = 0.5)) +  # Display count labels inside the bars
  theme_minimal()

# # Treatments #############################################################
# venn_list <- list(
#   "Endocrine" = sub_clinical$Sample[sub_clinical$Endo == 1 & sub_clinical$Group==clin_group],
#   "Chemo" = sub_clinical$Sample[sub_clinical$Chemo == 1 & sub_clinical$Group==clin_group],
#   "Immu" = sub_clinical$Sample[sub_clinical$Immu == 1 & sub_clinical$Group==clin_group])

# # Create the Venn diagram
# plot_7 <- ggVennDiagram(venn_list, label_alpha = 0) + 
#   scale_fill_gradientn(colors = c("#FFFFFF","#FFFFFF","#FFFFFF")) + 
#   theme_void() +  
#   theme(legend.position = "none") +
#   labs(
#     title = paste0("Treatment in ERpHER2n; n=",
#                    length(sub_clinical$Sample[sub_clinical$Group==clin_group])),
#     subtitle = paste0("None n=", 
#     length(clinical$Sample[clinical$TreatGroup == "None" & clinical$Group==clin_group]),
#     "; Missing n=",length(clinical$Sample[clinical$TreatGroup == "Missing" & clinical$Group==clin_group]))
#   )


#######################################################################
# Check missing data in RNA-seq
#######################################################################

#res <- apply(sub_RNAseq_expr,1,sum(is.na()))


#######################################################################
# save plots to pdf
#######################################################################

# Set up the PDF output
pdf(file = outfile_1, onefile = TRUE)#, height = 8.27/2, width = 11.69/2)

grid.arrange(plot_1, plot_2, plot_3, plot_6, ncol = 2, nrow = 2)
#print(plot_7)
print(plot_4)
print(plot_5)

dev.off()
