#!/usr/bin/env Rscript
# Script: overview data modalities SCAN-B
# Author: Lennart Hohmann
# Date: 11.03.2025
#-------------------
# empty environment
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence/")
#-------------------
# packages
#source("./scripts/src/")
if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, ggVennDiagram, gridExtra)
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
clin_group <- "ER+HER2-"
#-------------------
# output paths
outfile_1 <- paste0(output_path,"SCANB_Modalities_",clin_group,".pdf")
outfile_2 <- paste0(output.path,"SCANB_Modalities_",clin_group,".txt") 
#-------------------
# storing objects 
plot.list <- list() # object to store plots
txt.out <- c() # object to store text output, ggf. use capture.output()

#######################################################################
# load data
#######################################################################

sample_modalities <- read.csv(infile_1)
clinical <- read.csv(infile_2)
RNAseq_expr <- read.csv(infile_3) # nolint
RNAseq_mut <- read.csv(infile_4)
DNAmethyl <- read.csv(infile_5) # until the real data is loaded
names(DNAmethyl) <- c("Sample")

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
    sub_DNAmethyl <- DNAmethyl[DNAmethyl$Sample %in% sub_clinical$Sample, ]
}

#######################################################################
# Pie chart FU cohort ClinGroups
#######################################################################

group_counts <- table(clinical$Group)[c("ER+HER2-","ER+HER2+","ER-HER2+","TNBC","Other")]
group_counts <- as.data.frame(group_counts)
group_counts$Percentage <- round((group_counts$Freq / sum(group_counts$Freq)) * 100,0)
# Plot the pie chart using ggplot2
plot.1 <- ggplot(data = group_counts, aes(x = "", y = Freq, fill = Var1)) +
  geom_bar(stat = "identity", width = 1, show.legend = TRUE) + 
  coord_polar(theta = "y") +  # This makes the chart circular
  labs(title = "ER & HER2 status") +
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
# Whole FU cohort modalities
#######################################################################

# Prepare list of sample groups
venn_list <- list(
  "DNAmethyl" = modalities.df$Sample[modalities.df$DNAmethyl == 1],
  "RNAseq.mut" = modalities.df$Sample[modalities.df$RNAseq.mut == 1],
  "RNAseq.gex" = modalities.df$Sample[modalities.df$RNAseq.gex == 1]
)

# Create the Venn diagram
plot.1 <- ggVennDiagram(venn_list, label_alpha = 0) + 
  scale_fill_gradientn(colors = c("#F8BBD0", "#EC407A", "#C2185B"), 
                       values = c(0, 0.2, 1)) +
  theme_void() +  
  theme(legend.position = "none") +
  labs(
    title = paste0("SCAN-B; All; n=",length(anno$Sample)),
    subtitle = "With variants from RNAseq"
  )

# Prepare list of sample groups
# venn_list <- list(
#   "DNAmethyl" = modalities.df$Sample[modalities.df$DNAmethyl == 1],
#   "WGS.mut" = modalities.df$Sample[modalities.df$WGS.mut == 1],
#   "RNAseq.gex" = modalities.df$Sample[modalities.df$RNAseq.gex == 1]
# )

# # Create the Venn diagram
# plot.2 <- ggVennDiagram(venn_list, label_alpha = 0) + 
#   scale_fill_gradientn(colors = c("#F8BBD0", "#EC407A", "#C2185B"), 
#                        values = c(0, 0.2, 1)) + 
#   theme_void() +  
#   theme(legend.position = "none") +
#   labs(
#     title = paste0("SCAN-B; All; n=",length(anno$Sample),
#     subtitle = "With variants from WGS"
#   ))


#######################################################################
# ERpHER2n FU cohort modalities
#######################################################################

clin.group <- "ER+HER2-"

# Prepare list of sample groups
venn_list <- list(
  "DNAmethyl" = modalities.df$Sample[modalities.df$DNAmethyl == 1 & 
                                       modalities.df$Sample %in% anno$Sample[anno$Group==clin.group]],
  "RNAseq.mut" = modalities.df$Sample[modalities.df$RNAseq.mut == 1 & 
                                        modalities.df$Sample %in% anno$Sample[anno$Group==clin.group]],
  "RNAseq.gex" = modalities.df$Sample[modalities.df$RNAseq.gex == 1 & 
                                        modalities.df$Sample %in% anno$Sample[anno$Group==clin.group]]
)

# Create the Venn diagram
plot.3 <- ggVennDiagram(venn_list, label_alpha = 0) + 
  scale_fill_gradientn(colors = c("#F8BBD0", "#EC407A", "#C2185B"), 
                       values = c(0, 0.2, 1)) +
  theme_void() +  
  theme(legend.position = "none") +
  labs(
    title = paste0("SCAN-B; ER+HER2-; n=",
                   length(anno$Sample[anno$Group==clin.group])),
    subtitle = "With variants from RNAseq"
  )

# # Prepare list of sample groups
# venn_list <- list(
#   "DNAmethyl" = modalities.df$Sample[modalities.df$DNAmethyl == 1 & 
#                                        modalities.df$Sample %in% anno$Sample[anno$Group==clin.group]],
#   "WGS.mut" = modalities.df$Sample[modalities.df$WGS.mut == 1 & 
#                                      modalities.df$Sample %in% anno$Sample[anno$Group==clin.group]],
#   "RNAseq.gex" = modalities.df$Sample[modalities.df$RNAseq.gex == 1 & 
#                                         modalities.df$Sample %in% anno$Sample[anno$Group==clin.group]]
# )

# # Create the Venn diagram
# plot.4 <- ggVennDiagram(venn_list, label_alpha = 0) + 
#   scale_fill_gradientn(colors = c("#F8BBD0", "#EC407A", "#C2185B"), 
#                        values = c(0, 0.2, 1)) + 
#   theme_void() +  
#   theme(legend.position = "none") +
#   labs(
#     title = paste0("SCAN-B; ER+HER2-; n=",
#                    length(anno$Sample[anno$Group==clin.group])),
#     subtitle = "With variants from WGS"
#   )



#######################################################################
# venn ERpHER2n FU cohort treatment #TreatGroup
#######################################################################

venn_list <- list(
  "Endocrine" = anno$Sample[anno$Endo == 1 & anno$Group==clin.group],
  "Chemo" = anno$Sample[anno$Chemo == 1 & anno$Group==clin.group],
  "Immu" = anno$Sample[anno$Immu == 1 & anno$Group==clin.group])

# Create the Venn diagram
plot.6 <- ggVennDiagram(venn_list, label_alpha = 0) + 
  scale_fill_gradientn(colors = c("#7fbf7b", "#c2a5cf","#fdc086")) + 
  theme_void() +  
  theme(legend.position = "none") +
  labs(
    title = paste0("Treatment in ERpHER2n; n=",
                   length(anno$Sample[anno$Group==clin.group])),
    subtitle = paste0("None n=", 
    length(anno$Sample[anno$TreatGroup == "None" & anno$Group==clin.group]),
    "; Missing n=",length(anno$Sample[anno$TreatGroup == "Missing" & anno$Group==clin.group]))
  )

#######################################################################
# save plots to pdf
#######################################################################

# Set up the PDF output
pdf(file = outfile.1, onefile = TRUE, width = 8.27, height = 11.69)

# Arrange the plots in a grid (2x2)
# grid.arrange(plot.1, plot.2, plot.3, plot.4, 
#              plot.5, plot.6, ncol = 2, nrow = 3)
grid.arrange(plot.5,plot.1, plot.3, plot.6, ncol = 2, nrow = 2)
# Close the PDF device to save the file
dev.off()
