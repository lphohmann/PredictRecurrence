# Script: overview data modalities SCAN-B
# Author: Lennart Hohmann
# Date: 02.03.2025
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
output.path <- "./output/"
#dir.create(output.path)
#-------------------
# input paths
infile.1 <- "./data/raw/Summarized_SCAN_B_rel4_NPJbreastCancer_with_ExternalReview_Bosch_data.RData"
infile.2 <- "./data/raw/genematrix_noNeg.Rdata" 
infile.3 <- "./data/raw/GSE278586_ProcessedData_LUepic_n499.txt"
infile.4 <- "./data/raw/SCANBrel4_n6614_ExpressedSomaticVariants.csv"
# rna seq mutations
# infile.3 <- ""  #499 cases with DNA methylation dat (EPIC)
# infile.4 <- ""  #499 cases with WGS dat
# CNA?

# output paths
outfile.1 <- paste0(output.path,"SCANB_Modalities.pdf") #.txt

#######################################################################

# function: loads RData data file and allows to assign it directly to variable
loadRData <- function(file.path){
  load(file.path)
  get(ls()[ls() != "file.path"])
}

#######################################################################
# clinical data
#######################################################################

# annotation 
anno <- loadRData(infile.1)
anno <- anno[anno$Follow.up.cohort == TRUE,]

#View(anno)
anno <- anno[c("Sample","GEX.assay","ER","PR","HER2","LN.spec",
     "NHG","Size.mm","TreatGroup","DRFi_days",
     "Age",  "OSbin","OS","RFIbin","RFI",
     "DRFIbin","DRFI","NCN.PAM50")]

# split treatment
anno$TreatGroup[anno$TreatGroup == ""] <- NA
anno$TreatGroup[is.na(anno$TreatGroup)] <- "Missing"
anno$Chemo <- ifelse(grepl("Chemo", anno$TreatGroup), 1, 0)
anno$Endo <- ifelse(grepl("Endo", anno$TreatGroup), 1, 0)
anno$Immu <- ifelse(grepl("Immu", anno$TreatGroup), 1, 0)

anno$Group <- ifelse(anno$ER == "Positive" & 
                     anno$HER2 == "Negative",
                     "ER+HER2-",
                     ifelse(anno$ER == "Positive" & 
                            anno$HER2 == "Positive",
                            "ER+HER2+",
                            ifelse(anno$ER == "Negative" & 
                                   anno$HER2 == "Positive",
                                   "ER-HER2+",
                                   ifelse(anno$ER == "Negative" & 
                                          anno$HER2 == "Negative" & 
                                          anno$PR == "Negative", 
                                          "TNBC",
                                            "Other"))))

#table(anno$Group,anno$NCN.PAM50)

#######################################################################
# RNA-seq data
#######################################################################

# load and prep gene expr. data
gex.dat <- as.data.frame(loadRData(infile.2))

# correct colnames
gex.dat <- gex.dat[anno$GEX.assay]
names(gex.dat) <- anno$Sample[match(colnames(gex.dat),
                                    anno$GEX.assay)]

#######################################################################
# DNA methylation data (n=499)
#######################################################################

# Read the first line of the file
epic.samples <- readLines(infile.3, n = 1)
epic.samples <- strsplit(epic.samples, "\t")[[1]]
epic.samples <- epic.samples[!grepl("Detection_Pval", epic.samples)]
epic.samples <- sub("\\..*", "", epic.samples[-1])

#epic.dat <- read.table(infile.3) # too big to load

#######################################################################
# WGS variant calling data (n=?) 499 -> is published?; 
# alternative RNAseq mutations
#######################################################################

mut.dat <- read.csv(infile.4)
RNAmut.samples <- names(mut.dat)

#######################################################################
# Available modalities overview table
#######################################################################

# overview table
modalities.df <- anno[c("Sample")]
modalities.df$Clinical <- 1
modalities.df$RNAseq.gex <- ifelse(modalities.df$Sample %in% names(gex.dat),1,0)
modalities.df$RNAseq.mut <- ifelse(modalities.df$Sample %in% RNAmut.samples,1,0)
modalities.df$DNAmethyl <- ifelse(modalities.df$Sample %in% epic.samples,1,0)
modalities.df$WGS.mut <- ifelse(modalities.df$Sample %in% epic.samples,1,0) # hypothetically

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
    title = paste0("SCAN-B; All; n=",length(anno$Sample),
    subtitle = "With variants from RNAseq"
  ))

# Prepare list of sample groups
venn_list <- list(
  "DNAmethyl" = modalities.df$Sample[modalities.df$DNAmethyl == 1],
  "WGS.mut" = modalities.df$Sample[modalities.df$WGS.mut == 1],
  "RNAseq.gex" = modalities.df$Sample[modalities.df$RNAseq.gex == 1]
)

# Create the Venn diagram
plot.2 <- ggVennDiagram(venn_list, label_alpha = 0) + 
  scale_fill_gradientn(colors = c("#F8BBD0", "#EC407A", "#C2185B"), 
                       values = c(0, 0.2, 1)) + 
  theme_void() +  
  theme(legend.position = "none") +
  labs(
    title = paste0("SCAN-B; All; n=",length(anno$Sample),
    subtitle = "With variants from WGS"
  ))


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

# Prepare list of sample groups
venn_list <- list(
  "DNAmethyl" = modalities.df$Sample[modalities.df$DNAmethyl == 1 & 
                                       modalities.df$Sample %in% anno$Sample[anno$Group==clin.group]],
  "WGS.mut" = modalities.df$Sample[modalities.df$WGS.mut == 1 & 
                                     modalities.df$Sample %in% anno$Sample[anno$Group==clin.group]],
  "RNAseq.gex" = modalities.df$Sample[modalities.df$RNAseq.gex == 1 & 
                                        modalities.df$Sample %in% anno$Sample[anno$Group==clin.group]]
)

# Create the Venn diagram
plot.4 <- ggVennDiagram(venn_list, label_alpha = 0) + 
  scale_fill_gradientn(colors = c("#F8BBD0", "#EC407A", "#C2185B"), 
                       values = c(0, 0.2, 1)) + 
  theme_void() +  
  theme(legend.position = "none") +
  labs(
    title = paste0("SCAN-B; ER+HER2-; n=",
                   length(anno$Sample[anno$Group==clin.group])),
    subtitle = "With variants from WGS"
  )

#######################################################################
# pie chart FU cohort ClinGroups
#######################################################################

# Step 2: Summarize counts of each group using table
group_counts <- table(anno$Group)[c("ER+HER2-","ER+HER2+","ER-HER2+","TNBC","Other")]
group_counts <- as.data.frame(group_counts)
group_counts$Percentage <- round((group_counts$Freq / sum(group_counts$Freq)) * 100,0)

# Step 3: Plot the pie chart using ggplot2
plot.5 <- ggplot(data = group_counts, aes(x = "", y = Freq, fill = Var1)) +
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
    title = paste0("SCAN-B; ERpHER2n; n=",
                   length(anno$Sample[anno$Group==clin.group])),
    subtitle = paste0("Treatment groups; None n=", 
    length(anno$Sample[anno$TreatGroup == "None" & anno$Group==clin.group]),
    "; Missing n=",length(anno$Sample[anno$TreatGroup == "Missing" & anno$Group==clin.group]))
  )

#######################################################################
# save plots to pdf
#######################################################################

# Set up the PDF output
pdf(file = outfile.1, onefile = TRUE, width = 8.27, height = 11.69)

# Arrange the plots in a grid (2x2)
grid.arrange(plot.1, plot.2, plot.3, plot.4, 
             plot.5, plot.6, ncol = 2, nrow = 3)

# Close the PDF device to save the file
dev.off()
