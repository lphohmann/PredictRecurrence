#!/usr/bin/env Rscript
# Script: Standardizing patient identifiers and structure across SCAN-B datasets
# Author: Lennart Hohmann
# Date: 10.03.2025
#----------------------------------------------------------------------
# empty environment
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence/")
#----------------------------------------------------------------------
# packages
#source("./scripts/src/")
if (!require("pacman")) install.packages("pacman")
pacman::p_load(biomaRt,caret,data.table)
#----------------------------------------------------------------------
# input paths
infile.1 <- "./data/raw/Summarized_SCAN_B_rel4_NPJbreastCancer_with_ExternalReview_Bosch_data.RData"
infile.2 <- "./data/raw/genematrix_noNeg.Rdata" 
infile.3 <- "./data/standardized/SCANB_DNAmethylation.csv" # this was created before
infile.4 <- "./data/raw/SCANBrel4_n6614_ExpressedSomaticVariants.csv" 
#----------------------------------------------------------------------
# output paths
output.path <- "./data/standardized/"
dir.create(output.path, showWarnings = FALSE)
outfile.1 <- paste0(output.path,"SCANB_sample_modalities.csv")
outfile.2 <- paste0(output.path,"SCANB_clinical.csv")
outfile.3 <- paste0(output.path,"SCANB_RNAseq_expression.csv")
outfile.4 <- paste0(output.path,"SCANB_RNAseq_mutations.csv")
#######################################################################
# function: loads RData data file and allows to assign it directly to variable
loadRData <- function(file.path){
  load(file.path)
  get(ls()[ls() != "file.path"])
}
# function: convert ensemble  gene ids to hugo
ensemble_to_hgnc <- function(ensemble.ids) {
  # Connect to the Ensembl database
  mart <- useDataset("hsapiens_gene_ensembl", useMart("ensembl"))
  # Retrieve Hugo gene names using biomaRt
  res <- getBM(attributes = c("ensembl_gene_id", "hgnc_symbol"),
               filters = "ensembl_gene_id",
               values = ensemble.ids,
               mart = mart,
               uniqueRows = TRUE)
  return(res)
}
#######################################################################
# clinical data
#######################################################################
clin.dat <- loadRData(infile.1)
clin.dat <- clin.dat[clin.dat$Follow.up.cohort == TRUE,]
clin.dat$LN <- ifelse(clin.dat$LN > 0, "N+", "N0")
clin.dat$PR[clin.dat$PR==""] <-NA
# select clinical variables to include
clin.dat <- clin.dat[c("Sample","GEX.assay","ER","PR","HER2","LN",
     "NHG","Size.mm","TreatGroup","InvCa.type",
     "Age","NCN.PAM50",
     "DRFi_days","DRFi_event",
     "OS_days","OS_event",
     "RFi_days","RFi_event")]
# convert outcome to years
clin.dat$OS_years <- clin.dat$OS_days / 365
clin.dat$RFi_years <- clin.dat$RFi_days / 365
clin.dat$DRFi_years <- clin.dat$DRFi_days / 365
clin.dat$OS_days <- NULL
clin.dat$RFi_days <- NULL
clin.dat$DRFi_days <- NULL
# split treatment
clin.dat$TreatGroup[clin.dat$TreatGroup == ""] <- NA
clin.dat$TreatGroup[is.na(clin.dat$TreatGroup)] <- "Missing"
#clin.dat$Chemo <- ifelse(grepl("Chemo", clin.dat$TreatGroup), 1, 0)
#clin.dat$Endo <- ifelse(grepl("Endo", clin.dat$TreatGroup), 1, 0)
#clin.dat$Immu <- ifelse(grepl("Immu", clin.dat$TreatGroup), 1, 0)
# create column for clinical groups (ER and HER2 status)
clin.dat$Group <- ifelse(clin.dat$ER == "Positive" & clin.dat$HER2 == "Negative",
                     "ER+HER2-",
                     ifelse(clin.dat$ER == "Positive" & 
                            clin.dat$HER2 == "Positive",
                            "ER+HER2+",
                            ifelse(clin.dat$ER == "Negative" & 
                                   clin.dat$HER2 == "Positive",
                                   "ER-HER2+",
                                   ifelse(clin.dat$ER == "Negative" & 
                                          clin.dat$HER2 == "Negative" & 
                                          clin.dat$PR == "Negative", 
                                          "TNBC",
                                            "Other"))))
#######################################################################
# RNA-seq expression data
#######################################################################
# load and prep gene expr. data
RNAseq_expr.dat <- as.data.frame(loadRData(infile.2))
# correct colnames
RNAseq_expr.dat <- RNAseq_expr.dat[clin.dat$GEX.assay]
names(RNAseq_expr.dat) <- clin.dat$Sample[match(colnames(RNAseq_expr.dat),clin.dat$GEX.assay)]
clin.dat <- clin.dat[names(clin.dat) != "GEX.assay"]
# correct gene names
RNAseq_expr.dat$Ensemble_ID <- gsub("\\..*$", "", rownames(RNAseq_expr.dat))
rownames(RNAseq_expr.dat) <- NULL
hgnc.table <- ensemble_to_hgnc(RNAseq_expr.dat$Ensemble_ID)
RNAseq_expr.dat$Hgnc_ID <- hgnc.table$hgnc_symbol[match(RNAseq_expr.dat$Ensemble_ID, hgnc.table$ensembl_gene_id)]
RNAseq_expr.dat <- RNAseq_expr.dat[!duplicated(RNAseq_expr.dat$Hgnc_ID), ] # keep 1st row of each symbol
RNAseq_expr.dat <- RNAseq_expr.dat[!is.na(RNAseq_expr.dat$Hgnc_ID),]
RNAseq_expr.dat$Gene <- RNAseq_expr.dat$Hgnc_ID
RNAseq_expr.dat$Hgnc_ID <- NULL
RNAseq_expr.dat$Ensemble_ID <- NULL
RNAseq_expr.dat <- RNAseq_expr.dat[c("Gene", setdiff(names(RNAseq_expr.dat), "Gene"))]
#######################################################################
# RNA-seq variant data
#######################################################################
RNAseq_mut.dat <- read.csv(infile.4)
# exclude genes not in my gene expression data
RNAseq_mut.dat <- RNAseq_mut.dat[RNAseq_mut.dat$Gene %in% RNAseq_expr.dat$Gene,]
#######################################################################
# already standardized DNA methylation data (only load sampleIDs)
#######################################################################
# # Read the first line of the file
# DNA_methyl.ids <- readLines(infile.3, n = 1)
# DNA_methyl.ids <- strsplit(DNA_methyl.dat, ",")[[1]]
# DNA_methyl.ids <- DNA_methyl.ids[-1]
#######################################################################
# Available modalities overview table
#######################################################################
modalities.dat <- clin.dat[c("Sample")]
modalities.dat$clinical <- 1
modalities.dat$RNAseq_expression <- ifelse(modalities.dat$Sample %in% names(RNAseq_expr.dat),1,0)
modalities.dat$RNAseq_mutations <- ifelse(modalities.dat$Sample %in% names(RNAseq_mut.dat),1,0)
modalities.dat$DNAmethylation <- ifelse(modalities.dat$Sample %in% DNA_methyl.ids, 1, 0)
#######################################################################
# save output files
#######################################################################
write.csv(modalities.dat, file=outfile.1, row.names = FALSE)
write.csv(clin.dat, file=outfile.2, row.names = FALSE)
write.csv(RNAseq_expr.dat, file=outfile.3, row.names = FALSE)
write.csv(RNAseq_mut.dat, file=outfile.4, row.names = FALSE)