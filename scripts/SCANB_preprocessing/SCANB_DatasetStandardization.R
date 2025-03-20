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
pacman::p_load(biomaRt,caret)
#----------------------------------------------------------------------
# input paths
infile.1 <- "./data/raw/Summarized_SCAN_B_rel4_NPJbreastCancer_with_ExternalReview_Bosch_data.RData"
infile.2 <- "./data/raw/genematrix_noNeg.Rdata" 
infile.3 <- "./data/raw/GSE278586_ProcessedData_LUepic_n499.txt"
infile.4 <- "./data/raw/SCANBrel4_n6614_ExpressedSomaticVariants.csv" 
infile.5 <- "./data/raw/multiomics_cohort_EPIC_TRAIN.RData" 
infile.6 <- "./data/raw/id_multiomics_TEST_cohort_EPICassay.Rdata"
#----------------------------------------------------------------------
# output paths
output.path <- "./data/standardized/"
dir.create(output.path)
outfile.1 <- paste0(output.path,"SCANB_sample_modalities.csv")
outfile.2 <- paste0(output.path,"SCANB_clinical.csv")
outfile.3 <- paste0(output.path,"SCANB_RNAseq_expression.csv")
outfile.4 <- paste0(output.path,"SCANB_RNAseq_mutations.csv")
outfile.5 <- paste0(output.path,"SCANB_DNAmethylation.csv")
#outfile.6 <- paste0(output.path,"SCANB_WGS_mutations.csv")
#outfile.7 <- paste0(output.path,"SCANB_WGS_CNalterations.csv")

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

# annotation 
clin.dat <- loadRData(infile.1)
clin.dat <- clin.dat[clin.dat$Follow.up.cohort == TRUE,]
clin.dat$LN <- ifelse(clin.dat$LN > 0, "N+", "N0")
clin.dat$PR[clin.dat$PR==""] <-NA
#View(clin.dat)
clin.dat <- clin.dat[c("Sample","GEX.assay","ER","PR","HER2","LN",
     "NHG","Size.mm","TreatGroup","InvCa.type",
     "Age","NCN.PAM50",
     "DRFi_days","DRFi_event",
     "OS_days","OS_event",
     "RFi_days","RFi_event")]

# convert to years
clin.dat$OS_years <- clin.dat$OS_days / 365
clin.dat$RFi_years <- clin.dat$RFi_days / 365
clin.dat$DRFi_years <- clin.dat$DRFi_days / 365
clin.dat$OS_days <- NULL
clin.dat$RFi_days <- NULL
clin.dat$DRFi_days <- NULL

# split treatment
clin.dat$TreatGroup[clin.dat$TreatGroup == ""] <- NA
clin.dat$TreatGroup[is.na(clin.dat$TreatGroup)] <- "Missing"
clin.dat$Chemo <- ifelse(grepl("Chemo", clin.dat$TreatGroup), 1, 0)
clin.dat$Endo <- ifelse(grepl("Endo", clin.dat$TreatGroup), 1, 0)
clin.dat$Immu <- ifelse(grepl("Immu", clin.dat$TreatGroup), 1, 0)
#clin.dat$TreatGroup <- NULL

# change this, depending on if we go for HR+HER2- or ER+HER2-
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

#View(head(RNAseq_expr.dat))

#######################################################################
# RNA-seq variant data
#######################################################################

RNAseq_mut.dat <- read.csv(infile.4)
# exclude genes not in my gene expression data
RNAseq_mut.dat <- RNAseq_mut.dat[RNAseq_mut.dat$Gene %in% RNAseq_expr.dat$Gene,]

#######################################################################
# DNA methylation data (n=499)
#######################################################################

#epic.dat <- read.table(infile.3) # too big to load ask johan and then replace the epic.samples file

# Read the first line of the file, mimic later data structure
DNA_methyl.dat <- readLines(infile.3, n = 1)
DNA_methyl.dat <- strsplit(DNA_methyl.dat, "\t")[[1]]
DNA_methyl.dat <- DNA_methyl.dat[!grepl("Detection_Pval", DNA_methyl.dat)]
DNA_methyl.dat[-1] <- sub("\\..*", "", DNA_methyl.dat[-1])
empty.df <- data.frame(matrix(
  ncol = length(DNA_methyl.dat),
  nrow = 1000
))
colnames(empty.df) <- DNA_methyl.dat
empty.df[] <- NA
DNA_methyl.dat <- empty.df

#######################################################################
# MO training cohort
#######################################################################
mo.train <- loadRData(infile.5)
mo.train.ids <- mo.train$SpecimenName

#######################################################################
# MO test cohort
#######################################################################
mo.test <- loadRData(infile.6)
mo.test.ids <- mo.test$SpecimenName

#######################################################################
# Check which samples to add from the methlyation HERE cohort
#######################################################################

# all samples in the test/train MO cohort have methylation data + the ones published in HER2E paper (partial overlap)
length(intersect(names(DNA_methyl.dat)[-1], c(mo.test.ids,mo.train.ids))) # 310 overlap
length(setdiff(names(DNA_methyl.dat)[-1], c(mo.test.ids,mo.train.ids))) # 189 added
#ggVennDiagram(list("HER2E cohort"=names(DNA_methyl.dat)[-1],"MO cohort"=c(mo.test.ids,mo.train.ids)), label_alpha = 0)

# assign the ones not overlapping to train/test sets
added.methyl.samples <- clin.dat[clin.dat$Sample %in% setdiff(names(DNA_methyl.dat)[-1], c(mo.test.ids,mo.train.ids)),]
# consider TreatGroup as this should take the important ClinPath vars into consideration (they are all ER+HER2-)
samp <- createDataPartition(as.factor(added.methyl.samples$TreatGroup), p = 0.75, list = F)
added.methyl.samples.train <- added.methyl.samples[samp,]
added.methyl.samples.test <- added.methyl.samples[-samp,]
#table(added.methyl.samples.train$TreatGroup)
#table(added.methyl.samples.test$TreatGroup)

#######################################################################
# Available modalities overview table
#######################################################################
modalities.dat <- clin.dat[c("Sample")]
modalities.dat$clinical <- 1
modalities.dat$RNAseq_expression <- ifelse(modalities.dat$Sample %in% names(RNAseq_expr.dat),1,0)
modalities.dat$RNAseq_mutations <- ifelse(modalities.dat$Sample %in% names(RNAseq_mut.dat),1,0)
modalities.dat$DNAmethylation <- ifelse(modalities.dat$Sample %in% c(
  mo.test.ids,
  mo.train.ids, added.methyl.samples$Sample
), 1, 0)
# add train/test columns
modalities.dat$TRAIN <- ifelse(modalities.dat$Sample %in% c(
  mo.train.ids,
  added.methyl.samples.train$Sample
), 1, 0)
modalities.dat$TEST <- ifelse(modalities.dat$Sample %in% c(
  mo.test.ids,
  added.methyl.samples.test$Sample
), 1, 0)

#######################################################################
# Exclude samples not used in the study?
#######################################################################
#table(modalities.dat$TRAIN, modalities.dat$TEST)
#table(modalities.dat$DNAmethylation)
# NOTE: there are samples without methylatino data (are neither in test nor training), these will not be used in the study

#######################################################################
# save output files
#######################################################################

write.csv(modalities.dat, file=outfile.1, row.names = FALSE)
write.csv(clin.dat, file=outfile.2, row.names = FALSE)
write.csv(RNAseq_expr.dat, file=outfile.3, row.names = FALSE)
write.csv(RNAseq_mut.dat, file=outfile.4, row.names = FALSE)

# add the columns as if there was 1 methyl dataset
methyl.names <- c("ID_REF",mo.test.ids,mo.train.ids,added.methyl.samples$Sample)
empty.df <- data.frame(matrix(
  ncol = length(methyl.names),
  nrow = 1000
))
colnames(empty.df) <- methyl.names
empty.df[] <- NA
DNA_methyl.dat <- empty.df
write.csv(DNA_methyl.dat, file=outfile.5, row.names = FALSE)