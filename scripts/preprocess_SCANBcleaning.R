# Script: Cleaning SCAN-B data standardize patient identifiers
# Author: Lennart Hohmann
# Date: 02.03.2025
#-------------------
# empty environment 
rm(list=ls())
# set working directory to project directory
setwd("/Users/le7524ho/PhD_Workspace/PredictRecurrence")
getwd()
#-------------------
# packages
if (!require("pacman")) install.packages("pacman")
#pacman::p_load()
#-------------------
# set/create output directories
#output.path <- "./output/"
#dir.create(output.path)
#-------------------
# input paths
infile.1 <- "./data/raw/Summarized_SCAN_B_rel4_NPJbreastCancer_with_ExternalReview_Bosch_data.RData"
infile.2 <- "./data/raw/genematrix_noNeg.Rdata"
infile.3 <- "./data/raw/GSE278586_ProcessedData_LUepic_n499.txt"
#infile.4 <- "" # rnaseq data
# output paths
#outfile.1 <- paste0(output.path,"_modalitiesVenn.pdf") #.txt

#---------------------
# function: loads RData data file and allows to assign it directly to variable
loadRData <- function(file.path){
  load(file.path)
  get(ls()[ls() != "file.path"])
}
# ---------

clinical.dat <- loadRData(infile.1)

RNAseq.dat <- loadRData(infile.2)

epic.dat <- read.table(infile.3,sep="\t")
head(epic.dat)
print("hi")
