# Script: Cleaning SCAN-B data standardize patient identifiers
# Author: Lennart Hohmann
# Date: 02.03.2025
#-------------------
# empty environment 
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence")
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
infile.1 <- "./data/SCANB/1_clinical/raw/Summarized_SCAN_B_rel4_NPJbreastCancer_with_ExternalReview_Bosch_data.RData"
#infile.2 <- ""
#infile.3 <- ""
#infile.4 <- ""
# output paths
#outfile.1 <- paste0(output.path,"_modalitiesVenn.pdf") #.txt