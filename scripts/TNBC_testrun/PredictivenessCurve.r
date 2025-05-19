#!/usr/bin/env Rscript
# Script: Predictiveness curve for risk score
# Author: Lennart Hohmann
# Date: 24.04.2025
#-------------------
# empty environment
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence/")
#-------------------
# packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  data.table, prodlim,
  survival, Publish
)
source("./src/untils.R")
#-------------------
# set/create output directories
output_path <- "./output/CoxNet_manual/"
dir.create(output_path, showWarnings = FALSE)
#-------------------
# input paths
infile_1 <- "./data/raw/tnbc_anno.csv"
#infile_2 <- "./data/raw/tnbc235.csv"
infile_3 <- "./output/CoxNet_manual/risk_scores_from_loaded_model.csv"
#-------------------
# output paths
outfile_1 <- paste0(output_path, "PredictivenessCurve.pdf")
pdf(onefile = TRUE, file = outfile_1, height = 10, width = 15)
#-------------------
# storing objects 
#results <- list() # object to store results

# delete

# f1 = "./data/raw/TCGA_TNBC_betaAdj.RData"
# f2 = "./data/raw/TCGA_TNBC_MergedAnnotations.RData"
# f1 <- loadRData(f1)
# write.csv(f1, file = "./data/raw/TCGA_TNBC_betaAdj.csv", row.names = TRUE)
# f2 <- loadRData(f2)
# str(f2)
# f2$Cibersort.relative <- NULL
# write.csv(f2, file = "./data/raw/TCGA_TNBC_MergedAnnotations.csv", row.names = TRUE)

#-------------------
# storing objects 
#results <- list() # object to store results


#######################################################################
# load data
#######################################################################

clinical <- read.csv(infile_1)
risk_scores <- read.csv(infile_3)
names(risk_scores)[1] <- "PD_ID"
comb <- merge(clinical, risk_scores, by = "PD_ID")

# molec dat
#dat <- data.table::fread(infile_2)
#dat <- as.data.frame(dat)
# Move gene names to rownames
#rownames(dat) <- dat$V1
#dat <- dat[, -1]
#dat[1:5,1:5]
# Ensure clinical data has the same samples as dat
#clinical <- clinical[clinical$PD_ID %in% colnames(dat), ]
# Ensure the samples in clinical data are in the same order as in dat
#clinical <- clinical[match(colnames(dat), clinical$PD_ID), ]
#identical(clinical$PD_ID,colnames(dat))

#######################################################################
# script from aurelien
#######################################################################

# set var names 
comb$time<-comb$RFI
summary(comb$time)
comb$status<-comb$RFIbin
table(comb$status)

hist(comb$risk_score)


km0<-prodlim(Hist(time,status)~1,data=comb)
plot(km0)
comb$risk_score_scaled <- scale(comb$risk_score)
summary(comb$risk_score_scaled)
var(na.omit(comb$risk_score_scaled))
hist(comb$risk_score_scaled)
risk_score<-coxph(Surv(time,status)~risk_score_scaled,data=comb)
publish(risk_score)

# predictiveness of risk score
res1<-get_risk(Surv(time,status)~risk_score_scaled,data=comb,prediction.time = 0)
cutoffRFI <- plotpredictiveness(res1, comb$risk_score_scaled, comb$status)

print(cutoffRFI)
table(comb$risk_score_scaled>cutoffRFI)

gof1<-cox.zph(risk_score)
plot(gof1)


# binarisation of risk score
#
comb$risk_score_scaled_binary<-ifelse(comb$risk_score_scaled>cutoffRFI,1,0)
#
comb$time <- as.numeric(comb$time)
comb$status <- as.integer(comb$status)
comb$risk_score_scaled_binary <- as.integer(comb$risk_score_scaled_binary)


plot(prodlim(Hist(time, status) ~ risk_score_scaled_binary, data = comb))

plot(
  prodlim(Hist(time, status) ~ risk_score_scaled_binary, data = comb),
  legend.x = "bottomright")

title(main = "Stratified by cutoffRFI")

dev.off()
