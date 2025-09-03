#!/usr/bin/env Rscript
# Script: survival analyses in TNBC subset SCAN-B methylation FU cohort
# Author: Lennart Hohmann 
# Date: 19.03.2025
#-------------------
# empty environment
rm(list=ls())
setwd("~/PhD_Workspace/PredictRecurrence/")

#-------------------
# packages
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("gridExtra")) install.packages("gridExtra")
if (!require("ggfortify")) install.packages("ggfortify")
if (!require("survival")) install.packages("survival")
if (!require("survminer")) install.packages("survminer")
if (!require("grid")) install.packages("grid")

#remove.packages(c("survminer", "survival"))

#-------------------
# set/create output directories
output_path <- "./output/SCANB_EDA/"
dir.create(output_path, showWarnings = FALSE)

#-------------------
# input paths
infile_0 <- "./data/set_definitions/train_subcohorts/TNBC_train_ids.csv"
infile_1 <- "./data/train/train_clinical.csv"
#-------------------
# output paths
outfile_1 <- paste0(output_path,"SCANB_TNBC_surv.pdf")
outfile_2 <- paste0(output_path,"SCANB_TNBC_surv.txt")

#-------------------
# storing objects
plot_list <- list()
txt_out <- c()

#######################################################################
# load data
#######################################################################

train_ids <- read.csv(infile_0,header=FALSE)[[1]]
clinical <- read.csv(infile_1)
clinical <- clinical[clinical$Sample %in% train_ids,]

clinical$NHG <- as.character(clinical$NHG)
clinical$LN <- factor(clinical$LN, levels = c("N0","N+"))
sub_clinical <- clinical

#######################################################################
# Cox regression and KM curves for TNBC
#######################################################################

OM_list <- c("OS_years", "RFi_years")
OMbin_list <- c("OS_event", "RFi_event")

for (i in seq_along(OM_list)) {
  OM <- OM_list[i]
  OMbin <- OMbin_list[i]

  surv_obj <- Surv(sub_clinical[[OM]], sub_clinical[[OMbin]])
  surv_fit <- survminer::surv_fit(surv_obj ~ 1, # no grouping within TNBC
                                  data = sub_clinical,
                                  conf.type = "log-log")

  plot_title <- paste0(OM)
  txt_out <- append(txt_out, c(
    plot_title, "\n",
    paste0("Analyses with clinical endpoint: ", OM),
    "\n###########################################\n"
  ))

  # -------------------------------
  # Print cumulative number of events up to selected time points
  # -------------------------------
  time_points <- seq(0, ceiling(max(sub_clinical[[OM]])), by = 1)
  surv_summary <- summary(surv_fit, times = time_points)

  # cumulative events
  cum_events <- cumsum(surv_summary$n.event)

  event_table <- data.frame(
  time = surv_summary$time,
  n_risk = surv_summary$n.risk,
  cum_event = cum_events
  )

  txt_out <- append(txt_out, c(
  "Cumulative events up to selected time points:\n",
  capture.output(event_table),
  "\n###########################################\n"
  ))

  # KM curve (single group)
  plot <- ggsurvplot(surv_fit,
                     data = sub_clinical,
                     pval = FALSE, conf.int = FALSE,
                     xlab = OM,
                     break.x.by = 1,
                     break.y.by = 0.05,
                     ylab = paste0(OM, " probability"),
                     ylim = c(0.5, 1),
                     title = plot_title,
                     risk.table = TRUE)
  plot_list <- append(plot_list, list(plot))
}

#######################################################################
# save outputs
#######################################################################

pdf(file = outfile_1, onefile = TRUE)
for (i in 1:length(plot_list)) {
  print(plot_list[[i]])
}
dev.off()

writeLines(txt_out, outfile_2)
