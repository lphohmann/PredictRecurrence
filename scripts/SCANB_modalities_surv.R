#!/usr/bin/env Rscript
# Script: survival analyses in subsets SCAN-B
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
pacman::p_load(ggplot2, gridExtra, ggfortify,
               survival,
               survminer,
               grid)
               #ggplotify,
               #remotes,
               #stringi)
#-------------------
# set/create output directories
output_path <- "./output/"
dir.create(output_path)
#-------------------
# input paths
infile_1 <- "./data/standardized/SCANB_sample_modalities.csv"
infile_2 <- "./data/standardized/SCANB_clinical.csv"
#-------------------
# which clin group to run for
clin_group <- "ER+HER2-"
#-------------------
# output paths
outfile_1 <- paste0(output_path,"SCANB_Modalities_surv_",clin_group,".pdf")
outfile_2 <- paste0(output_path,"SCANB_Modalities_surv_",clin_group,".txt") 
#-------------------
# storing objects 
plot_list <- list() # object to store plots
txt_out <- c() # object to store text output, ggf. use capture.output()

#######################################################################
# load data
#######################################################################

sample_modalities <- read.csv(infile_1)
clinical <- read.csv(infile_2)

# Subgroup data
if (clin_group == "All") {
  sub_sample_modalities <- sample_modalities
  sub_clinical <- clinical
  } else {
    sub_clinical <- subset(clinical, Group == clin_group)
    sub_sample_modalities <- subset(sample_modalities, Sample %in% sub_clinical$Sample)
}
#names(sub_sample_modalities)

# sub cohort
sub_cohort <- "DNAmethylation"
sub_clinical$Subset <- sub_sample_modalities$DNAmethylation[match(sub_clinical$Sample,sub_sample_modalities$Sample)]
sub_clinical$TreatGroup_ERpHER2n <- ifelse(!(sub_clinical$TreatGroup %in% c("ChemoEndo","Endo","None")),NA,sub_clinical$TreatGroup)

#######################################################################
# Num events in WGS/DNAmethylation subset
#######################################################################

events <- lapply(list(
  sub_clinical$OS_event[sub_clinical$Subset==1],
  sub_clinical$RFi_event[sub_clinical$Subset==1], 
  sub_clinical$DRFi_event[sub_clinical$Subset==1]),
  table)
events <- data.frame(
  Outcome = c("OS","OS","RFi","RFi","DRFi","DRFi"),
  Count = c(as.vector(events[[1]]),as.vector(events[[2]]),as.vector(events[[3]])), Event = c(0,1,0,1,0,1))
plot <- ggplot(events, aes(x = Outcome, y = Count, fill = as.factor(Event))) +
  geom_bar(stat = 'identity', position = 'stack') +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5)) +
  labs(x = 'Outcome', y = 'Count', title = "OS vs. RFI vs. DRFi Events in Subcohort") +
  geom_text(aes(label = Count), position = position_stack(vjust = 0.5)) +  # Display count labels inside the bars
  theme_minimal()
plot_list <- append(plot_list,list(plot))

#######################################################################
# Cox regression and KM curves for Subset
#######################################################################

OM_list <- c("OS_years", "DRFi_years", "RFi_years")
OMbin_list <- c("OS_event", "DRFi_event", "RFi_event")

for(i in seq_along(OM_list)) {
  OM <- OM_list[i]
  OMbin <- OMbin_list[i]

  # data, surv object, and fit
  surv_obj <- Surv(sub_clinical[[OM]], sub_clinical[[OMbin]])
  surv_fit <- survminer::surv_fit(surv_obj~Subset, data=sub_clinical, 
                                conf.type="log-log") 
  # label output
  plot_title <- paste0(OM,"; ",clin_group,"; Subset: ",sub_cohort)
  txt_out <- append(txt_out,
                    c(plot_title, "\n", 
                      paste0("Analyses with clinical endpoint: ",OM, " in ",clin_group," ; ",sub_cohort),"\n###########################################\n"))
  # subtype numbers
  txt_out <- append(txt_out,
                    c(capture.output(table(sub_clinical$Subset)),"\n###########################################\n"))

  # add log rank pval
  txt_out <- append(txt_out,
                    c("log-rank test", "\n", 
                      capture.output(surv_pvalue(surv_fit, data = sub_clinical)),"\n###########################################\n"))

  ##########################

  # uv cox
  main_subset <- coxph(surv_obj~Subset, data=sub_clinical)
  res <- summary(main_subset)
  plot <- ggforest(main_subset,data=sub_clinical,
                  main=plot_title) + theme_bw()

  plot_list <- append(plot_list,list(plot))
  txt_out <- append(txt_out,c(capture.output(res),"\n###########################################\n"))

  ##########################
  pval <- surv_pvalue(surv_fit, data = sub_clinical)
  pvalue_text <- paste("Log-rank p-value=", round(pval$pval,4))
  # KM plot
  plot <- ggsurvplot(surv_fit, 
                    data = sub_clinical,
                    pval = TRUE, conf.int = FALSE,
                    xlab = OM, 
                    break.x.by = 1,
                    break.y.by = 0.05,
                    ylab = paste0(OM," probability"),
                    ylim = c(0.5,1),
                    title = paste(plot_title, "\n", pvalue_text),
                    #legend = c(0.3,0.3),
                    legend.title = "Subset",
                    risk.table = TRUE)#[["plot"]]

  plot_list <- append(plot_list,list(plot))

  ##########################

  # mv cox
  main.all <- coxph(surv_obj~Subset+PR+Age+LN+Size.mm+NHG, 
                    data=sub_clinical) 
  res <- summary(main.all)
  plot <- ggforest(main.all,
                  data=sub_clinical,
                  main=plot_title) + theme_bw()

  plot_list <- append(plot_list,list(plot))
  txt_out <- append(txt_out,c(capture.output(res),"\n###########################################\n"))
}

#######################################################################
# Cox regression and KM curves for Treatment groups
#######################################################################

for(i in seq_along(OM_list)) {
  OM <- OM_list[i]
  OMbin <- OMbin_list[i]

  # data, surv object, and fit
  surv_obj <- Surv(sub_clinical[[OM]], sub_clinical[[OMbin]])
  surv_fit <- survminer::surv_fit(surv_obj~TreatGroup_ERpHER2n, data=sub_clinical, 
                                conf.type="log-log") 
  # label output
  plot_title <- paste0(OM,"; ",clin_group)
  txt_out <- append(txt_out,
                    c(plot_title, "\n", 
                      paste0("Analyses with clinical endpoint: ",OM, " in ",clin_group),"\n###########################################\n"))
  # subtype numbers
  txt_out <- append(txt_out,
                    c(capture.output(table(sub_clinical$TreatGroup_ERpHER2n)),"\n###########################################\n"))

  # add log rank pval
  txt_out <- append(txt_out,
                    c("log-rank test", "\n", 
                      capture.output(surv_pvalue(surv_fit, data = sub_clinical)),"\n###########################################\n"))

  ##########################

  # uv cox
  main_TreatGroup_ERpHER2n <- coxph(surv_obj~TreatGroup_ERpHER2n, data=sub_clinical)
  res <- summary(main_TreatGroup_ERpHER2n)
  plot <- ggforest(main_TreatGroup_ERpHER2n,data=sub_clinical,
                  main=plot_title) + theme_bw()

  plot_list <- append(plot_list,list(plot))
  txt_out <- append(txt_out,c(capture.output(res),"\n###########################################\n"))

  ##########################
  pval <- surv_pvalue(surv_fit, data = sub_clinical)
  pvalue_text <- paste("Log-rank p-value=", pval$pval,4)
  # KM plot
  plot <- ggsurvplot(surv_fit, 
                    data = sub_clinical,
                    pval = TRUE, conf.int = FALSE,
                    xlab = OM, 
                    break.x.by = 1,
                    break.y.by = 0.05,
                    ylab = paste0(OM," probability"),
                    ylim = c(0.5,1),
                    title = paste(plot_title, "\n", pvalue_text),
                    legend = c(0.3,0.4),
                    legend.title = "Treatment",
                    risk.table = TRUE,
                    legend.labs = gsub("TreatGroup_ERpHER2n=","",names(surv_fit$strata)))#[["plot"]]

  plot_list <- append(plot_list,list(plot))

  ##########################

  # mv cox
  main.all <- coxph(surv_obj~TreatGroup_ERpHER2n+PR+Age+LN+Size.mm+NHG, 
                    data=sub_clinical) 
  res <- summary(main.all)
  plot <- ggforest(main.all,
                  data=sub_clinical,
                  main=plot_title) + theme_bw()

  plot_list <- append(plot_list,list(plot))
  txt_out <- append(txt_out,c(capture.output(res),"\n###########################################\n"))
}


#######################################################################
# save plots to pdf
#######################################################################

# Set up the PDF output
# pdf(file = outfile_1, onefile = TRUE)#, height = 8.27/2, width = 11.69/2)

# grid.arrange(plot_1,plot_2,plot_5,plot_6,ncol = 2,nrow = 2)
# print(plot_3)
# print(plot_4)
# dev.off()

# save plots
pdf(file = outfile_1, onefile = TRUE)#, width = 8.3/8, height = 11.7/8) 
for (i in 1:length(plot_list)) {
  print(plot_list[[i]])
}
dev.off()

# save text
writeLines(txt_out, outfile_2)