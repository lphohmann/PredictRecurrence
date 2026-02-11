#!/usr/bin/env Rscript
# Script: Predictiveness curve for risk score cutoff
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
  survival, Publish, survival, survminer
)
source("./src/utils.R")
#-------------------
# set/create output directories
output_path <- "output/CoxNet_200k_simpleCV5/"
dir.create(output_path, showWarnings = FALSE)
#-------------------
# input paths
infile_3 = "output/CoxNet_200k_simpleCV5/SCANB_risk_scores.csv" 
infile_1 = "./data/train/train_clinical.csv" # replace with tnbc dat
#-------------------
# output paths
#outfile_1 <- paste0(output_path, "PredictivenessCurve.pdf")
#pdf(onefile = TRUE, file = outfile_1, height = 10, width = 15)
#-------------------

#######################################################################
# funcs
#######################################################################

## risk: an object class of "get_risk"  
## marker the continuous biomarquer 
## status the event for a binary outcome or status for a time-to-event oucome.
plotpredictiveness <- function(risk,marker,status){
  xlim <- c(0,100)
  breaks = seq(xlim[1], xlim[2], length.out = 5)
  k <- length(risk)
  x1<- sort(marker)
  y1 <- (((1:k)*100)/k)
  z <- sort(risk)
  prev<-sum(status)/length(status)
  Qt<-y1[which.min(abs(z-prev))]
  Tr<-quantile(x1, Qt/100)
  plot(y1,z,type='l',xlab="(%)population \n marker Value",ylab="Risk given marker (%)",main="Estimating the predictiveness curve of a continuous biomarker",
       ylim=c(0,1),axes=FALSE)
  axis(1,at= c(seq(from=0,to=100,by=20),Qt), label = c(seq(from=0,to=100,by=20),round(Qt, 2)))
  axis(2,at=seq(from=0,to=1,by=0.2),labels=seq(from=0,to=1,by=0.2)*100,las=2)
  axis(1,at= c(breaks,Qt), label = round(c(quantile(x1, prob = breaks/100),Tr), 2),pos=-0.26)
  abline(h=prev,col="blue",lty=2)
  abline(v=Qt,col="blue",lty=2)
  return(Threshold=Tr)
}

## a ’formula’ object consisting of outcome ~ markers and covariables 
## The outcome can be binary or a ’Surv’ object for time-to-event outcomes
## data.frame object used to fit and evaluate the model.
## prediction time used only when the outcome is a time-to-event Surv object
get_risk <- function(formula,data,prediction.time=NULL,family){
  event.name <- as.character(formula[[2]])
  if(event.name[[1]] == "Surv"){
    cox <- do.call(coxph,list(formula,data))
    sfit.a0 <- survfit(cox,newdata=data,se.fit = FALSE, conf.type = "none")
    if(prediction.time==0){
      return( c(1-summary(sfit.a0)$surv))
    }
    return( c(1-summary(sfit.a0,times=prediction.time)$surv))
  }
  else{
    yy <- data[[event.name]]
    if(  length(unique(yy))==2){
      outcome = "binary"
      family = binomial(link ="logit")
      #event = data[[as.character(formula[[2]])]]
      myglm<- glm(formula, data = data, family=family)
      return(myglm$fitted.value)
      
    }    
  }  
} 

#######################################################################
# load data
#######################################################################

# stratify patients by risk score tertiles
risk_scores_df <- read.csv(infile_3)
names(risk_scores_df)[1] <- "Sample"

# rows are patient IDs and columns are features (CpGs)
clinical_data_df <- read.csv(infile_1)
clinical_data_df <- clinical_data_df[, c("Sample", "OS_event", "OS_years","RFi_event", "RFi_years")]
head(risk_scores_df)
# merge 
comb <- merge(clinical_data_df, risk_scores_df, by="Sample")
head(comb)

# save
write.csv(comb, file = "output/CoxNet_200k_simpleCV5/SCANB_risk_scores_comb.csv", row.names = FALSE)

#######################################################################
# script from aurelien
#######################################################################

# set var names 
comb$time<-comb$RFi_years
summary(comb$time)
comb$status<-comb$RFi_event
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

risk_score<-coxph(Surv(time,status)~risk_score,data=comb)
publish(risk_score)

# predictiveness of risk score
res1<-get_risk(Surv(time,status)~risk_score_scaled,data=comb,prediction.time = 0)
cutoffRFI_scaled <- plotpredictiveness(res1, comb$risk_score_scaled, comb$status)

comb$risk_score_scaled <- scale(comb$risk_score)
median(comb$risk_score_scaled)
#cutoffRFI_scaled = -0.5326662

#res1<-get_risk(Surv(time,status)~risk_score,data=comb,prediction.time = 0)
#cutoffRFI <- plotpredictiveness(res1, comb$risk_score, comb$status)

#median(comb$risk_score_scaled) #-0.3024001

#cutoffRFI_scaled <- -0.3024001


print(cutoffRFI_scaled)
table(comb$risk_score_scaled>cutoffRFI_scaled)
gof1<-cox.zph(risk_score)
#plot(gof1)

# binarisation of risk score
comb$risk_score_scaled_binary<-ifelse(comb$risk_score_scaled>cutoffRFI_scaled,1,0)
table(comb$risk_score_scaled_binary)

comb$time <- as.numeric(comb$time)
comb$status <- as.integer(comb$status)
comb$risk_score_scaled_binary <- as.integer(comb$risk_score_scaled_binary)

# Save the plot to a variable

fit <- survfit(Surv(time, status) ~ risk_score_scaled_binary, data = comb)

# Save full ggsurvplot object
km_plot <- ggsurvplot(fit,
    data = comb,
    risk.table = TRUE,
    risk.table.y.text.col = TRUE,
    risk.table.y.text = FALSE,
    risk.table.position = "right",
    font.main = c(18, "bold"),
    font.x = c(16, "plain"),
    font.y = c(16, "plain"),
    font.tickslab = c(14, "plain"),
    risk.table.fontsize = 5,
    legend.title = "Risk Group",
    legend.labs = c("Low risk", "High risk"),
    font.legend = c(14, "plain"),
    title = "SCAN-B: DNA-methylation risk stratification in TNBC",
    palette = c("steelblue", "tomato"),
    xlab = "RFI (years)", # custom x-axis label
    ylab = "RFI probability",
    pval = TRUE
) # custom y-axis label)))

# Save to file — use arrange_ggsurvplots to combine main + table
#ggsave("output/CoxNet_200k_simpleCV5/SCANB_KMcurves_Median.png",
#  plot = arrange_ggsurvplots(list(km_plot)),
#  width = 12, height = 6, dpi = 300
#)
#ggsave("output/CoxNet_200k_simpleCV5/SCANB_KMcurves_Median.pdf",
#       plot = arrange_ggsurvplots(list(km_plot)),
#       width = 12, height = 6)
