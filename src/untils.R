# loads RData data file and allows to assign it directly to variable
loadRData <- function(file.path){
  load(file.path)
  get(ls()[ls() != "file.path"])
}

# convert beta-value to m-value
beta2m <- function(beta) {
  m <- log2(beta/(1-beta))
  return(m)
}

# convert m-value to beta-value
m2beta <- function(m) {
  beta <- 2^m / (2^m + 1)
  return(beta)
}

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

# variance filter for CpGs
# Tumor cell content correction

