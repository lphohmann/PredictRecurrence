impute_clinical_data <- function(data, 
                                  vars_to_impute,
                                  complete_vars,
                                  binary_vars = NULL,
                                  ordered_vars = NULL,
                                  maxit = 10,
                                  seed = 13) {
  
  all_vars <- c(vars_to_impute, complete_vars)

  # Print missingness before
  cat("\n=== Missingness BEFORE imputation ===\n")
  missing_counts <- colSums(is.na(data[vars_to_impute]))
  print(missing_counts)
  cat("Total missing:", sum(missing_counts), "\n")
  
  # Prepare data for MICE
  data_for_mice <- data[, all_vars, drop = FALSE]
  
  # Set up imputation methods
  # Default is PMM for continuous variables
  methods <- make.method(data_for_mice)
  
  # Override methods for categorical variables
  if (!is.null(binary_vars)) {
    for (var in binary_vars) {
      if (var %in% names(methods)) {
        methods[var] <- "logreg"
      }
    }
  }
  
  if (!is.null(ordered_vars)) {
    for (var in ordered_vars) {
      if (var %in% names(methods)) {
        methods[var] <- "polr"
      }
    }
  }
  
  # Print methods being used
  cat("\nImputation methods:\n")
  impute_methods <- methods[vars_to_impute]
  impute_methods <- impute_methods[impute_methods != ""]  # Remove empty ones
  print(impute_methods)
  
  # Run MICE
  cat("\nRunning MICE imputation (maxit =", maxit, ")...\n")
  
  mice_result <- mice(data_for_mice, 
                      m = 1,
                      maxit = maxit,
                      method = methods,
                      seed = seed,
                      printFlag = FALSE)
  
  # Check for logged events (problems during imputation)
  if (!is.null(mice_result$loggedEvents)) {
    cat("\nWARNING: MICE logged", nrow(mice_result$loggedEvents), "events:\n")
    print(mice_result$loggedEvents)
  }
  
  # Get imputed data
  imputed_data <- complete(mice_result, action = 1)
  
  # Replace missing values in original data
  result <- data
  for (var in vars_to_impute) {
    result[[var]][is.na(data[[var]])] <- imputed_data[[var]][is.na(data[[var]])]
  }
  
  # Print missingness after
  cat("\n=== Missingness AFTER imputation ===\n")
  missing_counts_after <- colSums(is.na(result[vars_to_impute]))
  print(missing_counts_after)
  cat("Total missing:", sum(missing_counts_after), "\n")
  
  # Warn if any variables still have missing values
  still_missing <- vars_to_impute[missing_counts_after > 0]
  if (length(still_missing) > 0) {
    cat("\nWARNING: These variables still have missing values:\n")
    print(missing_counts_after[still_missing])
    cat("Consider adding more predictors or using fallback imputation.\n")
  }
  
  cat("\n")
  return(result)
}

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



log_msg <- function(msg) {
  cat("\n=== ", msg, " ===\n", sep = "")
}

variance_filter <- function(df, top_n = NULL, min_var = NULL) {
  vars <- apply(df, 2, var)
  if (!is.null(top_n)) {
    keep <- names(sort(vars, decreasing = TRUE)[1:top_n])
  } else if (!is.null(min_var)) {
    keep <- names(vars[vars >= min_var])
  } else {
    stop("Must specify top_n or min_var")
  }
  df[, keep, drop = FALSE]
}

load_training_data <- function(train_ids, beta_path, clinical_path) {
  clinical <- fread(clinical_path) %>% column_to_rownames("Sample")
  clinical <- clinical[train_ids, , drop = FALSE]
  
  beta <- fread(beta_path, data.table = FALSE)
  rownames(beta) <- beta[[1]]
  beta <- t(beta[, -1])
  beta <- beta[train_ids, , drop = FALSE]
  
  list(beta = beta, clinical = clinical)
}

preprocess_data <- function(beta, top_n_cpgs) {
  log_msg(sprintf("Preprocessing: Converting to M-values and selecting top %d CpGs", top_n_cpgs))
  mvals <- beta2m(beta)
  variance_filter(mvals, top_n = top_n_cpgs)
}


