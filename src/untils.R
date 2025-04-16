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


# variance filter for CpGs


# Tumor cell content correction