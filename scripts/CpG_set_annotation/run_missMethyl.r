#!/usr/bin/env Rscript

# Load packages
suppressMessages({
  library(missMethyl)
  library(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)
  library(ggplot2)
  library(dplyr)
  library(data.table)    # fast CSV reading
})

# -------------------------------
# INPUTS (match your Python script)
# -------------------------------
# Full beta-value CSV (all tested CpGs)
infile_beta <- "./data/train/train_methylation_unadjusted.csv"

# CpG list selected by Cox/LASSO
infile_cpg_set <- "./output/CoxLasso/TNBC/Unadjusted/Selected_model/selected_cpgs.txt"

# Output directory (same as Python pipeline)
output_dir <- "./output/CoxLasso/TNBC/Unadjusted/Selected_model/"
outfile_go_plot   <- file.path(output_dir, "missMethyl_GO_enrichment.png")
outfile_kegg_plot <- file.path(output_dir, "missMethyl_KEGG_enrichment.png")

# -------------------------------
# LOAD DATA
# -------------------------------

# Step 1: Load the full beta-value matrix
beta_dat <- fread(infile_beta)  # large file, fread is fast

# Step 2: Extract all CpG names (column names) to use as background
all_cpgs <- beta_dat$ID_REF

# Selected CpGs
sig_cpgs <- readLines(infile_cpg_set)

# Background = all EPIC CpGs
#all_cpgs <- getAnnotation(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)$Name

# -------------------------------
# RUN ENRICHMENT (bias-corrected)
# -------------------------------

# GO enrichment
gst_go <- gometh(
  sig.cpg   = sig_cpgs,
  all.cpg   = all_cpgs,
  collection= "GO",
  array.type= "EPIC"
)

# KEGG enrichment
gst_kegg <- gometh(
  sig.cpg   = sig_cpgs,
  all.cpg   = all_cpgs,
  collection= "KEGG",
  array.type= "EPIC"
)


# -------------------------------
# FUNCTION TO PLOT TOP TERMS
# -------------------------------
# For GO
#colnames(gst_go)
# For KEGG
#colnames(gst_kegg)
# Convert to data.frame
go_df <- as.data.frame(gst_go)
kegg_df <- as.data.frame(gst_kegg)

# Print top 10 results to console
cat("\nTop GO enrichment:\n")
print(head(go_df, 10))

cat("\nTop KEGG enrichment:\n")
print(head(kegg_df, 10))

library(ggplot2)
library(dplyr)

plot_enrichment <- function(df, title, outfile, top_n=10) {
  df <- as.data.frame(df)
  
  # Determine column for term names
  if ("TERM" %in% colnames(df)) {
    term_col <- "TERM"
  } else if ("Description" %in% colnames(df)) {
    term_col <- "Description"
  } else {
    stop("Cannot find a column with term names for plotting.")
  }
  
  top_df <- df %>%
    arrange(FDR) %>%
    head(top_n) %>%
    mutate(!!term_col := factor(.data[[term_col]], levels = rev(.data[[term_col]])))
  
  # Create plot
  p <- ggplot(top_df, aes(x = !!sym(term_col), y = -log10(FDR))) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(title = title, x = "", y = "-log10(FDR)") +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text.y = element_text(size = 10)
    )
  
  # Save the plot
  ggsave(outfile, plot = p, width = 6, height = 4, dpi = 300)
}


# -------------------------------
# PLOTS
# -------------------------------

plot_enrichment(as.data.frame(gst_go),   "GO Enrichment (missMethyl)",   outfile_go_plot)
plot_enrichment(as.data.frame(gst_kegg), "KEGG Enrichment (missMethyl)", outfile_kegg_plot)

message("missMethyl enrichment plots saved:")
message("  GO:   ", outfile_go_plot)
message("  KEGG: ", outfile_kegg_plot)
