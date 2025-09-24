#!/usr/bin/env Rscript
# Script: Project cohort - Exploring clinical data
# Author: Lennart Hohmann
# Date: 20.03.2025
#-------------------
# empty environment
rm(list=ls())
# set working directory to the project directory
setwd("~/PhD_Workspace/PredictRecurrence/")
#-------------------
# packages
#source("./scripts/src/")
for(pkg in c("ggplot2", "ggVennDiagram", "gridExtra", "naniar")){
    if (!requireNamespace(pkg, quietly = TRUE)) {
        install.packages(pkg)
    }
    library(pkg, character.only = TRUE)
}
#-------------------
# set/create output directories
output_path <- "./output/SCANB_EDA/"
dir.create(output_path)
#-------------------
# input paths
infile_1 <- "./data/train/train_clinical.csv"
infile_2 <- "./data/train/test_clinical.csv"
#infile_1 <- "./data/standardized/SCANB_ProjectCohort/SCANB_PC_clinical.csv"
#-------------------
# output paths
outfile_1 <- paste0(output_path,"SCANB_PC_ExploreClinical.pdf")
#-------------------
# storing objects 
plot.list <- list() # object to store plots
txt.out <- c() # object to store text output, ggf. use capture.output()

#######################################################################
# load data
#######################################################################
# which clin groups to run for
# clin_groups <- c("All", "ER+", "ER+HER2-", "TNBC")
# clin_group <- "ER+HER2-" # All ER+HER2-

clinical.train <- read.csv(infile_1)
clinical.test <- read.csv(infile_2)
clinical <- rbind(clinical.train, clinical.test)
clinical <- clinical.test
clinical$NHG <- as.character(clinical$NHG)

#######################################################################
# Analyses in general PC
#######################################################################

# 1. Pie chart project cohort ClinGroups
clinical$Group <- factor(clinical$Group, levels = c("ER+HER2-", "TNBC", "Other"))

group_counts <- table(clinical$Group)

group_counts <- as.data.frame(group_counts)
group_counts$Percentage <- round((group_counts$Freq / sum(group_counts$Freq)) * 100,0)
# Plot the pie chart using ggplot2
plot <- ggplot(data = group_counts, aes(x = "", y = Freq, fill = Var1)) +
  geom_bar(stat = "identity", width = 1, show.legend = TRUE) +
  coord_polar(theta = "y") + # This makes the chart circular
  labs(title = "ER & HER2 status in Project cohort") +
  theme_void() + # Remove the background grid
  theme(
    legend.position = "none",
    axis.text.x = element_blank(), # Remove x-axis text
    axis.ticks = element_blank(), # Remove axis ticks
    legend.title = element_blank()
  ) + # Remove legend title
  scale_fill_manual(values = c(
    "ER+HER2-" = "#abd9e9",
    #"ER+HER2+" = "#f1b6da",
    #"ER-HER2+" = "#d01c8b",
    "TNBC" = "#d7191c",
    "Other" = "#bababa"
  )) +
  geom_text(aes(label = paste(Var1, "\n", Freq, " (", Percentage, "%)", sep = "")),
    position = position_stack(vjust = 0.5), size = 4
  )

plot.list <- append(plot.list, list(plot))


# 2. Num events in PC
#events <- lapply(list(
#  clinical$OS_event,
#  clinical$RFi_event, 
#  clinical$DRFi_event),
#  table)
#events <- data.frame(
#  Outcome = c("OS","OS","RFi","RFi","DRFi","DRFi"),
#  Count = c(as.vector(events[[1]]),as.vector(events[[2]]),as.vector(events[[3]])), Event = c(0,1,0,1,0,1))
#plot <- ggplot(events, aes(x = Outcome, y = Count, fill = as.factor(Event))) +
#  geom_bar(stat = "identity", position = "stack") +
#  theme(axis.text.x = element_text(angle = 0, hjust = 0.5)) +
#  labs(x = "Outcome", y = "Count", title = "Events in PC") +
#  geom_text(aes(label = Count), position = position_stack(vjust = 0.5)) + # Display count labels inside the bars
#  theme_minimal()

#plot.list <- append(plot.list, list(plot))

# 3. Num events in Group

# Initialize an empty list to store the data
#events_list <- list()

# Iterate through each treatment group
#for (group in unique(clinical$Group)[!is.na(unique(clinical$Group))]) {
#  # Calculate the event counts for OS, RFi, and DRFi for each treatment group
#  events <- lapply(
#    list(
#      clinical$OS_event[clinical$Group == group],
#      clinical$RFi_event[clinical$Group == group],
#      clinical$DRFi_event[clinical$Group == group]
#    ),
#    table
#  )

#  # Prepare the data for the current treatment group
#  events_df <- data.frame(
#    Outcome = rep(c("OS", "RFi", "DRFi"), each = 2), # Repeat OS, RFi, DRFi for 0 and 1 events
#    Count = c(as.vector(events[[1]]), as.vector(events[[2]]), as.vector(events[[3]])), # Count of events
#    Event = rep(c(0, 1), times = 3), # 0 and 1 event counts for each outcome
#    Group = rep(group, 6) # Add the  group as a new column
#  )

  # Append the current  groups data to the events list
#  events_list[[group]] <- events_df
#}

# Combine all data into a single dataframe
#events_combined <- do.call(rbind, events_list)

# Create a stacked barplot for the groups
#plot <- ggplot(events_combined, aes(x = Outcome, y = Count, fill = as.factor(Event))) +
#  geom_bar(stat = "identity", position = "stack") + # Stacked bar plot
#  facet_wrap(~Group) + # Separate the plots by tratment group
#  labs(
#    title = "Events in PC by Group",
#    x = "Outcome", y = "Count", fill = "Event"
#  ) +
#  theme_minimal() + # Use a clean theme
#  theme(axis.text.x = element_text(angle = 0, hjust = 0.5)) + # Adjust x-axis labels
#  geom_text(aes(label = Count), position = position_stack(vjust = 0.5)) # Add labels inside bars

#plot.list <- append(plot.list, list(plot))

#######################################################################
# Analyses to run in all subgroups
#######################################################################

#for (clin_group in clin_groups) {
#  
#  # Subgroup data
#  if (clin_group == "All") {
#    sub_clinical <- clinical
#  } else {
#    sub_clinical <- clinical[grepl(
#      clin_group,
#      clinical$Group,
#      fixed = TRUE # dont treat like regex, otherwise + issues
#    ), ]
#  }

  # Missing data
#  sub_clinical$TreatGroup[sub_clinical$TreatGroup == "Missing"] <- NA
#  plot <- vis_miss(sub_clinical[c("ER", "HER2", "PR", "Age", "PR", "LN", "NHG", "Size.mm", "TreatGroup", "RFi_event", "RFi_years", #"OS_event", "OS_years", "DRFi_event", "DRFi_years")]) +
#    scale_fill_manual(values = c("black", "red")) +
#    theme(
#      axis.text.x = element_text(size = 14, angle = 45, hjust = 0.5),
#      axis.title = element_text(size = 16),
#      axis.text.y = element_text(size = 14)
#    ) +
#    ggtitle(paste0(clin_group," n=",nrow(sub_clinical)))
#  
#  plot.list <- append(plot.list, list(plot))
#}

#######################################################################
# save plots to pdf
#######################################################################


# Set up the PDF output
pdf(file = outfile_1, onefile = TRUE, width = 8.27, height = 11.69)

for(i in 1:length(plot.list)) {
  grid.arrange(plot.list[[i]], ncol = 1, nrow = 2)
  #print(plot.list[[i]])
}

dev.off()

#grid.arrange(plot_1, plot_2, ncol = 2, nrow = 3)
#grid.arrange(plot_3, plot_4, ncol = 1, nrow = 2)
