## Code to produce plots of error versus time for GPs trained using CG and preconditioned CG

DATASET = "concrete"
DATASET = "powerplant"
DATASET = "protein"
DATASET = "credit"
DATASET = "spam"
DATASET = "eeg"

## KERNEL_TYPE = "RBF"
KERNEL_TYPE = "ARD"



ps.options(width=10, height=8, paper="special", horizontal=F, pointsize=32)
pdf.options(width=10, height=8, pointsize=32)

TIME_IN_LOG = T ## Should the time axis of the plots be in log scale?
if(TIME_IN_LOG) XLAB = expression(log[10](seconds))
if(!TIME_IN_LOG) XLAB = "seconds"
YLAB = list()
if(DATASET %in% c("concrete", "powerplant", "protein")) YLAB[["RMSE"]] = "RMSE"
if(DATASET %in% c("credit", "spam", "eeg")) YLAB[["RMSE"]] = "Error Rate"
YLAB[["NEG_LLIK"]] = "Negative Test Log-Lik"

NAMES_ERROR_MEASURES_GPSTUFF = list()
NAMES_ERROR_MEASURES_GPSTUFF[["RMSE"]] = "MSE"
NAMES_ERROR_MEASURES_GPSTUFF[["NEG_LLIK"]] = "NMLL"

NAMES_KERNEL_TITLE_PLOT = list()
NAMES_KERNEL_TITLE_PLOT[["RBF"]] = "isotropic kernel"
NAMES_KERNEL_TITLE_PLOT[["ARD"]] = "ARD kernel"

NAMES_DATASET_TITLE_PLOT= list()               
NAMES_DATASET_TITLE_PLOT[["concrete"]] = "Concrete"
NAMES_DATASET_TITLE_PLOT[["powerplant"]] = "Power Plant"
NAMES_DATASET_TITLE_PLOT[["protein"]] = "Protein"
NAMES_DATASET_TITLE_PLOT[["credit"]] = "Credit"
NAMES_DATASET_TITLE_PLOT[["spam"]] = "Spam"
NAMES_DATASET_TITLE_PLOT[["eeg"]] = "EEG"

## ## ************************************************** 

NRVECT = 4
if(DATASET == "concrete") {
  STEPSIZE = 1.0
  PREDICTEVERY = 5
  PREDICTEVERY_CHOL = 3
  NFOLDS = 5
}

if(DATASET == "powerplant") {
  STEPSIZE = 1.0
  PREDICTEVERY = 5
  PREDICTEVERY_CHOL = 3
  NFOLDS = 5
}

if(DATASET == "protein") {
  STEPSIZE = 1.0
  PREDICTEVERY = 5
  PREDICTEVERY_CHOL = 1
  NFOLDS = 3
}

if(DATASET == "credit") {
  STEPSIZE = 1.0
  PREDICTEVERY = 5
  PREDICTEVERY_CHOL = 3
  NFOLDS = 5
}

if(DATASET == "spam") {
  STEPSIZE = 1.0
  PREDICTEVERY = 5
  PREDICTEVERY_CHOL = 1
  NFOLDS = 5
}

if(DATASET == "eeg") {
  STEPSIZE = 1.0
  PREDICTEVERY = 5
  PREDICTEVERY_CHOL = 1
  NFOLDS = 3
}


for(ERROR_MEASURE in c("RMSE", "NEG_LLIK")) {
error_vs_time = list()

for(SOLVER in c("PCG", "CG")) {
  ntokeep = Inf
  for(FOLD in 1:NFOLDS) {
      OPTIONS = paste(DATASET, KERNEL_TYPE, ERROR_MEASURE, SOLVER, "STEPSIZE", STEPSIZE, "Nr", NRVECT, "PREDICTEVERY", PREDICTEVERY, "FOLD", FOLD, sep="_")
      ntokeep = min(ntokeep, dim(read.table(paste("results/ERROR_VS_TIME_", OPTIONS, ".txt", sep="")))[1])
    }
  
  error_vs_time[[SOLVER]] = matrix(0, ntokeep, 2)
  for(FOLD in 1:NFOLDS) {
      OPTIONS = paste(DATASET, KERNEL_TYPE, ERROR_MEASURE, SOLVER, "STEPSIZE", STEPSIZE, "Nr", NRVECT, "PREDICTEVERY", PREDICTEVERY, "FOLD", FOLD, sep="_")
      error_vs_time[[SOLVER]] = error_vs_time[[SOLVER]] + read.table(paste("results/ERROR_VS_TIME_", OPTIONS, ".txt", sep=""))[1:ntokeep,] / NFOLDS
    }
}

ntokeep = Inf
for(FOLD in 1:NFOLDS) {
  OPTIONS = paste(DATASET, KERNEL_TYPE, ERROR_MEASURE, "CHOL", "PREDICTEVERY", PREDICTEVERY_CHOL, "FOLD", FOLD, sep="_")
  ntokeep = min(ntokeep, dim(read.table(paste("results/ERROR_VS_TIME_", OPTIONS, ".txt", sep="")))[1])
}

error_vs_time[['CHOL']] = matrix(0, ntokeep, 2)
for(FOLD in 1:NFOLDS) {
  OPTIONS = paste(DATASET, KERNEL_TYPE, ERROR_MEASURE, "CHOL", "PREDICTEVERY", PREDICTEVERY_CHOL, "FOLD", FOLD, sep="_")
  error_vs_time[['CHOL']] = error_vs_time[['CHOL']] + read.table(paste("results/ERROR_VS_TIME_", OPTIONS, ".txt", sep=""))[1:ntokeep,] / NFOLDS
}

if(KERNEL_TYPE == "RBF") {
  if(DATASET == "concrete") base_dir_gpstuff = paste("../pcgComparison/GpStuff Comparison/", KERNEL_TYPE, "/RBF_RESULTS_CONCRETE/", sep="")
  if(DATASET == "powerplant") base_dir_gpstuff = paste("../pcgComparison/GpStuff Comparison/", KERNEL_TYPE, "/RBF_RESULTS_POWER/", sep="")
  if(DATASET == "protein") base_dir_gpstuff = paste("../pcgComparison/GpStuff Comparison/", KERNEL_TYPE, "/RBF_RESULTS_PROTEIN/", sep="")
}

## if(KERNEL_TYPE == "ARD") {
##   if(DATASET == "concrete") base_dir_gpstuff = paste("../pcgComparison/GpStuff Comparison/", KERNEL_TYPE, "/ARD_RESULTS_CONC/", sep="")
##   if(DATASET == "powerplant") base_dir_gpstuff = paste("../pcgComparison/GpStuff Comparison/", KERNEL_TYPE, "/ARD_RESULTS_POWER/", sep="")
##   if(DATASET == "protein") base_dir_gpstuff = paste("../pcgComparison/GpStuff Comparison/", KERNEL_TYPE, "/ARD_RESULTS_PROTEIN/", sep="")
##   if(DATASET == "credit") base_dir_gpstuff = paste("../pcgComparison/GpStuff Comparison/CLASS/CL_RESULTS_CREDIT/", sep="")
##   if(DATASET == "spam") base_dir_gpstuff = paste("../pcgComparison/GpStuff Comparison/CLASS/CL_RESULTS_SPAM/", sep="")
## }

if(KERNEL_TYPE == "ARD") {
  if(DATASET == "concrete") base_dir_gpstuff = paste("../pcgComparison/GpStuff Comparison/post_submission_results/REG_CONC_RES/", sep="")
  if(DATASET == "powerplant") base_dir_gpstuff = paste("../pcgComparison/GpStuff Comparison/post_submission_results/REG_POWER_RES/", sep="")
  if(DATASET == "protein") base_dir_gpstuff = paste("../pcgComparison/GpStuff Comparison/post_submission_results/REG_PROT_RES/", sep="")
  if(DATASET == "credit") base_dir_gpstuff = paste("../pcgComparison/GpStuff Comparison/post_submission_results/CL_CREDIT_RES/", sep="")
  if(DATASET == "spam") base_dir_gpstuff = paste("../pcgComparison/GpStuff Comparison/post_submission_results/CL_SPAM_RES/", sep="")
  if(DATASET == "eeg") base_dir_gpstuff = paste("../pcgComparison/GpStuff Comparison/post_submission_results/CL_EEG_RES/", sep="")  
}

if(TIME_IN_LOG) {
  for(i in 1:length(error_vs_time)) error_vs_time[[i]][,1] = log10(error_vs_time[[i]][,1])
}

error_vs_time[['FITC']] = read.table(paste(base_dir_gpstuff, "FIC_", NAMES_ERROR_MEASURES_GPSTUFF[[ERROR_MEASURE]], ".txt", sep=""))[,-1]
error_vs_time[['PITC']] = read.table(paste(base_dir_gpstuff, "PIC_", NAMES_ERROR_MEASURES_GPSTUFF[[ERROR_MEASURE]], ".txt", sep=""))[,-1]
error_vs_time[['VAR']] = read.table(paste(base_dir_gpstuff, "VAR_", NAMES_ERROR_MEASURES_GPSTUFF[[ERROR_MEASURE]], ".txt", sep=""))[,-1]

xlim = ylim = c(+Inf, -Inf)
for(i in 1:length(error_vs_time)) {
  if(xlim[1] > min(error_vs_time[[i]][,1])) xlim[1] = min(error_vs_time[[i]][,1])
  if(xlim[2] < max(error_vs_time[[i]][,1])) xlim[2] = max(error_vs_time[[i]][,1])
  if(ylim[1] > min(error_vs_time[[i]][,2])) ylim[1] = min(error_vs_time[[i]][,2])
  if(ylim[2] < max(error_vs_time[[i]][,2])) ylim[2] = max(error_vs_time[[i]][,2])
}

MAIN = paste(NAMES_DATASET_TITLE_PLOT[[DATASET]], NAMES_KERNEL_TITLE_PLOT[[KERNEL_TYPE]], sep=" - ")

linetypes = c("F1", "11", "22", "42", "2111", "3111")

pdf(paste("results/PLOT_", DATASET, "_", KERNEL_TYPE, "_", ERROR_MEASURE, ".pdf", sep=""))
par("mar"=c(3.0,3.0,1.1,0.3), "mgp"=c(1.8,0.6,0))
plot(error_vs_time[[1]], col=1, lwd=8, type="l", xlab=XLAB, ylab=YLAB[[ERROR_MEASURE]], xlim=xlim, ylim=ylim, main = MAIN)
for(i in 2:length(error_vs_time)) {
  points(error_vs_time[[i]], col=i, lwd=8, lty=linetypes[i], type="l")
}
## legend(0.6*(max(xlim)-min(xlim))+min(xlim), max(ylim), lwd=8, col=c(1:length(error_vs_time)), legend=names(error_vs_time))
dev.off()

}


## ## ## ## ## ************************************************** Create a legend "box"
## pdf.options(width=9, height=0.7, pointsize=16)
## pdf("results/PLOT_COMPARE_ERROR_VS_TIME_LEGEND.pdf")
## par("mar"=c(0.2,0.2,0.3,0.2), "mgp"=c(0,0,0))
## plot(1, type = "n", axes=FALSE, xlab="", ylab="")
## plot_colors <- c("blue","black", "green", "orange", "pink")
## legend("top", inset=0, lwd=4, col=c(1:length(error_vs_time)), legend=names(error_vs_time), horiz = TRUE, box.lwd=2, lty=linetypes)
## dev.off()
