## Code to generate the results of error vs time for CG/PCG/CHOL methods 
##
## CG - Gaussian processes formulated using linear systems only - all linear systems are solved using standard Conjugate Gradient
## PCG - Gaussian processes formulated using linear systems only - all linear systems are solved using Preconditioned Conjugate Gradient - preconditioning is done using the Nystrom method
## CHOL - Gaussian processes formulated using Cholesky factorizations of large matrices
##
## All methods assume that the covariance matrix K is stored - CG/PCG can easily be adapted to the case where K is not stored (currently not implemented but soon to come)
##
## RBF kernel refers to the isotropic RBF kernel
## ARD kernel refers to the ARD RBF kernel


DATASET = "concrete"
KERNEL_TYPE = "RBF"
NRVECT = 4  ## Number of vectors to unbiasedly approximate the trace term in the gradient of GPs
STEPSIZE = 1.0  ## Stepsize in Adagrad
NSGR_ITERATIONS = 200

for(FOLD in 1:5) {

  for(SOLVER in c("CHOL")) {
    
    if(SOLVER == "CHOL") PREDICTEVERY = 3
    if(SOLVER %in% c("CG", "PCG")) PREDICTEVERY = 5

    ## Depending on the dataset, choose a seed for random number generation and a time budget (which is roughly the amount of seconds after which the simulation should stop)
    if(DATASET == "concrete") {
      MODEL = "regression"
      SEED = 634834
      TIME_BUDGET = 4 * 60 
    }
    
    if(DATASET == "powerplant") {
      MODEL = "regression"
      SEED = 538942
      TIME_BUDGET = 60 * 60
    }
    
    if(DATASET == "protein") {
      MODEL = "regression"
      SEED = 123786
      TIME_BUDGET = 6 * 60 * 60
      if(SOLVER == "CHOL") PREDICTEVERY = 1
    }

    if(DATASET == "credit") {
      MODEL = "classification"
      SEED = 51532
      TIME_BUDGET = 2 * 60
    }
    
    if(DATASET == "spam") {
      MODEL = "classification"
      SEED = 21539
      TIME_BUDGET = 60 * 60
      if(SOLVER == "CHOL") PREDICTEVERY = 1
    }
    
    ## Load functions common to all methods
    source("gp_functions/data_functions.r")
    source("gp_functions/kernel_functions.r")

    ## Load functions relevant to the chosen solver
    if(SOLVER %in% c("CG", "PCG")) {
      source("gp_functions/optimization_functions.r")
      source("gp_functions/conjugate_gradient_functions.r")
    }
    
    if(MODEL == "regression") {
      if(SOLVER == "CHOL") source("gp_functions/gp_regression_chol.r")
      if(SOLVER %in% c("CG", "PCG")) source("gp_functions/gp_regression_cg.r")
    }

    if(MODEL == "classification") {
      if(SOLVER == "CHOL") source("gp_functions/gp_classification_chol.r")
      if(SOLVER %in% c("CG", "PCG")) source("gp_functions/gp_classification_cg.r")  
    }

    ## ******************** Load dataset, normalize the features/labels and create training and test set
    set.seed(SEED)
    
    DATA = create_dataset(DATASET, FOLD, type=MODEL)
    
    if(KERNEL_TYPE == "RBF") NPAR = 3
    if(KERNEL_TYPE == "ARD") NPAR = 2 + DATA$d
    
    SEED = as.integer(SEED/10) + FOLD * 103
    set.seed(SEED)
    
    ## Jitter to add to the diagonal of K
    OMEGA = 1e-6

    ## Define some variables that we will track during execution and save to file at the end of the simulation
    ALL_SAMPLES = matrix(0, NSGR_ITERATIONS, NPAR)
    error_rmse_vs_time = matrix(0, NSGR_ITERATIONS, 2)
    error_neg_llik_vs_time = matrix(0, NSGR_ITERATIONS, 2)
    time_total_rmse = 0
    time_total_neg_llik = 0
    
    ## Initialize to zero variables that will contain solutions to linear systems and cumulative norm of stochastic grad for ADAGRAD
    if(SOLVER %in% c("CG", "PCG")) {
      NITERATIONS <<- 0
      niterations_total = 0
      historical_grad <<- 0
      
      if(MODEL == "regression") {
        K.inv.y <<- rep(0, DATA$n)
        K.inv.r <<- matrix(0, DATA$n, NRVECT)
        K.inv.k_star = matrix(0, DATA$n, DATA$ntest)
      }
      
      if(MODEL == "classification") {
        B.inv.r <<- matrix(0, DATA$n, NRVECT)
        B.inv.w12.k_star = matrix(0, DATA$n, DATA$ntest)
      }
    }
    
    ## Initialization of parameter is not consistent across different models because we setup the experiments at different times
    if(MODEL == "regression") theta0 = c(log(1), log(0.01), rep(log(1/sqrt(DATA$d)), NPAR-2))
    if(MODEL == "classification") theta0 = c(log(10), 0, rep(log(1/2), NPAR-2))

    ## In the original paper submission, for GP regression we initialized the parameters optimizing the log-marginal likelihood on a subset of the data
    if(MODEL == "regression") {
      source("gp_functions/gp_regression_chol.r")
      
      n_subsample = as.integer(sqrt(DATA$n)*4)
      subsample = sample(c(1:(DATA$n)), n_subsample)

      tmp.X = DATA$X
      tmp.y = DATA$y
      tmp.n = DATA$n

      DATA$X = DATA$X[subsample,]
      DATA$y = DATA$y[subsample]
      DATA$n = n_subsample
      
      theta_global = 0
      time_initialize = system.time(tmp <- optim(theta0, log.p.y.theta, gr=grad.log.p.y.theta, method="L-BFGS", control=list(maxit=1000, fnscale=-1)))[3]
      time_total_rmse = time_total_rmse + time_initialize
      time_total_neg_llik = time_total_neg_llik + time_initialize
      
      theta0 = tmp[[1]]

      DATA$X = tmp.X
      DATA$y = tmp.y
      DATA$n = tmp.n

      if(SOLVER %in% c("CG", "PCG")) {
        source("gp_functions/gp_regression_cg.r")

        K.inv.y <<- rep(0, DATA$n)
        K.inv.r <<- matrix(0, DATA$n, NRVECT)
        K.inv.k_star = matrix(0, DATA$n, DATA$ntest)
      }
      
      rm(tmp.X, tmp.y, tmp.n)
    }

    ## theta_global is a variable that keeps track of the current valus of covariance parameters - when this changes, the code updates any quantities needed to make predictions and calculate gradients/stochastic gradients
    theta_global = 0

    ## For CG/PCG computing the negative log-likelihood on test data is more expensive than computing the RMSE. The code computes both at the same time but keeps track of them separately. When the time budget is used up for continuing reporting the NEG_LLIK score, flag_continue_predicting_neg_llik is set to zero and NEG_LLIK is no longer computed.
    flag_continue_predicting_neg_llik = 1

    time_batch_training = 0

    sgr_iteration = 0
    ## Start loop to evaluate prediction error vs iterations
    while(1) {
      sgr_iteration = sgr_iteration + 1

      ## Predict for the current value of theta
      time_batch_predicting_rmse = system.time(assign("predictions", predict_gp_mean_only(theta0)))[3]
      if(MODEL == "regression") error_rmse = sqrt(mean((DATA$ytest - predictions$mu_star)^2))
      if(MODEL == "classification") {
        prediction_labels = c(-1, 1)[(predictions$mu_star > 0)+1]
        error_rmse = mean((DATA$ytest != prediction_labels))
      }
      
      if(flag_continue_predicting_neg_llik == 1) {
        time_batch_predicting_neg_llik = system.time(assign("predictions", predict_gp(theta0)))[3]
        predictions$sigma2_star[predictions$sigma2_star < 1e-12] = 1e-12
        if(MODEL == "regression") error_neg_llik = -sum(dnorm(DATA$ytest, mean=predictions$mu_star, sd=sqrt(predictions$sigma2_star), log=T))
        if(MODEL == "classification") {
          predictions$sigma2_star[predictions$sigma2_star < 1e-12] = 1e-12
          error_neg_llik = -sum(pnorm(DATA$ytest * predictions$mu_star / sqrt(1 + predictions$sigma2_star), log=T))
        }
      }

      ## Keep track of current theta, prediction errors and time
      ALL_SAMPLES[sgr_iteration,] = theta0
      
      time_total_rmse = time_total_rmse + time_batch_predicting_rmse
      time_total_neg_llik = time_total_neg_llik + time_batch_predicting_neg_llik
      
      error_rmse_vs_time[sgr_iteration,] = c(time_total_rmse, error_rmse)
      error_neg_llik_vs_time[sgr_iteration,] = c(time_total_neg_llik, error_neg_llik)
      
      cat(error_rmse, error_neg_llik)
      
      cat("\n")

      if(sgr_iteration > 1) {
        ## Stop if reached the max number of iterations
        if(sgr_iteration == NSGR_ITERATIONS) break
        
        ## Stop if it is foreseen to go beyond the time budget after the next iteration
        if(time_total_neg_llik + time_batch_training + time_batch_predicting_neg_llik > TIME_BUDGET) flag_continue_predicting_neg_llik = 0
        if(time_total_rmse + time_batch_training > TIME_BUDGET) break
       }
      
      cat("EPOCH ",  sgr_iteration, "   ")
      ## If solver is CG/PCG then run ADAGRAD
      if(SOLVER %in% c("CG", "PCG")) {
        time_batch_training = system.time(assign("res", ADAGRAD(theta0, stochastic_gradient_gp, stepsize=STEPSIZE, MAXIT=PREDICTEVERY)))[3]
        theta0 = res[[1]][PREDICTEVERY,]
      }
      
      ## If solver is CHOL then run L-BFGS
      if(SOLVER == "CHOL") {
        time_batch_training = system.time(res <- optim(theta0, log.p.y.theta, gr=grad.log.p.y.theta, method="L-BFGS", control=list(maxit=PREDICTEVERY, fnscale=-1)))[3]
        theta0 = res[[1]]
      }
      
      time_total_rmse = time_total_rmse + time_batch_training
      time_total_neg_llik = time_total_neg_llik + time_batch_training
    }

    if(SOLVER == "CHOL") {
      if(MODEL == "regression") rm(L, K.mat)
      if(MODEL == "classification") rm(L.B.mat, K.mat)
    }

    ## Clean up data to save
    ALL_SAMPLES = ALL_SAMPLES[1:sgr_iteration,]
    if(sgr_iteration < NSGR_ITERATIONS) tmp = min(which(error_rmse_vs_time[,1] == 0)) - 1
    if(sgr_iteration == NSGR_ITERATIONS) tmp = dim(error_rmse_vs_time)[1]
    error_rmse_vs_time = error_rmse_vs_time[1:tmp,]
    if(sgr_iteration < NSGR_ITERATIONS) tmp = min(which(error_neg_llik_vs_time[,1] == 0)) - 1
    if(sgr_iteration == NSGR_ITERATIONS) tmp = dim(error_neg_llik_vs_time)[1]
    error_neg_llik_vs_time = error_neg_llik_vs_time[1:tmp,]

    par(mfrow=c(1,2))
    plot(error_rmse_vs_time[,1], error_rmse_vs_time[,2], type="l", xlab="time", ylab="RMSE")
    plot(error_neg_llik_vs_time[,1], error_neg_llik_vs_time[,2], type="l", xlab="time", ylab="NEG_LLIK")
    
    ## Save results in files named according to the compbination of options chosen
    if(SOLVER %in% c("CG", "PCG")) OPTIONS = paste(DATASET, KERNEL_TYPE, "RMSE", SOLVER, "STEPSIZE", STEPSIZE, "Nr", NRVECT, "PREDICTEVERY", PREDICTEVERY, "FOLD", FOLD, sep="_")
    if(SOLVER == "CHOL") OPTIONS = paste(DATASET, KERNEL_TYPE, "RMSE", "CHOL", "PREDICTEVERY", PREDICTEVERY, "FOLD", FOLD, sep="_")
    
    filesave = paste("results/SAMPLES_", OPTIONS, ".txt", sep="")
    write.table(ALL_SAMPLES, file=filesave, col.names=F, row.names=F, quote=F)
    filesave = paste("results/ERROR_VS_TIME_", OPTIONS, ".txt", sep="")
    write.table(error_rmse_vs_time, file=filesave, col.names=F, row.names=F, quote=F)
    
    if(SOLVER %in% c("CG", "PCG")) OPTIONS = paste(DATASET, KERNEL_TYPE, "NEG_LLIK", SOLVER, "STEPSIZE", STEPSIZE, "Nr", NRVECT, "PREDICTEVERY", PREDICTEVERY, "FOLD", FOLD, sep="_")
    if(SOLVER == "CHOL") OPTIONS = paste(DATASET, KERNEL_TYPE, "NEG_LLIK", "CHOL", "PREDICTEVERY", PREDICTEVERY, "FOLD", FOLD, sep="_")
    filesave = paste("results/SAMPLES_", OPTIONS, ".txt", sep="")
    write.table(ALL_SAMPLES, file=filesave, col.names=F, row.names=F, quote=F)
    filesave = paste("results/ERROR_VS_TIME_", OPTIONS, ".txt", sep="")
    write.table(error_neg_llik_vs_time, file=filesave, col.names=F, row.names=F, quote=F)
    
  }
}
