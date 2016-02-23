## Code containing some tests to check correctness of the code, particularly in the solution of linear systems and in the calculation of the gradients

## Global parameters
KERNEL_TYPE = "ARD"
DATASET = "credit"
SOLVER = "CG"
NRVECT = 5
FOLD = 2
PREDICTEVERY = 2
NSGR_ITERATIONS = 5
SEED = 123

if(DATASET == "concrete") MODEL = "regression"
if(DATASET == "credit") MODEL = "classification"
niterations_total = 0

set.seed(SEED)

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


## ******************** Load dataset
DATA = create_dataset(DATASET, FOLD, type=MODEL)

DATA$n = 50
DATA$X = DATA$X[1:DATA$n,]
DATA$y = DATA$y[1:DATA$n]

if(KERNEL_TYPE == "RBF") NPAR = 3
if(KERNEL_TYPE == "ARD") NPAR = 2 + DATA$d

## Jitter to add to the diagonal
OMEGA = 1e-9

## Initialize the parameter values 
if(MODEL == "regression") theta0 = c(log(1), log(0.01), rep(log(1/(2 * DATA$d)), NPAR-2))
if(MODEL == "classification") theta0 = c(log(10), 0, rep(log(1/(2 * DATA$d)), NPAR-2))

if(MODEL == "regression") {
  K.inv.y <<- rep(0, DATA$n)
  K.inv.r <<- matrix(0, DATA$n, NRVECT)
  K.inv.k_star = matrix(0, DATA$n, DATA$ntest)
}

if(MODEL == "classification") {
  B.inv.r <<- matrix(0, DATA$n, NRVECT)
  B.inv.w12.k_star = matrix(0, DATA$n, DATA$ntest)
}

NITERATIONS = 0

theta_global = 0

cat("\n\n*********************************\n")
cat("**********", KERNEL_TYPE, "CASE\n")
cat("*********************************\n\n")

stochastic_gradient_gp(theta0)

## Test that linear systems are solved correctly
cat("\n*** Test inverse\n")

if(MODEL == "regression") {
  cat("Error linear system with y : "); cat(sum((solve(K.mat, DATA$y) - K.inv.y)^2), "\n")
  for(i in 1:NRVECT) {
    cat("Error linear system with r", i, " : ", sep="");
    cat(sum((solve(K.mat, rvect[,i]) - K.inv.r[,i])^2), "\n")
  }
}

if(MODEL == "classification") {
  cat("Error linear system K^(-1) f.hat : "); cat(sum((solve(K.mat, f.hat) - a.vect)^2), "\n")
  for(i in 1:NRVECT) {
    cat("Error linear system with r", i, " : ", sep="");
    cat(sum((solve(B.mat, rvect[,i]) - B.inv.r[,i])^2), "\n")
  }
}

## Test that the average of stochastic gradients is close to the exact gradient
cat("\n*** Test stochastic gradient\n")
NREP = 100
gtilde = rep(0, NPAR)
for(i in 1:NREP) {
    theta_global = 0
    gtilde = gtilde + stochastic_gradient_gp(theta0)
}
gtilde = gtilde / NREP

if(MODEL == "regression") source("gp_functions/gp_regression_chol.r")
if(MODEL == "classification") source("gp_functions/gp_classification_chol.r")
theta_global = 0
grad_loglik = grad.log.p.y.theta(theta0)

cat("\nExact gradient", grad_loglik, "Stochastic gradient", gtilde)
cat("\nNorm difference between average stochastic gradient and exact gradient: ", sep="");
cat(crossprod(gtilde - grad_loglik), "\n")


## Test that the average of stochastic gradients is close to the exact gradient
cat("\n*** Test exact gradient to ensure that the derivatives of K are correct\n")

theta_global = 0
py.theta = log.p.y.theta(theta0)

epsilon = 1e-6

for(i in 1:NPAR) {
    delta = rep(0, NPAR)
    delta[i] = epsilon
    theta = theta0 + delta
    py.theta.plus.epsilon = log.p.y.theta(theta)

    cat("\nError exact gradient component", i, ":", (py.theta.plus.epsilon - py.theta)/epsilon - grad_loglik[i], "\n")
}
