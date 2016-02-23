## This function updates some of the quantities needed to carry out calculations of stochastic gradients and predictions for GP regression
check_update_theta = function(theta) {

  if(crossprod(theta - theta_global) != 0) {
    psi.sigma <<- theta[1]
    psi.lambda <<- theta[2]
    psi.tau <<- theta[-c(1:2)]
    
    theta_global <<- theta
    
    ## Construct K
    if(KERNEL_TYPE == "RBF") k.fun.Kxx(c(psi.sigma, psi.lambda, psi.tau))
    if(KERNEL_TYPE == "ARD") k.fun.Kxx.without.derivatives(c(psi.sigma, psi.lambda, psi.tau))
    
    ## If required, construct the preconditioner
    if(SOLVER == "PCG") {
      n.u <<- 4 * as.integer(sqrt(DATA$n)) 
      ind.u <<-sample(c(1:DATA$n), n.u, replace=T)
      Kuu <<- k.fun.xx(DATA$X[ind.u,], c(psi.sigma, -Inf, psi.tau))
      Kxu <<- k.fun.xy(DATA$X, DATA$X[ind.u,], c(psi.sigma, -Inf, psi.tau))
      
      ABAt = eigen(Kuu, symmetric=T)      
      ABAt[[1]][ABAt[[1]] < OMEGA] = OMEGA
      
      UU <<- crossprod(ABAt[[2]], t(Kxu)) / sqrt(ABAt[[1]])
      
      woodbury_inverse <<- solve(diag(exp(psi.lambda) + OMEGA, n.u) + tcrossprod(UU))
    }
    
    ## solve(K, y)
    K.inv.y <<- conjugate_gradient_solver(DATA$y, K.inv.y)
    niterations_total <<- niterations_total + NITERATIONS
  }
}

## **************************************************************************************************** 
## **************************************************************************************************** 
## ************************************************** Stochastic gradient for GP regression

if(KERNEL_TYPE == "RBF") {
  stochastic_gradient_gp = function(theta) {
    
    check_update_theta(theta)
    
    ## Redraw vectors needed to carry out the stochastic estimate of the trace term in gtilde
    rvect <<- matrix(sample(c(-1,1), DATA$n * NRVECT, replace=T, prob=c(0.5, 0.5)), ncol=NRVECT)
    
    ## solve(K, r^{(i)})
    for(i in 1:NRVECT) {
      K.inv.r[,i] <<- conjugate_gradient_solver(rvect[,i], K.inv.r[,i])
      niterations_total <<- niterations_total + NITERATIONS
    }
    
    ## Compute dK/dtheta_i * y and dK/dtheta_i * K^{-1} r^{(i)}
    dK.dpsi.sigma.times.K.inv.y <<- crossprod(K.mat, K.inv.y) - (exp(psi.lambda) + OMEGA) * K.inv.y 
    dK.dpsi.sigma.times.r <<- crossprod(K.mat, rvect) - (exp(psi.lambda) + OMEGA) * rvect
    
    dK.dpsi.lambda.times.K.inv.y <<- exp(psi.lambda) * K.inv.y
    dK.dpsi.lambda.times.r <<- exp(psi.lambda) * rvect
    
    dK.dpsi.tau.times.K.inv.y <<- crossprod(dK.dpsi.tau, K.inv.y)
    dK.dpsi.tau.times.r <<- crossprod(dK.dpsi.tau, rvect)
    
    ## Compute the stochastic gradient
    gtilde = rep(0, NPAR)
    
    ## Accumulate the contributions from the r^{(i)} vectors
    for(i in 1:NRVECT) {
      gtilde[1] = gtilde[1] + crossprod(K.inv.r[,i], dK.dpsi.sigma.times.r[,i]) 
      gtilde[2] = gtilde[2] + crossprod(K.inv.r[,i], dK.dpsi.lambda.times.r[,i]) 
      gtilde[3] = gtilde[3] + crossprod(K.inv.r[,i], dK.dpsi.tau.times.r[,i]) 
    }
    
    ## Average the contributions from the r^{(i)} vectors and add the remaining part
    gtilde[1] = -0.5 * gtilde[1]/NRVECT + 0.5 * crossprod(K.inv.y, dK.dpsi.sigma.times.K.inv.y)
    gtilde[2] = -0.5 * gtilde[2]/NRVECT + 0.5 * crossprod(K.inv.y, dK.dpsi.lambda.times.K.inv.y)
    gtilde[3] = -0.5 * gtilde[3]/NRVECT + 0.5 * crossprod(K.inv.y, dK.dpsi.tau.times.K.inv.y)
    
    gtilde[gtilde > 1e19] = 1e19
    gtilde[gtilde < -1e19] = -1e19
    
    gtilde
  }
}


if(KERNEL_TYPE == "ARD") {
  stochastic_gradient_gp = function(theta) {
    
    check_update_theta(theta)
    
    ## Redraw vectors needed to carry out the stochastic estimate of the trace term in gtilde
    rvect <<- matrix(sample(c(-1,1), DATA$n * NRVECT, replace=T, prob=c(0.5, 0.5)), ncol=NRVECT)
    
    ## solve(K, r^{(i)})
    for(i in 1:NRVECT) {
      K.inv.r[,i] <<- conjugate_gradient_solver(rvect[,i], K.inv.r[,i])
      niterations_total <<- niterations_total + NITERATIONS
    }
    
    ## Compute dK/dtheta_i * y and dK/dtheta_i * K^{-1} r^{(i)}
    dK.dpsi.sigma.times.K.inv.y <<- crossprod(K.mat, K.inv.y) - (exp(psi.lambda) + OMEGA) * K.inv.y 
    dK.dpsi.sigma.times.r <<- crossprod(K.mat, rvect) - (exp(psi.lambda) + OMEGA) * rvect
    
    dK.dpsi.lambda.times.K.inv.y <<- exp(psi.lambda) * K.inv.y
    dK.dpsi.lambda.times.r <<- exp(psi.lambda) * rvect
    
    dK.dpsi.tau.times.K.inv.y <<- matrix(0, DATA$n, DATA$d)
    dK.dpsi.tau.times.r <<- array(0, c(DATA$n, NRVECT, DATA$d))
    
    for(j in 1:DATA$d) {
      scaled_data = DATA$X[,j] * exp(psi.tau[j]/2) * sqrt(2)
      twice_pairwise_dots = tcrossprod(scaled_data)
      norms_scaled_data = diag(twice_pairwise_dots)/2
      neg_distances2 = twice_pairwise_dots - tcrossprod(norms_scaled_data, rep(1, DATA$n)) - tcrossprod(rep(1, DATA$n), norms_scaled_data)
      dK.dpsi.tau.j = K.mat * neg_distances2
      
      dK.dpsi.tau.times.K.inv.y[,j] <<- crossprod(dK.dpsi.tau.j, K.inv.y)
      dK.dpsi.tau.times.r[,,j] <<- crossprod(dK.dpsi.tau.j, rvect)
    }
    
    ## Compute the stochastic gradient
    gtilde = rep(0, NPAR)
    
    for(i in 1:NRVECT) {
      gtilde[1] = gtilde[1] + crossprod(K.inv.r[,i], dK.dpsi.sigma.times.r[,i]) 
      gtilde[2] = gtilde[2] + crossprod(K.inv.r[,i], dK.dpsi.lambda.times.r[,i])
      gtilde[-c(1:2)] = gtilde[-c(1:2)] + crossprod(K.inv.r[,i], dK.dpsi.tau.times.r[,i,]) 
    }
    
    ## Average the contributions from the r^{(i)} vectors and add the remaining part
    gtilde[1] = -0.5 * gtilde[1]/NRVECT + 0.5 * crossprod(K.inv.y, dK.dpsi.sigma.times.K.inv.y)
    gtilde[2] = -0.5 * gtilde[2]/NRVECT + 0.5 * crossprod(K.inv.y, dK.dpsi.lambda.times.K.inv.y)
    gtilde[-c(1:2)] = -0.5 * gtilde[-c(1:2)]/NRVECT + 0.5 * crossprod(K.inv.y, dK.dpsi.tau.times.K.inv.y)
    
    gtilde[gtilde > 1e19] = 1e19
    gtilde[gtilde < -1e19] = -1e19
    
    gtilde
  }
}


## Predictions for a set of test points - mean only
predict_gp_mean_only = function(theta) {

  check_update_theta(theta)

  k_star = k.fun.xy(DATA$X, DATA$Xtest, theta)
  
  mu_star = crossprod(k_star, K.inv.y)
  
  list(mu_star = mu_star)
}

## Predictions for a set of test points - mean and variance
predict_gp = function(theta) {

  check_update_theta(theta)
  
  k_star = k.fun.xy(DATA$X, DATA$Xtest, theta)
  
  mu_star = crossprod(k_star, K.inv.y)
  
  sigma2_star = rep(0, DATA$ntest)
  for(i in 1:DATA$ntest) {
    K.inv.k_star[,i] <<- conjugate_gradient_solver(k_star[,i], K.inv.k_star[,i])
    niterations_total <<- niterations_total + NITERATIONS
    
    sigma2_star[i] = exp(psi.sigma) + exp(psi.lambda) - crossprod(k_star[,i], K.inv.k_star[,i])
  }
  
  list(mu_star = mu_star, sigma2_star = sigma2_star)
}
