## Construct and decompose K and compute K^{-1} y  if parameters have changed
check_update_theta = function(theta) {

  if(crossprod(theta - theta_global) != 0) {
      psi.sigma <<- theta[1]
      psi.lambda <<- theta[2]
      psi.tau <<- theta[-c(1:2)]

      theta_global <<- theta
      k.fun.Kxx.without.derivatives(c(psi.sigma, psi.lambda, psi.tau))
      L <<- t(chol(K.mat))
      K.inv.y <<- backsolve(t(L), forwardsolve(L, DATA$y))
  }
}
    
## Exact log-marginal likelihood
log.p.y.theta = function(theta) {
  
  print(theta)

  check_update_theta(theta)
  
  - sum(log(diag(L))) - 0.5 * drop(crossprod(DATA$y, K.inv.y))
}

if(KERNEL_TYPE == "RBF") {
## Exact gradient of the log-marginal likelihood
grad.log.p.y.theta = function(theta) {

  check_update_theta(theta)
      
  K.inv = backsolve(t(L), forwardsolve(L, diag(DATA$n)))
      
  grad = rep(0, NPAR)

  ## Compute dK/dtheta_i * y and dK/dtheta_i * K^{-1} r^{(i)}
  dK.dpsi.sigma = K.mat
  diag(dK.dpsi.sigma) = diag(dK.dpsi.sigma) - exp(psi.lambda) + OMEGA
  dK.dpsi.sigma.times.K.inv.y = crossprod(dK.dpsi.sigma, K.inv.y)
  grad[1] = sum(K.inv * dK.dpsi.sigma)
  
  dK.dpsi.lambda.times.K.inv.y = exp(psi.lambda) * K.inv.y
  grad[2] = sum(diag(K.inv)) * exp(psi.lambda)
  ## grad[2] = sum(K.inv * diag(exp(psi.lambda), DATA$n))

  W = matrix(0, DATA$d, DATA$d); diag(W) = exp(psi.tau/2) * sqrt(2)
  scaled_data = t(tcrossprod(W, DATA$X))
  twice_pairwise_dots = tcrossprod(scaled_data)
  norms_scaled_data = diag(twice_pairwise_dots)/2
  neg_distances2 = twice_pairwise_dots - tcrossprod(norms_scaled_data, rep(1, DATA$n)) - tcrossprod(rep(1, DATA$n), norms_scaled_data)
  dK.dpsi.tau <<- K.mat * neg_distances2

  grad[3] = sum(K.inv * dK.dpsi.tau)

  dK.dpsi.tau.times.K.inv.y = crossprod(dK.dpsi.tau, K.inv.y)
  
  ## Add the trace term and the quadratic form in y
  grad[1] = -0.5 * grad[1] + 0.5 * crossprod(K.inv.y, dK.dpsi.sigma.times.K.inv.y)
  grad[2] = -0.5 * grad[2] + 0.5 * crossprod(K.inv.y, dK.dpsi.lambda.times.K.inv.y)
  grad[3] = -0.5 * grad[3] + 0.5 * crossprod(K.inv.y, dK.dpsi.tau.times.K.inv.y)

  grad[grad > 1e19] = 1e19
  grad[grad < -1e19] = -1e19

  grad
}
}

if(KERNEL_TYPE == "ARD") {

## Exact gradient of the log-marginal likelihood
grad.log.p.y.theta = function(theta) {

  check_update_theta(theta)
      
  K.inv = backsolve(t(L), forwardsolve(L, diag(DATA$n)))
      
  grad = rep(0, NPAR)

  ## Compute dK/dtheta_i * y and dK/dtheta_i * K^{-1} r^{(i)}
  dK.dpsi.sigma = K.mat
  diag(dK.dpsi.sigma) = diag(dK.dpsi.sigma) - exp(psi.lambda) + OMEGA
  dK.dpsi.sigma.times.K.inv.y = crossprod(dK.dpsi.sigma, K.inv.y)
  grad[1] = sum(K.inv * dK.dpsi.sigma)
  
  dK.dpsi.lambda.times.K.inv.y = exp(psi.lambda) * K.inv.y
  grad[2] = sum(diag(K.inv)) * exp(psi.lambda)

  dK.dpsi.tau.times.K.inv.y = matrix(0, DATA$n, DATA$d)
  for(j in 1:DATA$d) {
    scaled_data = DATA$X[,j] * exp(psi.tau[j]/2) * sqrt(2)
    twice_pairwise_dots = tcrossprod(scaled_data)
    norms_scaled_data = diag(twice_pairwise_dots)/2
    neg_distances2 = twice_pairwise_dots - tcrossprod(norms_scaled_data, rep(1, DATA$n)) - tcrossprod(rep(1, DATA$n), norms_scaled_data)
    dK.dpsi.tau.j = K.mat * neg_distances2
      
    grad[j+2] = sum(K.inv * dK.dpsi.tau.j)

    dK.dpsi.tau.times.K.inv.y[,j] = crossprod(dK.dpsi.tau.j, K.inv.y)
  }
  
  ## Add the trace term and the quadratic form in y
  grad[1] = -0.5 * grad[1] + 0.5 * crossprod(K.inv.y, dK.dpsi.sigma.times.K.inv.y)
  grad[2] = -0.5 * grad[2] + 0.5 * crossprod(K.inv.y, dK.dpsi.lambda.times.K.inv.y)
  grad[-c(1:2)] = -0.5 * grad[-c(1:2)] + 0.5 * crossprod(K.inv.y, dK.dpsi.tau.times.K.inv.y)

  grad[grad > 1e19] = 1e19
  grad[grad < -1e19] = -1e19

  grad
}
    
}


## Predictions for a set of test points - mean only
predict_gp_mean_only = function(theta) {

  check_update_theta(theta)
      
  k_star = k.fun.xy(DATA$X, DATA$Xtest, c(psi.sigma, psi.lambda, psi.tau))
  
  mu_star = crossprod(k_star, K.inv.y)

  list(mu_star = mu_star)
}

## Predictions for a set of test points - mean and variance
predict_gp = function(theta) {

  check_update_theta(theta)

  k_star = k.fun.xy(DATA$X, DATA$Xtest, c(psi.sigma, psi.lambda, psi.tau))
  K.inv.k_star = backsolve(t(L), forwardsolve(L, k_star))
  
  mu_star = crossprod(k_star, K.inv.y)

  sigma2_star = rep(0, DATA$ntest)
  for(i in 1:DATA$ntest) {
    sigma2_star[i] = exp(psi.sigma) + exp(psi.lambda) - crossprod(k_star[,i], K.inv.k_star[,i])
  }
  
  list(mu_star = mu_star, sigma2_star = sigma2_star)
}
