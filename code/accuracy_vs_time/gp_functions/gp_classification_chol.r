## Laplace approximation for GP classification (probit likelihood) with Cholesky decompositions
check_update_theta = function(theta) {

  if(crossprod(theta - theta_global) != 0) {    
      psi.sigma <<- theta[1]
      psi.lambda <<- theta[2]
      psi.tau <<- theta[-c(1:2)]

      theta_global <<- theta
      LA_chol(theta)
  }
}

LA_chol = function(theta) {

  check_update_theta(theta)
  
  ## Construct K
  if(KERNEL_TYPE == "RBF") k.fun.Kxx(theta)
  if(KERNEL_TYPE == "ARD") k.fun.Kxx.without.derivatives(theta)
  
  f.vect = rep(0, DATA$n)
  
  converged = F
  for(iiii in 1:DATA$n) {
    pdf.over.cdf = exp(dnorm(f.vect, log=T) - pnorm(DATA$y * f.vect, log=T))
    diag.W.mat <<- c(pdf.over.cdf * (pdf.over.cdf + DATA$y * f.vect))
    sqrt.diag.W.mat <<- sqrt(diag.W.mat)
    
    B.mat <<- diag(DATA$n) + t(t(sqrt.diag.W.mat * K.mat) * sqrt.diag.W.mat)
    L.B.mat <<- t(chol(B.mat))
        
    grad.wrt.f <<- DATA$y * pdf.over.cdf
    
    b.vect = diag.W.mat * f.vect + grad.wrt.f
    s1 = backsolve(t(L.B.mat), forwardsolve(L.B.mat, sqrt.diag.W.mat * crossprod(K.mat, b.vect)))
    
    a.vect <<- b.vect - sqrt.diag.W.mat * s1
    
    if(converged == T) break
    
    f.new = crossprod(K.mat, a.vect)

    f.new[f.new > 100] = 100
    f.new[f.new < -100] = -100
    
    if(crossprod(f.new - f.vect) < (1e-6 * DATA$n)) converged = T 
    
    f.vect = f.new
  }
  
  f.hat <<- f.vect
}

## Laplace approximated log-marginal likelihood
log.p.y.theta = function(theta) {

  print(theta)

  check_update_theta(theta)
  
  - sum(log(diag(L.B.mat))) - 0.5 * drop(crossprod(f.hat, a.vect)) + sum(pnorm(DATA$y * f.hat, log=T))
}

## Gradient of the Laplace approximated log-marginal likelihood 
grad.log.p.y.theta = function(theta) {

  check_update_theta(theta)
  
  B.inv = backsolve(t(L.B.mat), forwardsolve(L.B.mat, diag(1, DATA$n)))

  ## Compute trace(B^{-1} dB/dtheta_i), dK/dtheta_i * a, and dK/dtheta_i %*% grad_f{log[p (y | f)]}
  dK.dpsi.sigma.times.a <<- crossprod(K.mat, a.vect) - (OMEGA) * a.vect
  dK.dpsi.sigma.times.grad.wrt.f <<- crossprod(K.mat, grad.wrt.f) - (OMEGA) * grad.wrt.f
  trace.B.inv.dB.dpsi.sigma <<- sum(B.inv * (t(t(sqrt.diag.W.mat * (K.mat-diag(OMEGA, DATA$n))) * sqrt.diag.W.mat)))
    
  if(KERNEL_TYPE == "RBF") {
    dK.dpsi.tau.times.a <<- crossprod(dK.dpsi.tau, a.vect)
    dK.dpsi.tau.times.grad.wrt.f <<- crossprod(dK.dpsi.tau, grad.wrt.f)
    trace.B.inv.dB.dpsi.tau <<- sum(B.inv * (t(t(sqrt.diag.W.mat * (dK.dpsi.tau)) * sqrt.diag.W.mat)))
  }
  
  if(KERNEL_TYPE == "ARD") {
    dK.dpsi.tau.times.a <<- matrix(0, DATA$n, DATA$d)
    dK.dpsi.tau.times.grad.wrt.f <<- matrix(0, DATA$n, DATA$d)
    trace.B.inv.dB.dpsi.tau <<- rep(0, DATA$d)

    for(j in 1:DATA$d) {
      scaled_data = DATA$X[,j] * exp(psi.tau[j]/2) * sqrt(2)
      twice_pairwise_dots = tcrossprod(scaled_data)
      norms_scaled_data = diag(twice_pairwise_dots)/2
      neg_distances2 = twice_pairwise_dots - tcrossprod(norms_scaled_data, rep(1, DATA$n)) - tcrossprod(rep(1, DATA$n), norms_scaled_data)
      dK.dpsi.tau.j <<- K.mat * neg_distances2

      dK.dpsi.tau.times.a[,j] <<- crossprod(dK.dpsi.tau.j, a.vect)
      dK.dpsi.tau.times.grad.wrt.f[,j] <<- crossprod(dK.dpsi.tau.j, grad.wrt.f)
      trace.B.inv.dB.dpsi.tau[j] <<- sum(B.inv * (t(t(sqrt.diag.W.mat * (dK.dpsi.tau.j)) * sqrt.diag.W.mat)))
    }
  }
  
  ## (1) First term - trace term
  s1 = rep(0, NPAR)
  s1[1] = -0.5 * trace.B.inv.dB.dpsi.sigma
  if(KERNEL_TYPE == "RBF") s1[3] = -0.5 * trace.B.inv.dB.dpsi.tau
  if(KERNEL_TYPE == "ARD") s1[-c(1:2)] = -0.5 * trace.B.inv.dB.dpsi.tau
  
  ## (2) Second term - quadratic form in a.vect
  s2 = rep(0, NPAR)
  s2[1] = 0.5 * crossprod(a.vect, dK.dpsi.sigma.times.a)
  if(KERNEL_TYPE == "RBF") s2[3] = 0.5 * crossprod(a.vect, dK.dpsi.tau.times.a)
  if(KERNEL_TYPE == "ARD") s2[-c(1:2)] = 0.5 * crossprod(a.vect, dK.dpsi.tau.times.a)
  
  ## (3) Third term - implicit part of the derivative
  s3 = rep(0, NPAR)
  
  pdf.over.cdf = exp(dnorm(f.hat, log=T) - pnorm(DATA$y * f.hat, log=T))
  dW.df = 2 * pdf.over.cdf * (-diag.W.mat / DATA$y) + (DATA$y * pdf.over.cdf + f.hat * (-diag.W.mat))
  
  tmp = forwardsolve(L.B.mat, sqrt.diag.W.mat * K.mat)
  diag.D.mat = diag(K.mat - crossprod(tmp))
  
  ## D.mat = diag( K.mat %*% (diag(DATA$n) - diag(sqrt.diag.W.mat) %*% B.inv %*% diag(sqrt(diag.W.mat)) %*% K.mat))
  utilde = diag.D.mat * dW.df
  
  s3[1] = -0.5 * crossprod(utilde, crossprod(B.inv, sqrt.diag.W.mat * dK.dpsi.sigma.times.grad.wrt.f) / sqrt.diag.W.mat)
  if(KERNEL_TYPE == "RBF") s3[3] = -0.5 * crossprod(utilde, crossprod(B.inv, sqrt.diag.W.mat * dK.dpsi.tau.times.grad.wrt.f) / sqrt.diag.W.mat)
  if(KERNEL_TYPE == "ARD") s3[-c(1:2)] = -0.5 * crossprod(utilde, crossprod(B.inv, sqrt.diag.W.mat * dK.dpsi.tau.times.grad.wrt.f) / sqrt.diag.W.mat)
  
  ## Final add-up
  grad = s1 + s2 + s3
  
  grad[grad > 1e19] = 1e19
  grad[grad < -1e19] = -1e19
  
  grad
}


## Predictions for a set of test points - mean only
predict_gp_mean_only = function(theta) {
  
  check_update_theta(theta)
  
  k_star = k.fun.xy(DATA$X, DATA$Xtest, c(psi.sigma, psi.lambda, psi.tau))
  
  mu_star = crossprod(k_star, a.vect)
  
  list(mu_star = mu_star)
}


## Predictions for a set of test points - mean and variance
predict_gp = function(theta) {
  
  check_update_theta(theta)
  
  k_star = k.fun.xy(DATA$X, DATA$Xtest, c(psi.sigma, psi.lambda, psi.tau))
  
  mu_star = crossprod(k_star, a.vect)
  
  tmp = forwardsolve(L.B.mat, sqrt.diag.W.mat * k_star)
    
  sigma2_star = exp(psi.sigma) - apply(tmp^2, 2, sum)
  
  list(mu_star = mu_star, sigma2_star = sigma2_star)
}
