## Laplace approximation for GP classification (probit likelihood) without Cholesky decompositions
check_update_theta = function(theta) {

  if(crossprod(theta - theta_global) != 0) {    
    psi.sigma <<- theta[1]
    psi.lambda <<- theta[2]
    psi.tau <<- theta[-c(1:2)]
    
    theta_global <<- theta
    LA(theta)
  }
}
  
LA = function(theta) {
  
  ## Construct K
  if(KERNEL_TYPE == "RBF") k.fun.Kxx(theta)
  if(KERNEL_TYPE == "ARD") k.fun.Kxx.without.derivatives(theta)
  
  if(SOLVER == "PCG") {
    n.u <<- 4 * as.integer(sqrt(DATA$n)) 
    ind.u <<-sample(c(1:DATA$n), n.u, replace=T)
    Kuu <<- k.fun.xx(DATA$X[ind.u,], c(psi.sigma, -Inf, psi.tau))
    Kxu <<- k.fun.xy(DATA$X, DATA$X[ind.u,], c(psi.sigma, -Inf, psi.tau))
    
    ABAt = eigen(Kuu, symmetric=T)
    
    ABAt[[1]][ABAt[[1]] < OMEGA] = OMEGA

    fixed_UU <<- (crossprod(ABAt[[2]], t(Kxu)) / sqrt(ABAt[[1]]))
  }
  
  f.vect = rep(0, DATA$n)
  
  s1 = rep(0, DATA$n)
  converged = F
  for(iiii in 1:DATA$n) {
    pdf.over.cdf = exp(dnorm(f.vect, log=T) - pnorm(DATA$y * f.vect, log=T))
    diag.W.mat <<- c(pdf.over.cdf * (pdf.over.cdf + DATA$y * f.vect))
    sqrt.diag.W.mat <<- sqrt(diag.W.mat)
    
    if(SOLVER == "PCG") {
      UU <<- t(t(fixed_UU) * sqrt.diag.W.mat)
      
      woodbury_inverse <<- solve(diag(n.u) + tcrossprod(UU))
    }
    
    B.mat <<- diag(DATA$n) + t(t(sqrt.diag.W.mat * K.mat) * sqrt.diag.W.mat)
    
    grad.wrt.f <<- DATA$y * pdf.over.cdf

    b.vect = diag.W.mat * f.vect + grad.wrt.f
    s1 = conjugate_gradient_solver(sqrt.diag.W.mat * crossprod(K.mat, b.vect), s1)
    NITERATIONS <<- NITERATIONS + 1
    
    a.vect <<- b.vect - sqrt.diag.W.mat * s1
    
    if(converged == T) break
    
    f.new = crossprod(K.mat, a.vect)
    NITERATIONS <<- NITERATIONS + 1

    f.new[f.new > 100] = 100
    f.new[f.new < -100] = -100

    if(crossprod(f.new - f.vect) < (1e-6 * DATA$n)) converged = T 
    
    f.vect = f.new
  }
  
  f.hat <<- f.vect
}


#### ************************ Compute an unbiased estimate of the gradient of the approximate log-marginal likelihood 
stochastic_gradient_gp = function(theta) {

  check_update_theta(theta)
  
  gtilde = rep(0, NPAR)
  
  ## Redraw vectors needed to carry out the stochastic estimate of the trace term in gtilde and solve(K, r^{(i)})
  rvect <<- matrix(sample(c(-1,1), DATA$n * NRVECT, replace=T, prob=c(0.5, 0.5)), ncol=NRVECT)
  for(i in 1:NRVECT) {
    B.inv.r[,i] <<- conjugate_gradient_solver(rvect[,i], B.inv.r[,i])
    niterations_total <<- niterations_total + NITERATIONS
  }

  ## Compute dK/dtheta_i * a ,  dK/dtheta_i * K^{-1} r^{(i)} , and dK/dtheta_i %*% grad_f{log[p (y | f)]}
  dK.dpsi.sigma.times.a <<- crossprod(K.mat, a.vect) - (OMEGA) * a.vect
  dK.dpsi.sigma.times.w12.r <<- crossprod(K.mat, sqrt.diag.W.mat * rvect) - (OMEGA) * (sqrt.diag.W.mat * rvect)
  dK.dpsi.sigma.times.grad.wrt.f <<- crossprod(K.mat, grad.wrt.f) - (OMEGA) * grad.wrt.f
    
  if(KERNEL_TYPE == "RBF") {
    dK.dpsi.tau.times.a <<- crossprod(dK.dpsi.tau, a.vect)
    dK.dpsi.tau.times.w12.r <<- crossprod(dK.dpsi.tau, sqrt.diag.W.mat * rvect)  
    dK.dpsi.tau.times.grad.wrt.f <<- crossprod(dK.dpsi.tau, grad.wrt.f)
  }
  
  if(KERNEL_TYPE == "ARD") {
    dK.dpsi.tau.times.a <<- matrix(0, DATA$n, DATA$d)
    dK.dpsi.tau.times.grad.wrt.f <<- matrix(0, DATA$n, DATA$d)
    dK.dpsi.tau.times.w12.r <<- array(0, c(DATA$n, NRVECT, DATA$d))

    for(j in 1:DATA$d) {
      scaled_data = DATA$X[,j] * exp(psi.tau[j]/2) * sqrt(2)
      twice_pairwise_dots = tcrossprod(scaled_data)
      norms_scaled_data = diag(twice_pairwise_dots)/2
      neg_distances2 = twice_pairwise_dots - tcrossprod(norms_scaled_data, rep(1, DATA$n)) - tcrossprod(rep(1, DATA$n), norms_scaled_data)
      dK.dpsi.tau.j <<- K.mat * neg_distances2

      dK.dpsi.tau.times.a[,j] <<- crossprod(dK.dpsi.tau.j, a.vect)
      dK.dpsi.tau.times.grad.wrt.f[,j] <<- crossprod(dK.dpsi.tau.j, grad.wrt.f)
      dK.dpsi.tau.times.w12.r[,,j] <<- crossprod(dK.dpsi.tau.j, sqrt.diag.W.mat * rvect)              
    }        
  }
  
  ## (1) First term - accumulate the contributions from the r^{(i)} vectors
  s1 = rep(0, NPAR)
  if(KERNEL_TYPE == "RBF") {
    for(i in 1:NRVECT) {
      s1[1] = s1[1] + crossprod(B.inv.r[,i], sqrt.diag.W.mat * dK.dpsi.sigma.times.w12.r[,i]) 
      s1[3] = s1[3] + crossprod(B.inv.r[,i], sqrt.diag.W.mat * dK.dpsi.tau.times.w12.r[,i]) 
    }
    s1 = -0.5 * s1 / NRVECT
  }
  
  if(KERNEL_TYPE == "ARD") {
    for(i in 1:NRVECT) {
      s1[1] = s1[1] + crossprod(B.inv.r[,i], sqrt.diag.W.mat * dK.dpsi.sigma.times.w12.r[,i]) 
      for(j in 1:DATA$d) s1[j+2] = s1[j+2] + crossprod(B.inv.r[,i], sqrt.diag.W.mat * dK.dpsi.tau.times.w12.r[,i,j]) 
    }
    s1 = -0.5 * s1 / NRVECT
  }
  
  ## (2) Second term - quadratic form in a.vect
  s2 = rep(0, NPAR)
  s2[1] = 0.5 * crossprod(a.vect, dK.dpsi.sigma.times.a)
  if(KERNEL_TYPE == "RBF") s2[3] = 0.5 * crossprod(a.vect, dK.dpsi.tau.times.a)
  if(KERNEL_TYPE == "ARD") s2[-c(1:2)] = 0.5 * crossprod(a.vect, dK.dpsi.tau.times.a)
  
  ## (3) Third term - implicit part of the derivative
  s3 = rep(0, NPAR)
  
  linear_systems_utilde <<- matrix(0, DATA$n, NRVECT)  
  for(i in 1:NRVECT) {
    linear_systems_utilde[,i] <<- conjugate_gradient_solver(sqrt.diag.W.mat * crossprod(K.mat, rvect[,i]), rep(0, DATA$n))
    linear_systems_utilde[,i] <<- crossprod(K.mat, (rvect[,i] - sqrt.diag.W.mat * linear_systems_utilde[,i]))
  }
  
  pdf.over.cdf = exp(dnorm(f.hat, log=T) - pnorm(DATA$y * f.hat, log=T))
  dW.df = 2 * pdf.over.cdf * (-diag.W.mat / DATA$y) + (DATA$y * pdf.over.cdf + f.hat * (-diag.W.mat))
  
  utilde = rep(0, DATA$n)
  for(i in 1:NRVECT) utilde = utilde + linear_systems_utilde[,i] * dW.df * rvect[,i]
  utilde = utilde / NRVECT
  
  s3[1] = -0.5 * crossprod(utilde, conjugate_gradient_solver(sqrt.diag.W.mat * dK.dpsi.sigma.times.grad.wrt.f, rep(0, DATA$n)) / sqrt.diag.W.mat)
  if(KERNEL_TYPE == "RBF") s3[3] = -0.5 * crossprod(utilde, conjugate_gradient_solver(sqrt.diag.W.mat * dK.dpsi.tau.times.grad.wrt.f, rep(0, DATA$n)) / sqrt.diag.W.mat)
  if(KERNEL_TYPE == "ARD") {
    for(j in 1:DATA$d) s3[j+2] = -0.5 * crossprod(utilde, conjugate_gradient_solver(sqrt.diag.W.mat * dK.dpsi.tau.times.grad.wrt.f[,j], rep(0, DATA$n)) / sqrt.diag.W.mat)
  }
  
  ## Final add-up
  gtilde = s1 + s2 + s3
  
  gtilde[gtilde > 1e19] = 1e19
  gtilde[gtilde < -1e19] = -1e19
  
  gtilde
}


## Predictions for a set of test points - mean only
predict_gp_mean_only = function(theta) {
  
  check_update_theta(theta)
  
  k_star = k.fun.xy(DATA$X, DATA$Xtest, theta)
  
  mu_star = crossprod(k_star, a.vect)
  
  list(mu_star = mu_star)
}


## Predictions for a set of test points - mean and variance
predict_gp = function(theta) {

  check_update_theta(theta)
  
  k_star = k.fun.xy(DATA$X, DATA$Xtest, theta)
  
  mu_star = crossprod(k_star, a.vect)
  
  sigma2_star = rep(0, DATA$ntest)
  for(i in 1:DATA$ntest) {
    B.inv.w12.k_star[,i] <<- conjugate_gradient_solver(sqrt.diag.W.mat * k_star[,i], B.inv.w12.k_star[,i])
    niterations_total <<- niterations_total + NITERATIONS
    
    sigma2_star[i] = exp(psi.sigma) - crossprod(sqrt.diag.W.mat * k_star[,i], B.inv.w12.k_star[,i])
  }
  
  list(mu_star = mu_star, sigma2_star = sigma2_star)
}
