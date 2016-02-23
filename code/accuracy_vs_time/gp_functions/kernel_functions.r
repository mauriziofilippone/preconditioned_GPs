## Kernel functions
##
## The kernel function is defined as follows
## k(x, y | theta) = exp(psi.sigma) * exp( - exp(psi.tau) * ||x - y ||^2 )
##
## In the regression case, the variance of the noise is exp(psi.lambda)
## Parameters are combined in the vector theta in the following order (psi.sigma, psi.lambda, psi.tau)
## Note that for simplicity, in classification we retain the same ordering of parameters and psi.lambda and all the derivatives wrt to psi.lambda are set to zero throughout the algorithm
##

## Kernel functions computing the nxn kernel matrix of n training points. The covariance is considered isotropic when w is a scalar or diagonal when w is a vector
## In this version we also compute the derivatives wrt the parameter tau because it is cheap to do it here in the isotropic case
## This code is made so as to exploit linear algebra routines as much as poosible - this is useful in multicore architectures when using optimized linear algebra routines, such as OpenBLAS
## Note that for large matrices the bottleneck is in taking the exp of neg_distances + psi_sigma which does not exploit multithreading. Without much success we tried a few variants to improve on the code provided here to calculate K.
k.fun.Kxx = function(psi) {
  psi.sigma = psi[1]
  psi.lambda = psi[2]
  psi.tau = psi[-c(1:2)]

  W = matrix(0, DATA$d, DATA$d); diag(W) = exp(psi.tau/2) * sqrt(2)

  scaled_data = t(tcrossprod(W, DATA$X))
  twice_pairwise_dots = tcrossprod(scaled_data)
  norms_scaled_data = diag(twice_pairwise_dots)/2
  
  neg_distances2 = twice_pairwise_dots - tcrossprod(norms_scaled_data, rep(1, DATA$n)) - tcrossprod(rep(1, DATA$n), norms_scaled_data)
  ## distances2[distances2 < 0] = 0
  
  K.mat <<- exp(neg_distances2 + psi.sigma)

  if(KERNEL_TYPE == "RBF") dK.dpsi.tau <<- K.mat * neg_distances2
  if(KERNEL_TYPE == "ARD") {
    dK.dpsi.tau <<- array(0, c(DATA$n, DATA$n, DATA$d))
    for(j in 1:DATA$d) {
      scaled_data = DATA$X[,j] * exp(psi.tau[j]/2) * sqrt(2)
      twice_pairwise_dots = tcrossprod(scaled_data)
      norms_scaled_data = diag(twice_pairwise_dots)/2
      neg_distances2 = twice_pairwise_dots - tcrossprod(norms_scaled_data, rep(1, DATA$n)) - tcrossprod(rep(1, DATA$n), norms_scaled_data)
      dK.dpsi.tau[,,j] <<- K.mat * neg_distances2
    }
  }
  
  if(MODEL == "regression") diag(K.mat) <<- exp(psi.sigma) + exp(psi.lambda) + OMEGA
  if(MODEL == "classification") diag(K.mat) <<- exp(psi.sigma) + OMEGA
}

## Same as before but without the derivatives of K wrt tau
k.fun.Kxx.without.derivatives = function(psi) {
  psi.sigma = psi[1]
  psi.lambda = psi[2]
  psi.tau = psi[-c(1:2)]

  W = matrix(0, DATA$d, DATA$d); diag(W) = exp(psi.tau/2) * sqrt(2)

  scaled_data = t(tcrossprod(W, DATA$X))
  twice_pairwise_dots = tcrossprod(scaled_data)
  norms_scaled_data = diag(twice_pairwise_dots)/2
  
  neg_distances2 = twice_pairwise_dots - tcrossprod(norms_scaled_data, rep(1, DATA$n)) - tcrossprod(rep(1, DATA$n), norms_scaled_data)
  ## distances2[distances2 < 0] = 0
  
  K.mat <<- exp(neg_distances2 + psi.sigma)
  
  if(MODEL == "regression") diag(K.mat) <<- exp(psi.sigma) + exp(psi.lambda) + OMEGA
  if(MODEL == "classification") diag(K.mat) <<- exp(psi.sigma) + OMEGA
}

## Same as before but with input data as input to the function - tends to be slower than the previous where X is into a global environment (that here we called DATA)
k.fun.xx = function(x, psi) {
  d = dim(x)[2]
  n = dim(x)[1]
  
  psi.sigma = psi[1]
  psi.lambda = psi[2]
  psi.tau = psi[-c(1:2)]
  
  W = matrix(0, d, d); diag(W) = exp(psi.tau/2) * sqrt(2)

  scaled_data = t(tcrossprod(W, x))
  twice_pairwise_dots = tcrossprod(scaled_data)
  norms_scaled_data = diag(twice_pairwise_dots)/2
  
  neg_distances2 = twice_pairwise_dots - tcrossprod(norms_scaled_data, rep(1, n)) - tcrossprod(rep(1, n), norms_scaled_data)
  
  exp(neg_distances2 + psi.sigma) + diag(exp(psi.lambda) + OMEGA, n)
}

## Computes pairwise covariances between inputs in x and y
k.fun.xy = function(x, y, psi) {
  d = dim(x)[2]
  nx = dim(x)[1]
  ny = dim(y)[1]
  
  psi.sigma = psi[1]
  psi.lambda = psi[2]
  psi.tau = psi[-c(1:2)]
  
  W = matrix(0, d, d); diag(W) = exp(psi.tau/2) * sqrt(2)
  
  scaled_x = t(tcrossprod(W, x))
  scaled_y = t(tcrossprod(W, y))

  norms_scaled_x = tcrossprod(scaled_x^2, t(rep(1, d)))/2
  norms_scaled_y = tcrossprod(scaled_y^2, t(rep(1, d)))/2
  
  twice_pairwise_dots = tcrossprod(scaled_x, scaled_y)

  neg_distances2 = twice_pairwise_dots - tcrossprod(norms_scaled_x, rep(1, ny)) - tcrossprod(rep(1, nx), norms_scaled_y)

  exp(neg_distances2 + psi.sigma)
}
