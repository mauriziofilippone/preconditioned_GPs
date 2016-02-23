## Code implementing ADAGRAD for maximization
ADAGRAD = function(x, gtilde, stepsize, MAXIT=5000, epsilon=1e-6) {

  SAMPLES = matrix(0, MAXIT, NPAR)
  for(i in 1:MAXIT) {
    cat("(", i, ")    ")
    
    grad = gtilde(x)
    
    historical_grad <<- historical_grad + grad^2
    
    deltax = stepsize * grad / (sqrt(historical_grad) + epsilon)
    
    x = x + deltax
    
    SAMPLES[i,] = x

    ## cat("\n")
  }
  
  list(SAMPLES)
}
