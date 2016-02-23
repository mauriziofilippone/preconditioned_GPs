## Conjugate gradient algorithm
if(SOLVER == "CG") {
  conjugate_gradient_solver = function(b, x0) {
    
    n = DATA$n
    d = DATA$d

    NITERATIONS <<- 0
    
    x = x0
    if(MODEL == "regression") r = b - crossprod(K.mat, x)
    if(MODEL == "classification") r = b - crossprod(B.mat, x)
    NITERATIONS <<- NITERATIONS + 1
    
    p = r
    rsold = drop(crossprod(r))
    
    for (k in 1:n) {
      if(MODEL == "regression") K.mat.p = crossprod(K.mat, p)
      if(MODEL == "classification") K.mat.p = crossprod(B.mat, p)
      NITERATIONS <<- NITERATIONS + 1
      
      alpha = drop(rsold / crossprod(p, K.mat.p))
      
      x = x + alpha * p
      r = r - alpha * K.mat.p
      
      rsnew = drop(crossprod(r))
      if (rsnew < (1e-10 * DATA$n)) break
      
      p = r + p * rsnew / rsold
      
      rsold = rsnew
    }
    
    cat(NITERATIONS, "  ")
    
    x
  }
  
}

## Preconditioned conjugate gradient algorithm with Nystrom preconditioning
if(SOLVER == "PCG") {

  ## The matrices UU and woodbury_inverse are constructed using the Nystrom method every time theta is updated
  if(MODEL == "regression") {  
      nystrom_solve = function(v) {

          lambda = exp(psi.lambda)
          
          s2 = crossprod(UU, crossprod(woodbury_inverse, tcrossprod(UU, t(v))))
          
          (v - s2)/(lambda+OMEGA)
      }
  }


  if(MODEL == "classification") {  
      nystrom_solve = function(v) {
          
          s2 = crossprod(UU, crossprod(woodbury_inverse, tcrossprod(UU, t(v))))
          
          v - s2
      }
  }
  
  conjugate_gradient_solver = function(b, x0) {
    
    n = DATA$n
    d = DATA$d
  
    NITERATIONS <<- 0
    
    x = x0
    if(MODEL == "regression") r = b - crossprod(K.mat, x)
    if(MODEL == "classification") r = b - crossprod(B.mat, x)
    NITERATIONS <<- NITERATIONS + 1
    
    z = nystrom_solve(r)

    p = z
    rsold = drop(crossprod(r, z))
    
    for (k in 1:n) {
      if(MODEL == "regression") K.mat.p = crossprod(K.mat, p)
      if(MODEL == "classification") K.mat.p = crossprod(B.mat, p)
      NITERATIONS <<- NITERATIONS + 1
      
      alpha = drop(rsold / crossprod(p, K.mat.p))
      
      x = x + alpha * p
      r = r - alpha * K.mat.p

      if (crossprod(r) < (1e-10 * DATA$n)) break
      
      z = nystrom_solve(r)
      rsnew = drop(crossprod(r, z))
      
      p = z + p * rsnew / rsold
      
      rsold = rsnew
    }
    
    cat(NITERATIONS, "  ")
    
    x
  }
}
