import numpy as np

"""
Solve linear system using regular preconditioned conjugate gradient
Params:
    K - Covariance Matrix
    Y - Target labels
    P - Preconditioner Matrix (can be set to none)
    init - Initial solution
    threshold - Termintion criteria for outer loop
    preconInv - Inversion of preconditioner Matrix

"""
class RegularPcg(object):

	def __init__(self, K, Y, P, init=None, threshold=1e-9, preconInv=None):
		N = np.shape(K)[0]
		if init is None:
			init = np.zeros((N,1))

		if preconInv is None:
			preconInv = np.linalg.inv(P)

		self.K = K
		self.P = P
		self.Y = Y.flatten()

		x = init
		r = Y - np.dot(K, x) #initialise residual gradient
		z = np.dot(preconInv, r)
		p = z

		outerC = 0

		while True:
			alpha = np.dot(r.T, z) / np.dot(p.T,np.dot(K, p))
			x = x + alpha*p
			r_prev = r
			r = r - alpha*np.dot(K,p)

			if (np.dot(r.T, r).flatten() < threshold*N or outerC>10000):
				break
			z_prev = z
			z = np.dot(preconInv, r)
			beta = np.dot(z.T, r) / np.dot(z_prev.T, r_prev)
			p = z + beta*p
			outerC = outerC + 1
		
		self.iterations = outerC
		self.result = x
