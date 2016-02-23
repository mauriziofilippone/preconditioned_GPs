import numpy as np

"""
Solve linear system using conjugate gradient
Params:
	K - Covariance Matrix
	Y - Target labels
	init - Initial solution
	thershold - Termintion criteria

"""
class Cg(object):

	def __init__(self, K, Y, init=None, threshold=1e-9):
		N = np.shape(K)[0]
		if init is None:
			init = np.zeros((N,1))

		self.K = K
		self.Y = Y.flatten()

		x = init
		r = Y - np.dot(K, x) #initialise residual gradient
		p = r

		t = 0
		while True:
			alpha = np.dot(r.T, r) / np.dot(p.T, np.dot(K, p))
			x = x + alpha*p
			r_prev = r
			r = r - alpha*np.dot(K, p)

			if ((np.dot(r.T,r).flatten() < (threshold*N)) or (t>15000)):
				break
			beta = np.dot(r.T, r) / np.dot(r_prev.T, r_prev)
			p = r + beta*p
			t = t + 1
		
		self.iterations = t
		self.result = x
