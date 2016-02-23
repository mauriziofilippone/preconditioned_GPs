import numpy as np
from ..util.kronHelper import KronHelper
from scipy import sparse
import time

"""
Solve linear system using conjugate gradient (intended for SKI inference)
Params:
    K - Covariance Matrix
    Ws - Sparse representation of weight matrix W
    WTs - Sparse representation of transposed wieght matrix W
    Ku - Array of dimension-specific kernels
    Y - Target labels
    noise - Variance of the likelihood
    init - Initial solution
    threshold - Termintion criteria for algorithm

"""
class KronCgDirect(object):

	def __init__(self, K, Ws, WTs, Ku, Y, noise, init=None, threshold=1e-9):
		N = len(Y)
		if init is None:
			init = np.zeros(N)

		self.Y = Y.flatten()
		x = init
		prod = sparse.csr_matrix.dot(Ws, KronHelper().kron_mvprod(Ku, sparse.csr_matrix.dot(WTs, x))).flatten() + np.dot(noise*np.identity(N), x)
		r = self.Y - prod #initialise residual gradient
		p = r

		t = 1
		while True:
			prod = sparse.csr_matrix.dot(Ws, KronHelper().kron_mvprod(Ku, sparse.csr_matrix.dot(WTs, p))).flatten() + np.dot(noise*np.identity(N), p)
			alpha = np.dot(r.T, r) / np.dot(p.T, prod)
			x = x + np.dot(alpha, p)
			r_prev = r
			r = r - np.dot(alpha, prod)
			if (np.dot(r.T,r).flatten() < threshold*N or t>15000):
				break
			beta = np.dot(r.T, r) / np.dot(r_prev.T, r_prev)
			p = r + np.dot(beta, p)
			t = t + 1
		self.iterations = t
		self.result = x
