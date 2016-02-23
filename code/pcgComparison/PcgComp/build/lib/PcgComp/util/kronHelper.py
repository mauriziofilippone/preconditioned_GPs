#This implementation is based on the article:
#
# @article{gilboa2015scaling,
#   title={Scaling multidimensional inference for structured Gaussian processes},
#   author={Gilboa, Elad and Saat{\c{c}}i, Yunus and Cunningham, John P},
#   journal={Pattern Analysis and Machine Intelligence, IEEE Transactions on},
#   volume={37},
#   number={2},
#   pages={424--436},
#   year={2015},
#   publisher={IEEE}
# }

import numpy as np

"""
Helper class for methods relying on Kronecker inference.

"""
class KronHelper(object):

	def __init__(self):
		self.name = "KronHelper"

	"""
	Compute array of covariance matrices per grid dimension
		dimVector - Vector of inducing points per dimension
		D - Number of dimensions
		kern - Class of kernel function
	"""
	def kron_inference(self, dimVector, D, kern):

		Kds = np.zeros(D, dtype=object) #vector for holding covariance per dimension
		K_kron = 1 # kronecker product of eigenvalues

		# retrieve the one-dimensional variation of the designated kernel
		for d in xrange(D):
			xg = dimVector[d]
			xg = np.reshape(xg, (len(xg), 1))
			Kds[d] = kern.K_scalar(xg, xg, D)
			#K_kron = np.kron(K_kron, Kds[d])

		return [K_kron, Kds]

	"""
	Fast matrix-vector multiplication for Kronecker matrices
		A - Array of dimension-specific kernels
		b - Vector being multiplied
	"""
	def kron_mvprod(self, A, b):
		x = b
		N = 1
		D = len(A)
		G = np.zeros((D,1))
		for d in xrange(0, D):
			G[d] = len(A[d])
		N = np.prod(G)
		for d in xrange(D-1, -1, -1):
			X = np.reshape(x, (G[d], round(N/G[d])), order='F')
			Z = np.dot(A[d], X)
			Z = Z.T
			x = np.reshape(Z, (-1, 1), order='F')
		return x
