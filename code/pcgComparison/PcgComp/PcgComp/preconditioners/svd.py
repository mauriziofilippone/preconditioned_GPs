import numpy as np
import time
from sklearn.utils.extmath import randomized_svd
from preconditioner import Preconditioner


"""
Randomized Singular Value Decomposition (SVD) Preconditioner
"""
class SVD(Preconditioner):

	"""
	Construct preconditioning matrix
		X - Training data
		kern - Class of kernel function
		M - Rank of the decomposition
	"""
	def __init__(self, X, kern, M):
		super(SVD, self).__init__("SVD")

		start = time.time()
		self.X = X
		self.kern = kern

		K = kern.K(X, X)
		N = np.shape(X)[0]

		#(self.U, self.Sigma, self.VT) = fb.pca(K, M)#, n_iter=1, l=M)
		self.U, self.Sigma, self.VT = randomized_svd(K, M)
		self.precon = np.dot(self.U, np.dot(np.diag(self.Sigma), self.VT)) + self.kern.noise*np.identity(N)
		self.duration = time.time() - start

	"""
	Compute inversion of the preconditioner.
	"""
	def get_inversion(self):
		N = np.shape(self.X)[0]
		M = np.shape(self.Sigma)[0]
		noise = self.kern.noise
		inv_noise = float(1) / noise
		noise_matrix = noise*np.identity(M)

		# eigs, eigv = np.linalg.eig(np.diag(self.Sigma))
		# for i in xrange(len(eigv)):
		# 	if (eigs[i] < self.kern.jitter):
		# 		eigs[i] = self.kern.jitter
		# 	eigs[i] = np.sqrt(eigs[i])

		eigs = np.sqrt(self.Sigma)
		eigsD = np.diag(eigs)
		left = np.dot(self.U, eigsD)
		right = np.dot(eigsD, self.VT)

		return inv_noise*self.woodbury_inversion(np.identity(N), left, noise_matrix, right)

	"""
	Implementation of Woodbury's matrix inversion lemma.
	"""
	def woodbury_inversion(self, Ainv, U, Cinv, V):
		left_outer = np.dot(Ainv, U)
		right_outer = np.dot(V, Ainv)
		inner = np.linalg.inv(Cinv + np.dot(V, np.dot(Ainv, U)))
		return Ainv - np.dot(left_outer, np.dot(inner, right_outer))

	"""
	Direct computation of (K^-1)b exploiting the matrix inversion lemma.
	"""
	def inv_vec_prod(self, b):
		noise = self.kern.noise
		inv_noise = float(1) / noise
		inv_noise_matrix = inv_noise*np.identity(np.shape(self.X)[0])
		inv_sigma = np.diag(1 / self.Sigma)

		Ainv = inv_noise_matrix
		U = self.U
		Cinv = inv_sigma
		V = self.VT
		right_outer = np.dot(V, np.dot(Ainv, b))
		inner = np.linalg.inv(Cinv + np.dot(V, np.dot(Ainv, U)))
		left_outer = np.dot(Ainv, np.dot(U, np.dot(inner, right_outer)))
		return np.dot(Ainv, b) - left_outer

	"""
	Inversion of preconditioner for Laplace Approximation.
	"""
	def get_laplace_inversion(self, W , Wsqrt):
		self.N = np.shape(self.X)[0]
		self.M = np.shape(self.Sigma)[0]
		
		eigs = np.sqrt(self.Sigma)
		eigsD = np.diag(eigs)
		left = np.dot(self.U, eigsD)
		right = np.dot(eigsD, self.VT)

		return self.laplace_woodbury_inversion(left, right, W.flatten(), Wsqrt.flatten())

	def laplace_woodbury_inversion(self, U, V, W, Wsqrt):
		left_outer = np.dot(np.diag(Wsqrt), U)
		right_outer = np.dot(V, np.diag(Wsqrt))
		inner = np.linalg.inv(np.identity(self.M) + np.dot(V, np.dot(np.diag(W), U)))
		return np.identity(self.N) - np.dot(left_outer, np.dot(inner, right_outer))
