#This implementation of spectral GP approximation is based on the article:
#
# @article{lazaro2010sparse,
#   title={Sparse spectrum Gaussian process regression},
#   author={L{\'a}zaro-Gredilla, Miguel and Qui{\~n}onero-Candela, Joaquin and Rasmussen, Carl Edward and Figueiras-Vidal, An{\'\i}bal R},
#   journal={The Journal of Machine Learning Research},
#   volume={11},
#   pages={1865--1881},
#   year={2010},
#   publisher={JMLR. org}
# }

import numpy as np
import time
from preconditioner import Preconditioner
from nystrom import Nystrom
from ..util.ssgp import SsgpHelper

"""
Random Fourier Features (Spectral) Preconditioner
"""
class Spectral(Preconditioner):

	"""
	Construct preconditioning matrix
		X - Training data
		Y - Target labels
		kern - Class of kernel function
		M - Number of Fourier features
	"""
	def __init__(self, X, Y, kern, M):
		
		super(Spectral, self).__init__("Spectral")
		start = time.time()
		self.M = M
		self.kern = kern
		[N, D] = X.shape
		self.N = N
		ssgp_helper = SsgpHelper()
		S = ssgp_helper.optimize_frequency_points(X, kern, Y, M, D)

		W = np.reshape(S, (M, D), order='F')

		phi = 2*np.pi*np.dot(X, W.T)
		phi = np.sqrt(kern.variance/float(M))*np.hstack((np.cos(phi), np.sin(phi)))

		A = np.dot(phi, phi.T) + kern.noise*np.identity(N)

		self.precon = A
		self.Kxm = phi
		self.duration = time.time() - start

	"""
	Compute inversion of the Preconditioner
	"""
	def get_inversion(self):
		noise = self.kern.noise
		inv_noise = float(1) / noise
		noise_matrix = noise*np.identity(2*self.M)

		return inv_noise*self.woodbury_inversion(np.identity(self.N), self.Kxm, noise_matrix, self.Kxm.T)

	"""
	Implementation of Woodbury's matrix inversion lemma.
	"""
	def woodbury_inversion(self, Ainv, U, Cinv, V):
		left_outer = np.dot(Ainv, U)
		right_outer = np.dot(V, Ainv)
		inner = np.linalg.inv(Cinv + np.dot(right_outer, U))
		return Ainv - np.dot(np.dot(left_outer, inner), right_outer)
		
	"""
	Inversion of preconditioner for Laplace Approximation.
	"""
	def get_laplace_inversion(self, W, Wsqrt):
		return self.laplace_woodbury_inversion(self.Kxm, self.Kxm.T, W.flatten(), Wsqrt.flatten())

	def laplace_woodbury_inversion(self, U, V, W, Wsqrt):
		left_outer = np.dot(np.diag(Wsqrt), U)
		right_outer = np.dot(V, np.diag(Wsqrt))
		inner = np.linalg.inv(np.identity(2*self.M) + np.dot(V, np.dot(np.diag(W), U)))
		return np.identity(self.N) - np.dot(left_outer, np.dot(inner, right_outer))
