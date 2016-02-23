#This implementation is based on the article:
#
# @article{Quinonero-Candela:2005:UVS:1046920.1194909,
#  author = {Qui\~{n}onero-Candela, Joaquin and Rasmussen, Carl Edward},
#  title = {A Unifying View of Sparse Approximate Gaussian Process Regression},
#  journal = {J. Mach. Learn. Res.},
#  issue_date = {12/1/2005},
#  volume = {6},
#  month = dec,
#  year = {2005},
#  issn = {1532-4435},
#  pages = {1939--1959},
#  numpages = {21},
#  url = {http://dl.acm.org/citation.cfm?id=1046920.1194909},
#  acmid = {1194909},
#  publisher = {JMLR.org},
# } 


import numpy as np
from preconditioner import Preconditioner
import time

"""
Nystrom Preconditioner
"""
class Nystrom(Preconditioner):

	"""
	Construct preconditioning matrix
		X - Training data
		kern - Class of kernel function
		Xm - Inducing points
		addNoise - Flag indicating whether to add likelihood variance to kernel matrix
	"""
	def __init__(self, X, kern, Xm, addNoise=True):
		super(Nystrom, self).__init__("Nystrom")

		start = time.time()

		self.kern = kern
		self.X = X
		N = np.shape(X)[0]
		M = np.shape(Xm)[0]
		self.M = M
		self.N = N

		Kxm = kern.K(X, Xm)
		Km = kern.K(Xm, Xm)

		self.Kxm = Kxm
		self.Km = Km + 1e-6*np.identity(M) # jitter
		self.KmInv = np.linalg.inv(self.Km)

		if addNoise:
			self.precon = np.dot(np.dot(Kxm,self.KmInv),Kxm.T) + self.kern.noise*np.identity(N)
		else:
			self.precon = np.dot(np.dot(Kxm,self.KmInv),Kxm.T)

		self.duration = time.time() - start

	"""
	Compute inversion of the preconditioner.
	"""
	def get_inversion(self):
		N = np.shape(self.X)[0]
		M = np.shape(self.Km)[0]
		noise = self.kern.noise
		inv_noise = float(1) / noise
		noise_matrix = noise*np.identity(M)

		eigs, eigv = np.linalg.eig(self.KmInv)
		for i in xrange(len(eigv)):
			if (eigs[i] < self.kern.jitter):
				eigs[i] = self.kern.jitter
			eigs[i] = np.sqrt(eigs[i])

		eigsD = np.diag(eigs)
		left = np.dot(self.Kxm, np.dot(eigv, eigsD))
		right = np.dot(eigsD, np.dot(eigv.T, self.Kxm.T))

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

		Ainv = inv_noise_matrix
		U = self.Kxm
		Cinv = self.Km
		V = self.Kxm.T
		right_outer = np.dot(V, np.dot(Ainv, b))
		inner = np.linalg.inv(Cinv + np.dot(V, np.dot(Ainv, U)))
		left_outer = np.dot(Ainv, np.dot(U, np.dot(inner, right_outer)))
		return np.dot(Ainv, b) - left_outer

	"""
	Inversion of preconditioner for Laplace Approximation.
	"""
	def get_laplace_inversion(self, W, Wsqrt):

		eigs, eigv = np.linalg.eig(self.KmInv)
		for i in xrange(len(eigv)):
			if (eigs[i] < self.kern.jitter):
				eigs[i] = self.kern.jitter
			eigs[i] = np.sqrt(eigs[i])

		eigsD = np.diag(eigs)
		left = np.dot(self.Kxm, np.dot(eigv, eigsD))
		right = np.dot(eigsD, np.dot(eigv.T, self.Kxm.T))

		return self.laplace_woodbury_inversion(left, right, W.flatten(), Wsqrt.flatten())

	def laplace_woodbury_inversion(self, U, V, W, Wsqrt):
		left_outer = np.dot(np.diag(Wsqrt), U)
		right_outer = np.dot(V, np.diag(Wsqrt))
		inner = np.linalg.inv(np.identity(self.M) + np.dot(V, np.dot(np.diag(W), U)))
		return np.identity(self.N) - np.dot(left_outer, np.dot(inner, right_outer))

