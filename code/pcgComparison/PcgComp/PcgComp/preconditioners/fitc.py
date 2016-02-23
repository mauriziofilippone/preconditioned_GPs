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
import time
from preconditioner import Preconditioner
from nystrom import Nystrom

"""
Fully-independent Training Conditional (FITC) Preconditioner
"""
class FITC(Preconditioner):

	"""
	Construct preconditioning matrix
		X - Training data
		kern - Class of kernel function
		Xm - Inducing points
	"""
	def __init__(self, X, kern, Xm):
		
		super(FITC, self).__init__("FITC")
		M = np.shape(Xm)[0]
		N = np.shape(X)[0]
		self.kern = kern
		start = time.time()

		k = kern.K(X,X)
		self.nyst = Nystrom(X, kern, Xm, False)
		self.diag = np.diag(k - self.nyst.precon + (kern.noise)*np.identity(N))

		self.precon = self.nyst.precon + np.diag(self.diag)
		self.duration = time.time() - start

	"""
	Compute inversion of the preconditioner.
	"""
	def get_inversion(self):
		inv = 1 / self.diag

		M = np.shape(self.nyst.Km)[0]

		eigs, eigv = np.linalg.eig(self.nyst.KmInv)
		for i in xrange(len(eigv)):
			if (eigs[i] < self.kern.jitter):
				eigs[i] = self.kern.jitter
			eigs[i] = np.sqrt(eigs[i])

		eigsD = np.diag(eigs)
		left = np.dot(self.nyst.Kxm, np.dot(eigv, eigsD))
		right = np.dot(eigsD, np.dot(eigv.T, self.nyst.Kxm.T))

		return self.woodbury_inversion(np.diag(inv), left, np.identity(M), right)

	"""
	Implementation of Woodbury's matrix inversion lemma.
	"""
	def woodbury_inversion(self, Ainv, U, Cinv, V):
		left_outer = np.dot(Ainv, U)
		right_outer = np.dot(V, Ainv)
		inner = np.linalg.inv(Cinv + np.dot(right_outer, U))
		return Ainv - np.dot(np.dot(left_outer, inner), right_outer)

	"""
	Direct computation of (K^-1)b exploiting the matrix inversion lemma.
	"""
	def inv_vec_prod(self, b):
		Ainv = self.Ainv
		U = self.leftU
		Cinv = self.Cinv
		V = self.rightV
		right_outer = np.dot(V, np.dot(Ainv, b))
		if (self.inner is None):
			self.inner = np.linalg.inv(Cinv + np.dot(V, np.dot(Ainv, U)))
		left_outer = np.dot(Ainv, np.dot(U, np.dot(self.inner, right_outer)))
		return np.dot(Ainv, b) - left_outer


	"""
	Inversion of preconditioner for Laplace Approximation.
	"""
	def get_laplace_inversion(self, W, Wsqrt):

		M = np.shape(self.nyst.Km)[0]

		eigs, eigv = np.linalg.eig(self.nyst.KmInv)
		for i in xrange(len(eigv)):
			if (eigs[i] < self.kern.jitter):
				eigs[i] = self.kern.jitter
			eigs[i] = np.sqrt(eigs[i])

		eigsD = np.diag(eigs)
		left = np.dot(np.diag(Wsqrt.flatten()), np.dot(self.nyst.Kxm, np.dot(eigv, eigsD)))
		right = np.dot(eigsD, np.dot(eigv.T, np.dot(self.nyst.Kxm.T, np.diag(Wsqrt.flatten()))))

		A = np.reshape(self.diag,(-1,1))*W + 1
		Ainv = 1/A

		return self.woodbury_inversion(np.diag(Ainv.flatten()), left, np.identity(M), right)
