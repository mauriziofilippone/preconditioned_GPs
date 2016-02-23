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
from scipy.linalg import block_diag

"""
Partially-independent Training Conditional (FITC) Preconditioner
"""
class PITC(Preconditioner):

	"""
	Construct preconditioning matrix
		X - Training data
		kern - Class of kernel function
		Xm - Inducing points
	"""
	def __init__(self, X, kern, Xm):
		
		super(PITC, self).__init__("PITC")
		M = np.shape(Xm)[0]
		self.M = M

		start = time.time()
		X_split = np.array_split(X, M)
		self.kern = kern
		kern_blocks = np.zeros((M),dtype=object)

		for t in xrange(M):
			nyst = Nystrom(X_split[t], kern, Xm, False)
			size = np.shape(X_split[t])[0]
			kern_blocks[t] = kern.K(X_split[t], X_split[t]) - nyst.precon  + (kern.noise)*np.identity(size)

		self.blocks = kern_blocks
		blocked = block_diag(*kern_blocks)

		self.nyst = Nystrom(X, kern, Xm, False)
		self.precon = self.nyst.precon + blocked
		self.duration = time.time() - start

	"""
	Compute inversion of the preconditioner.
	"""
	def get_inversion(self):
		invertedBlock = self.get_block_inversion()

		M = np.shape(self.nyst.Km)[0]

		eigs, eigv = np.linalg.eig(self.nyst.KmInv)
		for i in xrange(len(eigv)):
			if (eigs[i] < self.kern.jitter):
				eigs[i] = self.kern.jitter
			eigs[i] = np.sqrt(eigs[i])

		eigsD = np.diag(eigs)
		left = np.dot(self.nyst.Kxm, np.dot(eigv, eigsD))
		right = np.dot(eigsD, np.dot(eigv.T, self.nyst.Kxm.T))

		return self.woodbury_inversion(invertedBlock, left, np.identity(M), right)

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
		inverted_block = self.get_block_inversion()

		Ainv = inverted_block
		U = self.nyst.Kxm
		Cinv = self.nyst.Km
		V = self.nyst.Kxm.T
		right_outer = np.dot(V, np.dot(Ainv, b))
		inner = np.linalg.inv(Cinv + np.dot(V, np.dot(Ainv, U)))
		left_outer = np.dot(Ainv, np.dot(U, np.dot(inner, right_outer)))
		return np.dot(Ainv, b) - left_outer

	"""
	Invert block diagonal matrix block by block.
	"""
	def get_block_inversion(self):
		diag_blocks = self.blocks
		inverted_blocks = np.zeros(len(diag_blocks), dtype=object)
		for i in xrange(len(diag_blocks)):
			inverted_blocks[i] = np.linalg.inv(diag_blocks[i])
		return block_diag(*inverted_blocks)

	"""
	Inversion of preconditioner for Laplace Approximation.
	"""
	def get_laplace_inversion(self, W, Wsqrt):
		inverted_block = self.get_laplace_block_inversion(Wsqrt)

		eigs, eigv = np.linalg.eig(self.nyst.KmInv)
		for i in xrange(len(eigv)):
			if (eigs[i] < self.kern.jitter):
				eigs[i] = self.kern.jitter
			eigs[i] = np.sqrt(eigs[i])

		eigsD = np.diag(eigs)
		left = np.dot(np.diag(Wsqrt.flatten()), np.dot(self.nyst.Kxm, np.dot(eigv, eigsD)))
		right = np.dot(eigsD, np.dot(eigv.T, np.dot(self.nyst.Kxm.T, np.diag(Wsqrt.flatten()))))

		return self.woodbury_inversion(inverted_block, left, np.identity(self.M), right)

	def get_laplace_block_inversion(self, Wsqrt):
		diag_blocks = self.blocks
		Wsqrt_split = np.array_split(Wsqrt, self.M)
		inverted_blocks = np.zeros(len(diag_blocks), dtype=object)
		for i in xrange(len(diag_blocks)):
			Wblock = np.diag(Wsqrt_split[i].flatten())
			block = np.dot(Wblock, np.dot(diag_blocks[i], Wblock))
			inverted_blocks[i] = np.linalg.inv(block + np.identity(len(Wblock)))
		return block_diag(*inverted_blocks)

