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

from __future__ import division
import numpy as np
import random as ran
from scipy.spatial.distance import cdist, squareform, pdist
from scipy.optimize import fmin_l_bfgs_b
from scipy.linalg import cholesky
import numpy.matlib,numpy.linalg
import math

"""
Helper class for methods based on Fourier features.

"""
class SsgpHelper(object):

	def __init__(self):
		self.name = "SsgpHelper"

	"""
	Evaluate likelihood for approximation usinf random fourier features.
	"""
	def ssgpr(self, X, kern, S, Y):
		[N, D] = X.shape
		m = len(S)/D
		W = np.reshape(S, (m, D), order='F')

		phi = np.dot(X, W.T)
		phi = np.hstack((np.cos(phi), np.sin(phi)))

		A = np.dot(phi.T, phi) + kern.noise*np.identity(2*m)
		R = cholesky(A, lower=False)
		PhiRi = np.linalg.lstsq(R.T, phi.T)[0] # PhiRi = phi/R
		Rtphity = np.dot(PhiRi, Y.flatten())

		return 0.5/kern.noise*(np.sum(np.power(Y,2))-kern.noise/m*np.sum(np.power(Rtphity,2))) + np.sum(np.log(np.diag(R))) + (N/2 - m)*np.log(kern.noise)+N/2*np.log(2*np.pi)

	"""
	Optimize random selection of frequency points by taking the set which maximises the likelihood 
	over a series of iterations.
	"""
	def optimize_frequency_points(self, X, kern, Y, M, D):
		nlml = np.inf
		for k in xrange(5):
			#S = np.random.randn(M*D)
			S = np.random.multivariate_normal(np.zeros(D), (1/(4*np.pi**2)*(1/kern.lengthscale**2)*np.identity(D)), M).flatten()
			#S = np.random.normal(0, 1/(4*np.pi**2*kern.lengthscale**2), M*D)
			nlmlc = self.ssgpr(X, kern, S, Y)
			if nlmlc<nlml:
				S_save = S
				nlml = nlmlc
		return S_save
