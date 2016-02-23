#This implementation is based on the article:

# @inproceedings{snelson2005sparse,
#   title={Sparse Gaussian processes using pseudo-inputs},
#   author={Snelson, Edward and Ghahramani, Zoubin},
#   booktitle={Advances in neural information processing systems},
#   pages={1257--1264},
#   year={2005}
# }

from __future__ import division
import numpy as np
from scipy.spatial.distance import cdist, squareform, pdist
from scipy.optimize import fmin_l_bfgs_b
from scipy.linalg import cholesky
import random as ran
import numpy.matlib,numpy.linalg,numpy.random

"""
Helper class for inducing point methods.

"""
class InducingPointsHelper(object):

	def __init__(self, seed):
		ran.seed(seed)
		self.name = "InducingPointsHelper"

	"""
	Returns a random selection of points from the given dataset
		X - Dataset
		M - Number of points to be selected
	"""
	def get_random_inducing_points(self, X, M):
		rand = ran.sample(range(0, X.shape[0]), M)
		return X[rand]

	"""
	Procedure for optimizing the given inducing points
		X - Dataset
		Y - Target labels
		M - Number of inducing points
		kern - Class of kernel function
	"""
	def optimize_inducing_points(self, X, Y, M, kern):
		dim = np.shape(X)[1]
		hyp_init = np.ones((dim+2, 1))
		for i in xrange(dim):
			hyp_init[i] = kern.lengthscale
		hyp_len = len(hyp_init)
		hyp_init[hyp_len - 2] = kern.variance
		hyp_init[hyp_len - 1] = kern.noise

		rand = ran.sample(range(0, X.shape[0]), M)
		I = X[rand]
		W = np.vstack((np.reshape(I, (M*dim,1), order='F'), hyp_init))
		res = fmin_l_bfgs_b(self.like_spgp, W, iprint=False, args=(X,Y,M))[0]
		return np.reshape(res[0:M*dim], (M, dim), order='F')

	def dist(self, x1, x2):
		x1 = np.reshape(x1,(-1,1))
		x2 = np.reshape(x2,(-1,1))
		n1 = len(x1)
		n2 = len(x2)
		return np.matlib.repmat(x1,1,n2) - np.matlib.repmat(x2.T,n1,1)

	"""
	Procedure for evaluating likelihood of the inducing point approximation
		W - Array of hyperparameters (incl. inducing points)
		x - Dataset
		y - Target labels
		M - Number of inducing points
		kern - Class of kernel function
	"""
	def like_spgp(self, W, x, y, M):

		jitter = 1e-6
		N = np.shape(x)[0]
		dim = np.shape(x)[1]
		
		length = len(W)
		pts = W[0:(length-2-dim)]
		xb = np.reshape(pts, (M, dim), order='F')
		
		b = np.exp(W[(length-2-dim):(length-2)])
		c = np.exp(W[length-2])
		sig = np.exp(W[length-1])
		
		xb = xb * np.matlib.repmat(np.sqrt(b).T, M, 1)
		x = x * np.matlib.repmat(np.sqrt(b).T, N, 1)
		
		Q = np.dot(xb, xb.T)
		diag = np.reshape(np.diag(Q), (-1,1))
		Q = np.matlib.repmat(diag,1,M) + np.matlib.repmat(diag.T, M, 1) - 2*Q
		Q = c*np.exp(-0.5*Q) + jitter*np.identity(M)
		
		x_sum = np.reshape(np.sum(x*x, axis=1), (-1,1))
		xb_sum = np.reshape(np.sum(xb*xb, axis=1), (-1,1))
		K = -2*np.dot(xb, x.T) + np.matlib.repmat(x_sum.T,M,1) + np.matlib.repmat(xb_sum,1,N)
		K = c*np.exp(-0.5*K)
		
		L = cholesky(Q,lower=False).T
		V = np.linalg.solve(L,K)
		vSum = np.reshape(np.sum(np.power(V,2), axis=0),(-1,1))
		ep = 1 + (c - vSum.T)/sig
		epSqrt = np.reshape(np.sqrt(ep), (-1, 1))
		K = K / np.matlib.repmat(epSqrt.T,M,1)
		V = V / np.matlib.repmat(epSqrt.T,M,1)
		y = y / epSqrt
		Lm = cholesky(sig*np.identity(M) + np.dot(V,V.T), lower=False).T
		invLmV = np.linalg.solve(Lm,V)
		bet = np.dot(invLmV, y)
		
		# Likelihood
		fw = np.sum(np.log(np.diag(Lm))) + (N-M)/2*np.log(sig) + (np.dot(y.T,y) - np.dot(bet.T,bet))/2/sig + np.sum(np.log(ep))/2 + 0.5*N*np.log(2*np.pi)

		# Derivatives
		
		Lt = np.dot(L,Lm)
		B1 = np.linalg.solve(Lt.T, invLmV)
		b1 = np.linalg.solve(Lt.T, bet)
		invLV = np.linalg.solve(L.T, V)
		invL = np.linalg.inv(L)
		invQ = np.dot(invL.T, invL)
		invLt = np.linalg.inv(Lt)
		invA = np.dot(invLt.T,invLt)
		
		mu = np.dot(np.linalg.solve(Lm.T,bet).T, V).T
		sumV = np.reshape(np.sum(np.power(V, 2),axis=0),(-1,1))
		sumVsq = sumV.T
		sumB = np.reshape(np.sum(invLmV*invLmV,axis=0), (-1,1))
		bigSum = y*np.dot(bet.T,invLmV).T/sig - sumB/2 - (np.power(y,2)+np.power(mu,2))/2/sig + 0.5
		
		TT = np.dot(invLV, (invLV.T*np.matlib.repmat(bigSum,1,M)))
		
		dfxb = np.zeros((M,dim))
		dfb = np.zeros(dim)
		for i in xrange(dim):
			dnnQ = self.dist(xb[:,i],xb[:,i])*Q
			dNnK = self.dist(-xb[:,i],-x[:,i])*K
			epdot = -2/sig*dNnK*invLV
			epPmod = -1*np.reshape(np.sum(epdot,axis=0),(-1,1))
			
			sum1 = np.reshape(np.sum((invQ - invA*sig)*dnnQ,axis=1), (-1,1))
			sum2 = np.reshape(np.sum(dnnQ*TT,axis=1), (-1,1))
			dfxb[:,i] = (-b1*(np.dot(dNnK, (y-mu))/sig + np.dot(dnnQ,b1)) + sum1 + np.dot(epdot,bigSum) - 2/sig*sum2).flatten()
						
			dNnK = dNnK*B1
			dfxb[:,i] = dfxb[:,i] + np.sum(dNnK,axis=1)
			
			dfxb[:,i] = dfxb[:,i] * np.sqrt(b[i])
			
		dfc = (M + jitter*np.trace(invQ-sig*invA) - sig*sum2)/2 - np.dot(mu.T, (y-mu))/sig + np.dot(b1.T, np.dot((Q - np.dot(jitter, np.identity(M))), b1))/2 + np.dot(epc,bigSum)
		
		#noise
		dfsig = np.sum(bigSum / ep.T)
		
		derivs = np.vstack((np.reshape(dfxb,(M*dim,1),order='F'),np.reshape(dfb[0].flatten(),(-1,1)),np.reshape(dfb[1].flatten(),(-1,1)),np.reshape(dfc,(-1,1)),np.reshape(dfsig,(-1,1)))).flatten()

		return fw, derivs