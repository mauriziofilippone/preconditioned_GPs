import numpy as np
from scipy.spatial.distance import cdist
from kernel import Kernel

"""
Implementation of isotropic RBF/SE kernel

"""
class RBF(Kernel):

    def __init__(self, lengthscale=1, variance=1, noise=1):
        super(RBF, self).__init__("RBF")
        self.lengthscale = lengthscale
        self.variance = variance
        self.jitter = 1e-9
        self.noise = noise / self.variance + self.jitter# dividing by variance for new strategy

    def K(self, X1, X2):
        """ GP squared exponential kernel """
        pairwise_dists = cdist(X1, X2, 'euclidean')
        return self.variance*np.exp(-0.5 * (pairwise_dists ** 2) / self.lengthscale ** 2)

    def K_scalar(self, X1, X2, original_dimensions):
        pairwise_dists = cdist(X1, X2, 'euclidean')
        return (self.variance**(float(1)/original_dimensions)) * np.exp(-0.5 * pairwise_dists ** 2 / self.lengthscale ** 2)
