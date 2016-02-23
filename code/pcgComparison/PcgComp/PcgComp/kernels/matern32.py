import numpy as np
from scipy.spatial.distance import cdist
from kernel import Kernel

"""
Implementation of isotropic Matern-3/2 kernel

"""
class Matern32(Kernel):

    def __init__(self, lengthscale=1, variance=1, noise=1):
        super(Matern32, self).__init__("Matern 3/2")
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise = noise # adding jitter for numerical stability

    def K(self, X1, X2):
        """ GP matern-3/2 kernel """
        pairwise_dists = cdist(X1, X2, 'euclidean')/self.lengthscale
        return self.variance * (1. + np.sqrt(3.) * pairwise_dists) * np.exp(-np.sqrt(3.) * pairwise_dists)

    def K_scalar(self, X1, X2, original_dimensions):
        raise NotImplementedError