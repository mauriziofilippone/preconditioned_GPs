#This implementation of Structured Kernel Interpolation is based on the article:
#
# @inproceedings{DBLP:conf/icml/WilsonN15,
#   author    = {Andrew Gordon Wilson and
#                Hannes Nickisch},
#   title     = {Kernel Interpolation for Scalable Structured Gaussian Processes {(KISS-GP)}},
#   booktitle = {Proceedings of the 32nd International Conference on Machine Learning,
#                {ICML} 2015, Lille, France, 6-11 July 2015},
#   pages     = {1775--1784},
#   year      = {2015},
#   crossref  = {DBLP:conf/icml/2015},
#   url       = {http://jmlr.org/proceedings/papers/v37/wilson15.html},
#   timestamp = {Sun, 05 Jul 2015 19:10:23 +0200},
#   biburl    = {http://dblp.uni-trier.de/rec/bib/conf/icml/WilsonN15},
#   bibsource = {dblp computer science bibliography, http://dblp.org}
# }

import numpy as np
import time
from preconditioner import Preconditioner
from ..util.kronHelper import KronHelper
import math

"""
SKI Preconditioner 
"""
class Kiss(Preconditioner):

    """
    Construct preconditioning matrix
        X - Training data
        kern - Class of kernel function
    """
    def __init__(self, X, kern):
        super(Kiss, self).__init__("Kiss")
        self.X = X
        self.kern = kern

        start = time.time()

        Xnew = self.normalize_columns(X)
        N = Xnew.shape[0]
        D = Xnew.shape[1]

        num_grid_interval = np.zeros((D))

        maximum = np.zeros(D)
        minimum = np.zeros(D)
        for i in xrange(D):
            maximum[i] = max(X[:,i])
            minimum[i] = min(X[:,i])
            num_grid_interval[i] = round(N**(float(3)/float(2*D)))#round((N**2)**(float(1)/D))
            if (num_grid_interval[i] == 1):
                num_grid_interval[i] = 2

        # construct grid vectors and intervals

        interval = np.zeros(D)
        vector = np.zeros(D, dtype=object)

        for i in xrange(D):
            [vector[i],interval[i]] = np.linspace(0, 1, num=num_grid_interval[i], retstep=True)
            
        for i in xrange(D):
            num_grid_interval[i] = len(vector[i])
            
        interval_matrix = np.zeros((N, D))
        assign = np.zeros(N)

        for i in xrange(D):
            for j in xrange(N):
                interval_matrix[j][i] = self.get_rounded_threshold(Xnew[j][i], interval[i], len(vector[i]), 0, 1)

        # construct weight matrix
        for j in xrange(N):
            val =0
            for t in xrange(D):
                val = val + interval_matrix[j][t]*np.prod(num_grid_interval[t+1:D])
            assign[j] = val
                
        W = np.zeros((N,np.prod(num_grid_interval)))

        for i in xrange(N):
            index = assign[i]
            W[i][index] = 1

        kron_helper = KronHelper()
        unnormalzed_vector = self.reverse_normalize(vector, minimum, maximum)
        [K, Kds] =  kron_helper.kron_inference(unnormalzed_vector, D, kern)
        #Kski = np.dot(np.dot(W, K), W.T)

        self.W = W
        self.Ku = Kds

        self.precon = None
        self.duration = time.time() - start

    """
    Normalize the given training data
    """
    def normalize_columns(self, array):
        arr = array.copy()
        rows, cols = arr.shape
        for col in xrange(cols):
            maxim = arr[:,col].max()
            minim = arr[:,col].min()
            arr[:,col] = (arr[:,col] - minim) / (maxim - minim)
        return arr

    """
    Reverse the normalization carried out on the data
    """
    def reverse_normalize(self, array, minimum, maximum):
        new_array = np.zeros(len(array), dtype=object)
        for i in xrange(len(array)):
            new_array[i] = array[i]*(maximum[i] - minimum[i]) + minimum[i]
        return new_array

    """
    Assign points to designated nearest location in the grid
    """
    def get_rounded_threshold(self, a, min_clip, max_interval, minim, maxim):
        interval = round(float(a) / min_clip)
        rounded_val = interval * min_clip
        if (rounded_val > maxim):
            return max_interval
        if (rounded_val < minim):
            return 0
        return interval
        