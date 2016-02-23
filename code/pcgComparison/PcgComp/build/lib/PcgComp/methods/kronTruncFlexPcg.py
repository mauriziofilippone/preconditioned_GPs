import numpy as np
from cg import Cg
from scipy import sparse
from kronCgDirect import KronCgDirect


"""
Solve linear system using truncated flexible conjugate gradient (intended for SKI inference)
Params:
    K - Covariance Matrix
    Y - Target labels
    P - Preconditioning matrix
    W - Weight matrix W
    Ku - Array of dimension-specific kernels
    kern - Kernel class
    init - Initial solution
    threshold - Termintion criteria for algorithm
    innerThreshold - Termination criteria for inner loop
"""
class KronTruncatedFlexiblePcg(object):

    def __init__(self, K, Y, P, W, Ku, kern=None, init=None, threshold=1e-9, innerThreshold=1e-9):

        mMax = 15
        N = np.shape(K)[0]
        if init is None:
        	init = np.zeros(N)

        self.K = K
        self.P = P
        self.Y = Y.flatten()

        x = init
        r_prev = np.zeros(N)
        r = self.Y - np.dot(self.K, x)
        p = np.zeros(6000,dtype=object)
        k = 0

        Ws = sparse.csr_matrix(W)
        WTs = sparse.csr_matrix(W.T)

        innerC = 0
        while True:
            if (np.dot(r.T,r).flatten() < threshold*N or k>15000):
                break
            interim = KronCgDirect(P, Ws, WTs, Ku, r, kern.noise, threshold=innerThreshold)
            z = interim.result
            count = interim.iterations
            innerC = innerC + count

            if (k == 0):
                p[k] = z
            else:
                m = max(1, k % (mMax+1))
                sum = 0
                if (k-m < 0):
                    start = 0
                else:
                    start = k - m
                for i in xrange((k-m), k):
                    frac = np.dot(z.T,np.dot(self.K,p[i]))/np.dot(p[i].T, np.dot(self.K, p[i]))
                    sum = sum + np.dot(frac, p[i])
                
                p[k] = z - sum
                
            alpha = np.dot(p[k].T, r) / np.dot(p[k].T, np.dot(self.K, p[k]))
            x = x + np.dot(alpha,p[k])
            r_prev = r
            r = r - np.dot(alpha, np.dot(K, p[k]))
            k = k + 1

        self.outer_iterations = k
        self.result = x
        self.iterations = innerC + k