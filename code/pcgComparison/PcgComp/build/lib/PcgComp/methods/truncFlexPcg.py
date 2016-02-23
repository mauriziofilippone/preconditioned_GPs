import numpy as np
from cg import Cg

"""
Solve linear system using flexible conjugate gradient (with truncation)
Params:
    K - Covariance Matrix
    Y - Target labels
    P - Preconditioner Matrix (can be set to none)
    init - Initial solution
    thershold - Termintion criteria for outer loop
    innerThreshold - Termination criteria for inner loop

"""
class TruncatedFlexiblePcg(object):

    def __init__(self, K, Y, P, init=None, threshold=1e-9, innerThreshold=1e-9):

        mMax = 15
        N = np.shape(K)[0]
        if init is None:
        	init = np.zeros((N,1))

        self.K = K
        self.P = P
        self.Y = Y.flatten()

        x = init
        r_prev = np.zeros((N,1))
        r = Y - np.dot(self.K, x)
        p = np.zeros(6000,dtype=object)
        k = 0

        innerC = 0
        while True:
            if (np.dot(r.T,r).flatten() < threshold*N or k>50000):
                break
            interim = Cg(P, r, threshold=innerThreshold)
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
                    sum = sum + frac* p[i]
                
                p[k] = z - sum
                
            alpha = np.dot(p[k].T, r) / np.dot(p[k].T, np.dot(self.K, p[k]))
            x = x + alpha*p[k]
            r_prev = r
            r = r - alpha*np.dot(K, p[k])
            k = k + 1

        self.outer_iterations = k
        self.result = x
        self.iterations = innerC + k