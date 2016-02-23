import numpy as np
from cg import Cg

"""
Solve linear system using flexible conjugate gradient (without truncation)
Params:
    K - Covariance Matrix
    Y - Target labels
    P - Preconditioner Matrix (can be set to none)
    init - Initial solution
    threshold - Termintion criteria for outer loop
    innerThreshold - Termination criteria for inner loop

"""
class FlexiblePcg(object):

    def __init__(self, K, Y, P, init=None, threshold=1e-9, innerThreshold=1e-9):
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

        innerC = 0
        while True:
            diff = r - r_prev
            if (np.dot(diff.T,diff).flatten() < threshold) or k>5000:
                break
            interim = Cg(P, r, threshold=innerThreshold)
            z = interim.result
            count = interim.iterations
            innerC = innerC + count

            if (k == 0):
                p[k] = z
            else:
                sum = 0
                for i in xrange(k):
                    frac = np.dot(z.T,np.dot(self.K,p[i]))/np.dot(p[i].T, np.dot(self.K, p[i]))
                    sum = sum + np.dot(frac, p[i])
                
                p[k] = z - sum
                
            alpha = np.dot(p[k].T, r) / np.dot(p[k].T, np.dot(self.K, p[k]))
            x = x + np.dot(alpha,p[k])
            r_prev = r
            r = r - np.dot(alpha, np.dot(K, p[k]))
            k = k + 1

            self.result = x
            self.iterations = innerC + k