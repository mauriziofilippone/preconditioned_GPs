import numpy as np
from preconditioner import Preconditioner
import time

class ILU(Preconditioner):

    def __init__(self, X, kern):
    	super(ILU, self).__init__("ILU")


        start = time.time()
    	K = kern.K(X,X)
    	N = np.shape(K)[0]
    	A = np.copy(K)

    	for k in xrange(N):
    		A[k][k] = np.sqrt(K[k][k])
    		for i in xrange(k+1,N):
    			if (A[i][k] != 0):
    				A[i][k] = A[i][k] / A[k][k]

    		for j in xrange(k+1,N):
    			for i in xrange(j,N):
    				if (A[i][j] != 0):
    					A[i][j] = A[i][j] - A[i][k]*A[j][k]

    	for i in xrange(N):
    		for j in xrange(i+1,N):
    			A[i][j] = 0

        self.duration = time.time() - start
    	#self.precon = np.dot(A, A.conj().T)
        self.L = A
