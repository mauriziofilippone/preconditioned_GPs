import numpy as np
from scipy.linalg import block_diag
from preconditioner import Preconditioner
import time

"""
Block Jacobi Preconditioner
"""
class BlockJacobi(Preconditioner):

    """
    Construct preconditioner
        X - Training data
        kern - Class of kernel function
        M - Number of points ber block
    """
    def __init__(self, X, kern, M):
        super(BlockJacobi, self).__init__("BlockJacobi")
        self.M = M

        start = time.time()
        X_split = np.array_split(X, M)
        kern_blocks = np.zeros((M),dtype=object)

        for t in xrange(M):
        	size = np.shape(X_split[t])[0]
        	kern_blocks[t] = kern.K(X_split[t], X_split[t]) + kern.noise*np.identity(size)

        self.duration = time.time()-start
        self.blocks = kern_blocks
        self.precon = block_diag(*kern_blocks)

    """
    Compute inversion of the preconditioner.
    """
    def get_inversion(self):
    	diag_blocks = self.blocks
    	inverted_blocks = np.zeros(len(diag_blocks), dtype=object)
    	for i in xrange(len(diag_blocks)):
    		inverted_blocks[i] = np.linalg.inv(diag_blocks[i])

    	inverted_diag = block_diag(*inverted_blocks)

    	return inverted_diag
        

    """
    Compute inversion of preconditioner for Laplace Approximation.
    """
    def get_laplace_inversion(self, W, Wsqrt):
        Wsqrt_split = np.array_split(Wsqrt, self.M)
        diag_blocks = self.blocks
        inverted_blocks = np.zeros(len(diag_blocks), dtype=object)
        for i in xrange(len(diag_blocks)):
            Wblock = np.diag(Wsqrt_split[i].flatten())
            block = np.dot(Wblock, np.dot(diag_blocks[i], Wblock))
            inverted_blocks[i] = np.linalg.inv(block + np.identity(len(Wblock)))

        inverted_diag = block_diag(*inverted_blocks)

        return inverted_diag
