import sys
import numpy as np
import random as ran
import PcgComp
import random as ran
import time

def standardizeData(array):
	arr = array.copy()
	rows, cols = arr.shape
	for col in xrange(cols):
		std = np.std(arr[:,col])
		mean = np.mean(arr[:,col])
		arr[:,col] = (arr[:,col] - mean) / std
	return arr

def normalizeColumns(array):
	arr = array.copy()
	rows, cols = arr.shape
	for col in xrange(cols):
		maxim = arr[:,col].max()
		minim = arr[:,col].min()
		arr[:,col] = (arr[:,col] - minim) / (maxim - minim)
	return arr

args = sys.argv

# <------ select dataset for regression ------>

data = np.loadtxt('PowerPlant_Data.csv',delimiter=',')
data = standardizeData(data)

X = data[:,:4]
Y = data[:,4][:,None]

seed = 48
np.random.seed(seed)

N = np.shape(X)[0]

var = float(args[2])
ls = float(args[3])
noise = float(args[4])

th = 1e-10

#file = open("resultsDirect.txt", "w")

# <------ randomly sample + optimize sub-inputs ------>

M = int(np.sqrt(N))
ipHelper = PcgComp.util.InducingPointsHelper(seed)
XmRandom = ipHelper.get_random_inducing_points(X,M)

# <------ randomly sample + optimize sub-inputs ------>

kern = PcgComp.kernels.RBF(ls, var, noise)
K = kern.K(X,X) + kern.noise*np.identity(N)
cg = PcgComp.methods.Cg(K,Y,threshold=th)
cgIterations = int(cg.iterations)

if (args[1] == 'kron'):
	P2 = PcgComp.preconditioners.Kiss(X, kern)
	pcg = PcgComp.methods.KronTruncatedFlexiblePcg(K, Y, P2.precon, P2.W, P2.Ku, kern, threshold=1e-10, innerThreshold=1e-10)
	pcgIterations = int(pcg.outer_iterations)
else:
	if (args[1] == 'block'):
		precon = PcgComp.preconditioners.BlockJacobi(X, kern, M)
	elif (args[1] == 'svd'):
		precon = PcgComp.preconditioners.SVD(X, kern, M)
	elif (args[1] == 'pitc'):
		precon = PcgComp.preconditioners.PITC(X, kern, XmRandom)
	elif (args[1] == 'fitc'):
		precon = PcgComp.preconditioners.FITC(X, kern, XmRandom)
	elif (args[1] == 'spectral'):
		precon = PcgComp.preconditioners.Spectral(X, Y, kern, M)
	else:
		precon = PcgComp.preconditioners.FITC(X, kern, XmRandom)

	P = precon.precon
	pcg = PcgComp.methods.RegularPcg(K, Y, P, threshold=th,preconInv=precon.get_inversion())
	pcgIterations = int(pcg.iterations)

print cgIterations
print pcgIterations
print np.log10(float(pcgIterations)/float(cgIterations))

