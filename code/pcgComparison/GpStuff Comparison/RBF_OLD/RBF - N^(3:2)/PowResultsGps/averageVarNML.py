import numpy as np
# names = ['Concrete - Block Precon','Concrete - Pitc Precon', 'Concrete - Fitc Precon','Concrete - Nyst Precon', 'Concrete - Spec Precon', 'Concrete - Randomized SVD']
# files = ['ConcBlock.txt','ConcPitc.txt', 'ConcFitc.txt','ConcNyst.txt','ConcSpec.txt','ConcSvd.txt']

# names = ['Protein - Block Precon','Protein - Pitc Precon','Protein - Nyst Precon']#, 'Protein - Spec Precon','Protein - Reg Precon']
# files = ['ProtBlock_output.txt','ProtPitc_output.txt','ProtNyst_output.txt']#,'ProtSpec_output.txt','ProtReg_output.txt']

files = ['VAR_POWER_FOLD_1.txt','VAR_POWER_FOLD_2.txt', 'VAR_POWER_FOLD_3.txt', 'VAR_POWER_FOLD_4.txt','VAR_POWER_FOLD_5.txt']

data = np.zeros(5, dtype=object)

iterations = 5
for d in xrange(len(files)):
	data[d] = np.loadtxt(files[d],delimiter=' ',usecols=(0, 1, 2))

lastT = np.zeros(len(files))
lastE = np.zeros(len(files))

count = 0
while(True):
	occur = 0
	totalT = 0
	totalE = 0
	iterations = 0
	count = count + 1
	for d in xrange(len(files)):
		fold = data[d]
		if len(fold) >= count:
			occur = occur+1
			iterations = fold[count-1][0]
			totalT = totalT + fold[count-1][1]
			totalE = totalE + fold[count-1][2]
			lastT[d] = fold[count-1][1]
			lastE[d] = fold[count-1][2]
		else:
			totalE = totalE + lastE[d]
	if (occur == 0):
		break
	avgT = totalT / occur
	avgE = totalE / len(files)
	if (count < 10 or count % 100 == 0):
		print '%s %s %s' % (iterations, np.log10(avgT), avgE)
