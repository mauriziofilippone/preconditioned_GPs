import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
# names = ['Concrete - Block Precon','Concrete - Pitc Precon', 'Concrete - Fitc Precon','Concrete - Nyst Precon', 'Concrete - Spec Precon', 'Concrete - Randomized SVD']
# files = ['ConcBlock.txt','ConcPitc.txt', 'ConcFitc.txt','ConcNyst.txt','ConcSpec.txt','ConcSvd.txt']

# names = ['Protein - Block Precon','Protein - Pitc Precon','Protein - Nyst Precon']#, 'Protein - Spec Precon','Protein - Reg Precon']
# files = ['ProtBlock_output.txt','ProtPitc_output.txt','ProtNyst_output.txt']#,'ProtSpec_output.txt','ProtReg_output.txt']

pp = PdfPages('Protein - GpStuff Results.pdf')

files = ['FIC_NMLL.txt','VAR_NMLL.txt','PIC_NMLL.txt']
names = ['FIC','VAR','PIC']

data = np.zeros(5, dtype=object)

for d in xrange(len(files)):
	data = np.loadtxt(files[d], delimiter=' ',usecols=(1, 2))
	plt.plot(data[:,0], data[:,1], label=names[d])

plt.legend(loc='upper left')
plt.title('Protein')
plt.ylabel('Negative Marginal Log Likelihood')
plt.xlabel("Time taken (seconds - log)")
pp.savefig()


files = ['FIC_MSE.txt','VAR_MSE.txt','PIC_MSE.txt']
names = ['FIC','VAR','PIC']

data = np.zeros(5, dtype=object)

for d in xrange(len(files)):
	data = np.loadtxt(files[d], delimiter=' ',usecols=(1, 2))
	plt.plot(data[:,0], data[:,1], label=names[d])

plt.legend(loc='upper left')
plt.title('Protein')
plt.ylabel('Mean Squared Error')
plt.xlabel("Time taken (seconds - log)")
pp.savefig()

pp.close()