import sys
import numpy as np
import random as ran
import matplotlib as m
import matplotlib
import PcgComp
import random as ran
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def show_values(pc, fmt="%.2f", **kw):
    from itertools import izip
    pc.update_scalarmappable()
    ax = pc.get_axes()
    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        color = (0.0, 0.0, 0.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

pp = PdfPages('Protein - Precon Results.pdf')

# names = ['Concrete - Block Precon','Concrete - Pitc Precon', 'Concrete - Fitc Precon','Concrete - Nyst Precon', 'Concrete - Spec Precon', 'Concrete - Randomized SVD']
# files = ['ConcBlock.txt','ConcPitc.txt', 'ConcFitc.txt','ConcNyst.txt','ConcSpec.txt','ConcSvd.txt']

# names = ['Protein - Block Precon','Protein - Pitc Precon','Protein - Nyst Precon']#, 'Protein - Spec Precon','Protein - Reg Precon']
# files = ['ProtBlock_output.txt','ProtPitc_output.txt','ProtNyst_output.txt']#,'ProtSpec_output.txt','ProtReg_output.txt']


names = ['Protein - Cg Iterations (log10)']
files = ['iterations.txt']

for d in xrange(len(files)):
	data = np.loadtxt(files[d],delimiter=',')

	# for i in xrange(np.shape(data)[0]):
	# 	for j in xrange(np.shape(data)[1]):
	# 		if (data[i][j]<1 and data[i][j]!=0):
	# 			data[i][j] = -1/data[i][j]

	row_labels = np.array([1e-3, 1e-2, 1e-1, 1, 10, 100])
	column_labels = np.array([1e-1, 1e-2, 1e-3])

	fig, ax = plt.subplots()
	im = ax.pcolor(data, cmap='YlOrRd', edgecolor='black', linestyle=':', lw=1,vmin=0,vmax=5)

	show_values(im)
	fig.colorbar(im)

	ax.yaxis.set(ticks=np.arange(0.5, len(column_labels)), ticklabels=column_labels, label='Lengthscale')
	ax.xaxis.set(ticks=np.arange(0.5, len(row_labels)), ticklabels=row_labels)

	plt.xlabel('Lengthscale')
	plt.ylabel('Noise')
	plt.title(names[d])



	pp.savefig()

names = ['Protein - Block Precon','Protein - Pitc Precon', 'Protein - Fitc Precon', 'Protein - Nyst Precon', 'Protein - Spec Precon', 'Protein - SVD Precon']
files = ['ProtBlock.txt','ProtPitc.txt', 'ProtFitc.txt', 'ProtNyst.txt','ProtSpec.txt', 'ProtSvd.txt']

for d in xrange(len(files)):
	data = np.loadtxt(files[d],delimiter=',')

	# for i in xrange(np.shape(data)[0]):
	# 	for j in xrange(np.shape(data)[1]):
	# 		if (data[i][j]<1 and data[i][j]!=0):
	# 			data[i][j] = -1/data[i][j]

	row_labels = np.array([1e-3, 1e-2, 1e-1, 1, 10, 100])
	column_labels = np.array([1e-1, 1e-2, 1e-3])

	fig, ax = plt.subplots()
	im = ax.pcolor(data, cmap='bwr', edgecolor='black', linestyle=':', lw=1,vmin=-2,vmax=2)

	show_values(im)
	fig.colorbar(im)

	ax.yaxis.set(ticks=np.arange(0.5, len(column_labels)), ticklabels=column_labels, label='Lengthscale')
	ax.xaxis.set(ticks=np.arange(0.5, len(row_labels)), ticklabels=row_labels)

	plt.xlabel('Lengthscale')
	plt.ylabel('Noise')
	plt.title(names[d])

	pp.savefig()

pp.close()