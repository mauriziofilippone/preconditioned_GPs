import sys
import numpy as np
import random as ran
import matplotlib as m
import matplotlib
# import PcgComp
import random as ran
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def show_values(pc, fmt="%s", **kw):
    from itertools import izip
    pc.update_scalarmappable()
    ax = pc.get_axes()
    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)

        if value < 0:
        	value = r'$-$'
        elif value > 0:
        	value = r'$+$'
        else:
        	value = r'$\circ$'

        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

pp = PdfPages('PowerPlant Data - Precon Results.pdf')

# names = ['Concrete - Block Precon','Concrete - Pitc Precon', 'Concrete - Fitc Precon','Concrete - Nyst Precon', 'Concrete - Spec Precon', 'Concrete - Randomized SVD']
# files = ['ConcBlock.txt','ConcPitc.txt', 'ConcFitc.txt','ConcNyst.txt','ConcSpec.txt','ConcSvd.txt']

# names = ['Protein - Block Precon','Protein - Pitc Precon','Protein - Nyst Precon']#, 'Protein - Spec Precon','Protein - Reg Precon']
# files = ['ProtBlock_output.txt','ProtPitc_output.txt','ProtNyst_output.txt']#,'ProtSpec_output.txt','ProtReg_output.txt']


names = ['PowerPlant Data - Cg Iterations (log10)']
files = ['iterations.txt']

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=38)


for d in xrange(len(files)):
	data = np.loadtxt(files[d],delimiter=',')

	# for i in xrange(np.shape(data)[0]):
	# 	for j in xrange(np.shape(data)[1]):
	# 		if (data[i][j]<1 and data[i][j]!=0):
	# 			data[i][j] = -1/data[i][j]

	row_labels = np.array([-3, -2, -1, 0, 1, 2])
	column_labels = np.array([-2, -4, -6])

	fig, ax = plt.subplots()
	im = ax.pcolor(data, cmap='YlOrRd', edgecolor='black', linestyle=':', lw=1,vmin=0,vmax=5)

	# show_values(im)
	#fig.colorbar(im)

	ax.yaxis.set(ticks=np.arange(0.5, len(column_labels)), ticklabels=column_labels, label='Lengthscale')
	ax.xaxis.set(ticks=np.arange(0.5, len(row_labels)), ticklabels=row_labels)
	ax.set_aspect('equal')
	ax.tick_params(axis='x', pad=5)
	ax.tick_params(axis='y', pad=3)
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif', size=46)
	plt.xlabel(r'$\log_{10}(l)$')
	plt.ylabel(r'$\log_{10}(\lambda)$')
	#plt.title(names[d])

	pp.savefig(bbox_inches='tight')


# names = ['Block Jacobi','PITC', 'FITC','Nystrom', 'Spectral', 'Randomized SVD', 'Regularized', 'KISS Interpolation']
# files = ['ConcBlock.txt','ConcPitc.txt', 'ConcFitc.txt','ConcNyst.txt','ConcSpec.txt','ConcSvd.txt','ConcReg.txt','ConcKron.txt']

names = ['Block Jacobi','PITC', 'FITC','Nystrom', 'Spectral', 'Randomized SVD', 'Regularized', 'SKI']
files = ['PowBlock.txt', 'PowPitc.txt', 'PowFitc.txt','PowNyst.txt','PowSpec.txt', 'PowSvd.txt', 'PowReg.txt', 'PowKron.txt']

# names = ['PowerPlant - Block Precon','PowerPlant - Pitc Precon', 'PowerPlant - Fitc Precon', 'PowerPlant - Nyst Precon', 'PowerPlant - Spec Precon', 'PowerPant - SVD Precon']
# files = ['PowBlock.txt','PowPitc.txt', 'PowFitc.txt', 'PowNyst.txt','PowSpec.txt', 'PowSvd.txt']

for d in xrange(len(files)):
	data = np.loadtxt(files[d],delimiter=',')

	# for i in xrange(np.shape(data)[0]):
	# 	for j in xrange(np.shape(data)[1]):
	# 		if (data[i][j]<1 and data[i][j]!=0):
	# 			data[i][j] = -1/data[i][j]

	row_labels = np.array([-3, -2, -1, 0, 1, 2])
	column_labels = np.array([-2, -4, -6])

	fig, ax = plt.subplots()
	im = ax.pcolor(data, cmap='bwr', edgecolor='black', linestyle=':', lw=1,vmin=-2,vmax=2)

	show_values(im)
	# fig.colorbar(im)
	ax.set_aspect('equal')
	# if (d==0 or d==2 or d==4):
	# 	ax.yaxis.set(ticks=np.arange(0.5, len(column_labels)), ticklabels=column_labels)
	# 	ax.xaxis.set(ticks=[])
	# 	# plt.ylabel('Noise')
	# elif (d==1 or d==3 or d==5):
	# 	ax.yaxis.set(ticks=[])
	# 	ax.xaxis.set(ticks=[])
	# elif (d==6):
	# 	ax.yaxis.set(ticks=np.arange(0.5, len(column_labels)), ticklabels=column_labels)
	# 	ax.xaxis.set(ticks=np.arange(0.5, len(row_labels)), ticklabels=row_labels)
	# 	# plt.xlabel('Lengthscale')
	# 	# plt.ylabel('Noise')
	# else:
	# 	ax.yaxis.set(ticks=[])
	# 	ax.xaxis.set(ticks=np.arange(0.5, len(row_labels)), ticklabels=row_labels)
	# 	# plt.xlabel('Lengthscale')

	ax.yaxis.set(ticks=np.arange(0.5, len(column_labels)), ticklabels=column_labels)
	ax.xaxis.set(ticks=np.arange(0.5, len(row_labels)), ticklabels=row_labels)

	ax.tick_params(axis='x', pad=40)
	ax.tick_params(axis='y', pad=20)

	# plt.xlabel('Lengthscale')
	# plt.ylabel('Noise')
	plt.title(names[d],y=1.08)

	pp.savefig(bbox_inches='tight')

pp.close()
