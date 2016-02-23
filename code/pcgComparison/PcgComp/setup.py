#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, Extension
import numpy

# Version number
version = '0.0.1'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# cos_module_np = Extension('cos_module_np',
#                      sources=['PcgComp/kernels/cos_module_np.c'],
#                      include_dirs=[numpy.get_include()])


setup(name = 'PcgComp',
      version = version,
      author = read('AUTHORS.txt'),
      author_email = "kurt.cutajar@eurecom.fr",
      description = ("Comparison of Preconditioning Techniques for Kernel Matrices"),
      license = "BSD 3-clause",
      keywords = "machine-learning gaussian-processes kernels preconditioning",
      packages = ["PcgComp.methods",
                   "PcgComp.kernels",
                   "PcgComp.preconditioners",
                   "PcgComp.util"],
      package_dir={'PcgComp': 'PcgComp'},
      py_modules = ['PcgComp.__init__'],
      install_requires=['numpy>=1.7', 'scipy>=0.12'],
      extras_require = {'docs':['matplotlib >=1.3','Sphinx','IPython']},
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']      )
