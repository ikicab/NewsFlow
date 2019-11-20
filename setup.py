# -*- coding: utf-8 -*-
"""
@author: Barbara Ikica
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize(Extension("NewsFlow", sources=["NewsFlow.pyx"], language="c++"),
    compiler_directives={'language_level' : 3}),
    include_dirs=[np.get_include()]
)

#directives = {'linetrace': False, 'language_level': 3}

#python setup.py build_ext --inplace
#cython -a NewsFlow.pyx