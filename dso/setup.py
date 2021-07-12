from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

# To build cython code using setup try:
# python setup.py build_ext --inplace

setup(  name='dso',
        version='1.0dev',
        description='Deep symbolic optimization.',
        author='LLNL',
        packages=['dso'],
        ext_modules=cythonize([os.path.join('dso','cyfunc.pyx')]), 
        include_dirs=[numpy.get_include()]
        )
