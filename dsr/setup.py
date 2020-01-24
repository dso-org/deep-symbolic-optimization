from distutils.core import setup
from Cython.Build import cythonize
import numpy

# To build cython code using setup try:
# python setup.py build_ext --inplace

setup(  name='dsr',
                version='1.0dev',
                description='Deep symbolic regression.',
                author='LLNL',
                packages=['dsr'],
                ext_modules=cythonize("dsr/dsr/cyfunc.pyx"), 
                include_dirs=[numpy.get_include()]
                )
