from distutils.core import setup
from Cython.Build import cythonize

setup(  name='dsr',
                version='1.0dev',
                description='Deep symbolic regression.',
                author='LLNL',
                packages=['dsr'],
                ext_modules=cythonize("dsr/cyfunc.pyx")
                )
