from setuptools import setup, Extension
import os
import numpy
from Cython.Build import cythonize

required = [
    "pytest",
    "cython",
    "numpy>=1.19,<1.24",
    "tensorflow>=2.4",
    "numba>=0.53",
    "sympy",
    "pandas",
    "scikit-learn",
    "click",
    "deap",
    "pathos",
    "seaborn",
    "progress",
    "tqdm",
    "commentjson",
    "PyYAML",
    "prettytable"
]

extras = {
    "control": [
        "mpi4py",
        "gym[box2d]==0.15.4",
        "pybullet",
        "stable-baselines[mpi]==2.10.0"
    ],
    "regression": []
}
extras['all'] = list(set([item for group in extras.values() for item in group]))

setup(
    name='dso',
    version='1.0dev',
    description='Deep symbolic optimization.',
    author='LLNL',
    packages=['dso'],
    ext_modules=cythonize([os.path.join('dso', 'cyfunc.pyx')]),
    include_dirs=[numpy.get_include()],
    install_requires=required,
    extras_require=extras
)
