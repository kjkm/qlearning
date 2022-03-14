import numpy
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='reinforcement learning',
    version='1.0',
    author='Kieran Kim-Murphy',
    author_email='kimmurkj@plu.edu',
    description='Reinforcement Learning Lab',
    ext_modules=cythonize("q_learning.pyx"),
    include_dirs=[numpy.get_include()]
)
