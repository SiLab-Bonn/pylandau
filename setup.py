#!/usr/bin/env python
from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('pyLandau.landau', ['pyLandau/cpp/landau.pyx'])
]

version = '1.0.0rc'
author = 'David-Leon Pohl'
author_email = 'pohl@physik.uni-bonn.de'

setup(
    name='pyLandau',
    version=version,
    description='A Landau PDF definition to be used in Python.',
    url='https://github.com/SiLab-Bonn/pyLandau',
    license='GNU LESSER GENERAL PUBLIC LICENSE Version 2.1',
    long_description='',
    author=author,
    maintainer=author,
    author_email=author_email,
    maintainer_email=author_email,
    install_requires=['cython', 'numpy'],
    include_package_data=True,
    packages=["pyLandau"],
    package_data={'': ['*.txt', 'VERSION'], 'docs': ['*'], 'examples': ['*']},
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
    keywords=['Landau', 'Langau', 'PDF'],
    platforms='any'
)