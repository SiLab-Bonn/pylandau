#!/usr/bin/env python
from setuptools import setup, find_packages, Extension  # This setup relies on setuptools since distutils is insufficient and badly hacked code
import numpy as np

# Check if cython exists, then use it. Otherwise compile already cythonized cpp file
have_cython = False
try:
    from Cython.Build import cythonize
    have_cython = True
except ImportError:
    pass

if have_cython:
    cpp_extension = cythonize(Extension('pylandau', ['pyLandau/cpp/pylandau.pyx']))
else:
    cpp_extension = [Extension('pylandau',
                               sources=['pyLandau/cpp/pylandau.cpp'],
                               language="c++")]

version = '2.0.1'
author = 'David-Leon Pohl'
author_email = 'pohl@physik.uni-bonn.de'

install_requires = ['cython', 'numpy']

setup(
    name='pylandau',
    version=version,
    description='A Landau PDF definition to be used in Python.',
    url='https://github.com/SiLab-Bonn/pyLandau',
    license='GNU LESSER GENERAL PUBLIC LICENSE Version 2.1',
    long_description='The Landau propability density function is defined in C++ and made available to python via cython. Also a fast Gaus+Landau convolution is available. The interface accepts numpy arrays.',
    author=author,
    maintainer=author,
    author_email=author_email,
    maintainer_email=author_email,
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,  # accept all data files and directories matched by MANIFEST.in or found in source control
    package_data={'': ['README.*', 'VERSION'], 'docs': ['*'], 'examples': ['*']},
    ext_modules=cpp_extension,
    include_dirs=[np.get_include()],
    keywords=['Landau', 'Langau', 'PDF'],
    platforms='any'
)
