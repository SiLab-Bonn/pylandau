#!/usr/bin/env python
from setuptools import setup, find_packages, Extension  # This setup relies on setuptools since distutils is insufficient and badly hacked code
from setuptools.command.build_ext import build_ext as _build_ext

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

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

version = '2.1.1'
author = 'David-Leon Pohl'
author_email = 'pohl@physik.uni-bonn.de'

install_requires = ['cython', 'numpy'] # scipy
setup_requires = ['numpy', 'cython']

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
    cmdclass={'build_ext':build_ext},
    install_requires=install_requires,
    setup_requires=setup_requires,
    packages=find_packages(),
    include_package_data=True,  # accept all data files and directories matched by MANIFEST.in or found in source control
    package_data={'': ['README.*', 'VERSION'], 'docs': ['*'], 'examples': ['*']},
    ext_modules=cpp_extension,
    keywords=['Landau', 'Langau', 'PDF'],
    platforms='any'
)
