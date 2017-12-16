#!/usr/bin/env python
# This setup relies on setuptools since distutils is insufficient and
# badly hacked code
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):

    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

# Check if cython exists, then use it. Otherwise compile already
# cythonized cpp file
have_cython = False
try:
    from Cython.Build import cythonize
    have_cython = True
except ImportError:
    pass

if have_cython:
    cpp_extension = cythonize(
        Extension('pylandau.landaulib', ['src/landaulib.pyx']))
else:
    cpp_extension = [Extension('pylandau_cpp',
                               sources=['src/landaulib.cpp'],
                               language="c++")]

version = '2.1.1'
author = 'David-Leon Pohl'
author_email = 'pohl@physik.uni-bonn.de'

install_requires = ['cython', 'numpy']
setup_requires = ['numpy', 'cython']

setup(
    name='pylandau',
    version=version,
    description='A Landau PDF definition to be used in Python.',
    url='https://github.com/SiLab-Bonn/pyLandau',
    license='GNU LESSER GENERAL PUBLIC LICENSE Version 2.1',
    long_description='Landau propability density function defined in C++'
    ' and made available to Python via cython. Also a fast'
    ' Gaus+Landau convolution is available. The interface'
    ' accepts numpy arrays.',
    author=author,
    maintainer=author,
    author_email=author_email,
    maintainer_email=author_email,
    cmdclass={'build_ext': build_ext},
    install_requires=install_requires,
    setup_requires=setup_requires,
    packages=find_packages(),
    # accept all data files and directories matched by MANIFEST.in or found in
    # source control
    include_package_data=True,
    package_data={
        '': ['README.*', 'VERSION'], 'docs': ['*'], 'examples': ['*']},
    ext_modules=cpp_extension,
    keywords=['Landau', 'Langau', 'PDF'],
    platforms='any'
)
