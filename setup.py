#!/usr/bin/env python
import builtins
from setuptools import setup, find_packages, Extension  # This setup relies on setuptools since distutils is insufficient and badly hacked code
from setuptools.command.build_ext import build_ext as _build_ext
from pylandau.pylandau_src import pylandau_numba_ext


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        builtins.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

install_requires = ['numba>=0.53', 'numpy>=1.21']

setup(
    cmdclass={'build_ext': build_ext},
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,  # accept all data files and directories matched by MANIFEST.in or found in source control
    package_data={'': ['README.*', 'VERSION'], 'docs': ['*'], 'examples': ['*']},
    ext_modules=[pylandau_numba_ext.distutils_extension()],
    keywords=['Landau', 'Langau', 'PDF'],
    platforms='any'
)