#!/usr/bin/env python
import os, sys
from setuptools import setup, find_packages  # This setup relies on setuptools since distutils is insufficient and badly hacked code

if os.getcwd() not in sys.path: # Needed for install instead of develop
    sys.path.append(os.getcwd())

from src.pylandau_src import pylandau_numba_ext

install_requires = ['numba>=0.53', 'numpy>=1.21', 'scipy>=1.7']


if __name__ == '__main__':
    setup(
        install_requires=install_requires,
        packages=find_packages(),
        include_package_data=True,  # accept all data files and directories matched by MANIFEST.in or found in source control
        package_data={'': ['README.*', 'VERSION'], 'docs': ['*'], 'examples': ['*']},
        ext_modules=[pylandau_numba_ext.distutils_extension()],
        keywords=['Landau', 'Langau', 'PDF'],
        platforms='any'
)