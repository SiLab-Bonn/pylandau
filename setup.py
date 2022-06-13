#!/usr/bin/env python
from setuptools import setup, find_packages  # This setup relies on setuptools since distutils is insufficient and badly hacked code
from src.pylandau_src import pylandau_numba_ext

install_requires = ['numba>=0.53', 'numpy>=1.21', 'scipy>=1.6']


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