# pylandau [![Build Status](https://travis-ci.org/SiLab-Bonn/pyLandau.svg?branch=master)](https://travis-ci.org/SiLab-Bonn/pyLandau) [![Build Status](https://ci.appveyor.com/api/projects/status/github/SiLab-Bonn/pyLandau)](https://ci.appveyor.com/project/DavidLP/pyLandau)
A simple [Landau](http://en.wikipedia.org/wiki/Landau_distribution) definition to be used in Python, since no common package (Scipy, Numpy, ...) provides this. Also a fast Landau + Gauss convolution is offered, that is usefull for fitting energy losses of charged particles in matter. The Landau is approximated according to  [Computer Phys. Comm. 31 (1984) 97-111](http://dx.doi.org/10.1016/0010-4655(84)90085-7) and the implementation is from [CERN ROOT Mathlibs] (https://project-mathlibs.web.cern.ch/project-mathlibs/sw/html/PdfFuncMathCore_8cxx_source.html).

# Installation

The code is hosted at PyPI. For installation just type:

pip install pylandau

Otherwise if you want to change the code download it and put it into a directory of your choise. Within the directory run:

python setup.py develop

# Usage

import numpy as np

import pylandau

x = np.arange(0, 100, 0.01)

y_landau = pylandau.landau(x)

y_langau = pylandau.langau(x)


