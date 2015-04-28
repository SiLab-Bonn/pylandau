# pyLandau

A simple Landau PDF definition to be used in Python, since no common package (Scipy, Numpy, ...) provides this.
The Landau approximation from  [Computer Phys. Comm. 31 (1984) 97-111](http://dx.doi.org/10.1016/0010-4655(84)90085-7) 
with its implementation from the [CERN ROOT Mathlibs] (https://project-mathlibs.web.cern.ch/project-mathlibs/sw/html/PdfFuncMathCore_8cxx_source.html).

# Installation

pip install git+git://github.com/SiLab-Bonn/pyLandau@master

# Usage

import numpy as np

from pyLandau import landau

x = np.arange(0, 100, 0.01)

y = landau.landau(x)


