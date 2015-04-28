# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
cnp.import_array()

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)

cdef extern from "landau_src.cpp":
    double * getLandauData(double*& data, const unsigned int& size, const double& eta, const double& mu) except +

cdef data_to_numpy_array_double(cnp.double_t* ptr, cnp.npy_intp N):
    cdef cnp.ndarray[cnp.double_t, ndim = 1] arr = cnp.PyArray_SimpleNewFromData(1, <cnp.npy_intp*> &N, cnp.NPY_DOUBLE, <cnp.double_t*> ptr)
    PyArray_ENABLEFLAGS(arr, cnp.NPY_OWNDATA)
    return arr

def landau_pdf(cnp.ndarray[cnp.double_t, ndim=1] data, eta=1, mu=0):
    cdef cnp.double_t* result = NULL
    result = getLandauData(<double*&> data.data, <const unsigned int&> data.shape[0], < const double&> eta, < const double&> mu)
    if result != NULL:
        return data_to_numpy_array_double(result, data.shape[0])

def landau(cnp.ndarray[cnp.double_t, ndim=1] data, A=1, eta=1, mu=0):
    landau = landau_pdf(data, eta, mu)
    return (landau / np.amax(landau) * A)
