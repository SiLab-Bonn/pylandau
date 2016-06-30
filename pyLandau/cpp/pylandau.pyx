# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
cnp.import_array()

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)

cdef extern from "pylandau_src.cpp":
    double* getLandauPDFData(double*& data, const unsigned int& size, const double& mu, const double& eta) except +
    double* getLangauPDFData(double*& data, const unsigned int& size, const double& mu, const double& eta, const double& sigma) except +
    double landauPDF(const double& x, const double& xi, const double& x0) except +
    double landauGaussPDF(const double& x, const double& mu, const double& eta, const double& sigma) except +
    double gaussPDF(const double& x, const double& mu, const double& sigma)


cdef data_to_numpy_array_double(cnp.double_t * ptr, cnp.npy_intp N):
    cdef cnp.ndarray[cnp.double_t, ndim = 1] arr = cnp.PyArray_SimpleNewFromData(1, < cnp.npy_intp * > & N, cnp.NPY_DOUBLE, < cnp.double_t * > ptr)
    PyArray_ENABLEFLAGS(arr, cnp.NPY_OWNDATA)
    return arr

cdef cnp.double_t* result = NULL


def get_landau_pdf(value, mu=0, eta=1):
    return landauPDF(<const double&> value, <const double&> mu, <const double&> eta)


def get_gauss_pdf(value, mu=0, sigma=1):
    return gaussPDF(<const double&> value, <const double&> mu, <const double&> sigma)


def get_langau_pdf(value, mu=0, eta=1, sigma=1):
    return landauGaussPDF(<const double&> value, <const double&> mu, <const double&> eta, <const double&> sigma)


def get_landau(value, mu=0, eta=1, A=1):
    result = get_landau_pdf(value, mu, eta)
    maximum = np.max(landau_pdf(np.arange(mu - 5 * eta, mu + 5 * eta, 0.1 * eta), mu, eta))
    return result / maximum * A  # Numerical scaling maximum to A


def get_langau(value, mu=0, eta=1, sigma=1, A=1):
    result = get_langau_pdf(value, mu, eta, sigma)
    maximum = np.max(langau_pdf(np.arange(mu - 5 * eta, mu + 5 * eta, 0.1 * eta), mu, eta, sigma))
    return result / maximum * A  # Numerical scaling maximum to A


def landau_pdf(cnp.ndarray[cnp.double_t, ndim=1] array, mu=0, eta=1):
    result = getLandauPDFData(< double*& > array.data, < const unsigned int&> array.shape[0], < const double&> mu, < const double&> eta)
    return data_to_numpy_array_double(result, array.shape[0])


def langau_pdf(cnp.ndarray[cnp.double_t, ndim=1] array, mu=0, eta=1, sigma=1):
    result = getLangauPDFData(< double*& > array.data, < const unsigned int&> array.shape[0], < const double&> mu, < const double&> eta, < const double&> sigma)
    if result != NULL:
        return data_to_numpy_array_double(result, array.shape[0])


def landau(cnp.ndarray[cnp.double_t, ndim=1] array, mu=0, eta=1, A=1):
    landau = landau_pdf(array, mu, eta)
    maximum = np.max(landau_pdf(np.arange(mu - 5 * eta, mu + 5 * eta, 0.1 * eta), mu, eta))
    return landau / maximum * A  # Numerical scaling maximum to A


def langau(cnp.ndarray[cnp.double_t, ndim=1] array, mu=0, eta=1, sigma=1, A=1):
    langau = langau_pdf(array, mu, eta, sigma)
    maximum = np.max(langau_pdf(np.arange(mu - 5 * eta, mu + 5 * eta, 0.1 * eta), mu, eta, sigma))
    return langau / maximum * A  # Numerical scaling maximum to A
