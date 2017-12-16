# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature =True

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
cnp.import_array()


cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(cnp.ndarray arr, int flags)

cdef extern from "pylandau_src.cpp":
    double * getLandauPDFData(double * & data, const unsigned int & size, const double & mu, const double & eta) except +
    double * getLangauPDFData(double * & data, const unsigned int & size, const double & mu, const double & eta, const double & sigma) except +
    double landauPDF(const double & x, const double & xi, const double & x0) except +
    double landauGaussPDF(const double & x, const double & mu, const double & eta, const double & sigma) except +
    double gaussPDF(const double & x, const double & mu, const double & sigma)


cdef data_to_numpy_array_double(cnp.double_t * ptr, cnp.npy_intp N):
    cdef cnp.ndarray[cnp.double_t, ndim = 1] arr = cnp.PyArray_SimpleNewFromData(1, < cnp.npy_intp * > & N, cnp.NPY_DOUBLE, < cnp.double_t * > ptr)
    PyArray_ENABLEFLAGS(arr, cnp.NPY_OWNDATA)
    return arr

cdef cnp.double_t * result = NULL

# The pdf are defined by the original algorithm where mu != MPV


def get_landau_pdf(value, mu=0, eta=1):
    return landauPDF(< const double&> value, < const double&> mu, < const double&> eta)


def get_gauss_pdf(value, mu=0, sigma=1):
    return gaussPDF(< const double&> value, < const double&> mu, < const double&> sigma)


def get_langau_pdf(value, mu=0, eta=1, sigma=1):
    return landauGaussPDF( < const double&> value, < const double&> mu, < const double&> eta, < const double&> sigma)


def landau_pdf(cnp.ndarray[cnp.double_t, ndim=1] array, mu=0, eta=1):
    mpv, eta, sigma, _ = _check_parameter(mpv=mu, eta=eta, sigma=0.)
    result = getLandauPDFData( < double*& > array.data, < const unsigned int&> array.shape[0], < const double&> mu, < const double&> eta)
    return data_to_numpy_array_double(result, array.shape[0])


def langau_pdf(cnp.ndarray[cnp.double_t, ndim=1] array, mu=0, eta=1, sigma=1):
    mpv, eta, sigma, _ = _check_parameter(mpv=mu, eta=eta, sigma=sigma)
    result = getLangauPDFData( < double*& > array.data, < const unsigned int&> array.shape[0], < const double&> mu, < const double&> eta, < const double&> sigma)
    if result != NULL:
        return data_to_numpy_array_double(result, array.shape[0])


def _check_parameter(mpv, eta, sigma, A=1.):
    if eta < 1e-9:
        print 'WARNING: eta < 1e-9 is not supported. eta set to 1e-9.'
        eta = 1e-9
    if sigma < 0:
        sigma *= -1
    if sigma > 100 * eta:
        print 'WARNING: sigma > 100 * eta can lead to oszillations. Check result.'
    if A < 0.:
        raise ValueError('A has to be >= 0')

    return np.float(mpv), np.float(eta), np.float(sigma), np.float(A)
