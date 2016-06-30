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

# The pdf are defined by the original algorithm where mu != MPV
def get_landau_pdf(value, mu=0, eta=1):
    return landauPDF(<const double&> value, <const double&> mu, <const double&> eta)


def get_gauss_pdf(value, mu=0, sigma=1):
    return gaussPDF(<const double&> value, <const double&> mu, <const double&> sigma)


def get_langau_pdf(value, mu=0, eta=1, sigma=1):
    return landauGaussPDF(<const double&> value, <const double&> mu, <const double&> eta, <const double&> sigma)


def landau_pdf(cnp.ndarray[cnp.double_t, ndim=1] array, mu=0, eta=1):
    result = getLandauPDFData(< double*& > array.data, < const unsigned int&> array.shape[0], < const double&> mu, < const double&> eta)
    return data_to_numpy_array_double(result, array.shape[0])


def langau_pdf(cnp.ndarray[cnp.double_t, ndim=1] array, mu=0, eta=1, sigma=1):
    result = getLangauPDFData(< double*& > array.data, < const unsigned int&> array.shape[0], < const double&> mu, < const double&> eta, < const double&> sigma)
    if result != NULL:
        return data_to_numpy_array_double(result, array.shape[0])


# These function add a amplitude parameter A and shift the function that mu = MPV
# This is done numerically very simple, stability might suffer 

def get_landau(value, mpv=0, eta=1, A=1, precision=0.01):
    # Determine maximum and MPV shift
    x = np.arange(mpv - 5 * eta, mpv + 5 * eta, mpv * precision)  # Has to cover maximum, precision defines x deviation around maximum
    y_pre = landau_pdf(x, mpv, eta)  # Get Landau in original definition first to be able to correct for MPV / mu shift
    index_maximum = np.argmax(y_pre)
    maximum = y_pre[index_maximum]
    mpv_shift = mpv - x[index_maximum]

    y = get_landau_pdf(value, mpv + mpv_shift, eta)  # Shift original mu parameter to get a Landau with mu = MPV
    return y / maximum * A  # Numerical scaling maximum to A


def get_langau(value, mpv=0, eta=1, sigma=1, A=1, precision=0.01):
    # Determine maximum and MPV shift
    x = np.arange(mpv - 5 * eta, mpv + 5 * eta, mpv * precision)  # Has to cover maximum, precision defines x deviation around maximum
    y_pre = langau_pdf(x, mpv, eta, sigma)  # Get Landau in original definition first to be able to correct for MPV / mu shift
    index_maximum = np.argmax(y_pre)
    maximum = y_pre[index_maximum]
    mpv_shift = mpv - x[index_maximum]

    y = get_langau_pdf(value, mpv + mpv_shift, eta, sigma)
    return y / maximum * A  # Numerical scaling maximum to A


def landau(cnp.ndarray[cnp.double_t, ndim=1] array, mpv=0, eta=1, A=1, precision=0.01):
    # Determine maximum and MPV shift
    x = np.arange(mpv - 5 * eta, mpv + 5 * eta, mpv * precision)  # Has to cover maximum, precision defines x deviation around maximum
    y_pre = landau_pdf(x, mpv, eta)  # Get Landau in original definition first to be able to correct for MPV / mu shift
    index_maximum = np.argmax(y_pre)
    maximum = y_pre[index_maximum]
    mpv_shift = mpv - x[index_maximum]

    y = landau_pdf(array, mpv + mpv_shift, eta)
    return y / maximum * A  # Numerical scaling maximum to A
 

def langau(cnp.ndarray[cnp.double_t, ndim=1] array, mpv=0, eta=1, sigma=1, A=1, precision=0.01):
    # Determine maximum and MPV shift
    x = np.arange(mpv - 5 * eta, mpv + 5 * eta, mpv * precision)  # Has to cover maximum, precision defines x deviation around maximum
    y_pre = langau_pdf(x, mpv, eta, sigma)  # Get Landau in original definition first to be able to correct for MPV / mu shift
    index_maximum = np.argmax(y_pre)
    maximum = y_pre[index_maximum]
    mpv_shift = mpv - x[index_maximum]

    y = langau_pdf(array, mpv + mpv_shift, eta, sigma)
    return y / maximum * A  # Numerical scaling maximum to A
