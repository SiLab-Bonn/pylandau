# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature =True

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
cnp.import_array()

from scipy.optimize import fmin


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
    return landauPDF( < const double&> value, < const double&> mu, < const double&> eta)


def get_gauss_pdf(value, mu=0, sigma=1):
    return gaussPDF( < const double&> value, < const double&> mu, < const double&> sigma)


def get_langau_pdf(value, mu=0, eta=1, sigma=1):
    return landauGaussPDF(< const double&> value, < const double&> mu, < const double&> eta, < const double&> sigma)


def landau_pdf(cnp.ndarray[cnp.double_t, ndim=1] array, mu=0, eta=1):
    mpv, eta, sigma, _ = _check_parameter(mpv=mu, eta=eta, sigma=0.)
    result = getLandauPDFData(< double*& > array.data, < const unsigned int&> array.shape[0], < const double&> mu, < const double&> eta)
    return data_to_numpy_array_double(result, array.shape[0])


def langau_pdf(cnp.ndarray[cnp.double_t, ndim=1] array, mu=0, eta=1, sigma=1):
    mpv, eta, sigma, _ = _check_parameter(mpv=mu, eta=eta, sigma=sigma)
    result = getLangauPDFData(< double*& > array.data, < const unsigned int&> array.shape[0], < const double&> mu, < const double&> eta, < const double&> sigma)
    if result != NULL:
        return data_to_numpy_array_double(result, array.shape[0])


# These function add a amplitude parameter A and shift the function that mu = MPV
# This is done numerically very simple, stability might suffer

def get_landau(value, mpv=0, eta=1, A=1):
    mpv, eta, sigma, A = _check_parameter(mpv=mpv, eta=eta, sigma=0., A=A)
    mpv_scaled, eta, sigma, A_scaled = _scale_to_mpv(mpv, eta, sigma=0, A=A)

    return get_landau_pdf(value, eta) * A_scaled  # Numerical scaling maximum to A


def get_langau(value, mpv=0, eta=1, sigma=1, A=1, scale_langau=True):
    mpv, eta, sigma, A = _check_parameter(mpv=mpv, eta=eta, sigma=sigma, A=A)
    mpv_scaled, eta, sigma, A_scaled = _scale_to_mpv(mpv, eta, sigma, A)

    if scale_langau:
        mpv_scaled, _, _, A_scaled = _scale_to_mpv(mpv, eta, sigma, A=A)
    else:
        mpv_scaled, _, _, A_scaled = _scale_to_mpv(mpv, eta, sigma=0, A=A)

    return get_langau_pdf(value, eta, sigma) * A_scaled  # Numerical scaling maximum to A


def landau(cnp.ndarray[cnp.double_t, ndim=1] array, mpv=0, eta=1, A=1):
    mpv, eta, sigma, A = _check_parameter(mpv=mpv, eta=eta, sigma=0., A=A)
    mpv_scaled, eta, sigma, A_scaled = _scale_to_mpv(mpv, eta, sigma=0., A=A)

    return landau_pdf(array, mpv_scaled, eta) * A_scaled  # Numerical scaling maximum to A


def langau(cnp.ndarray[cnp.double_t, ndim=1] array, mpv=0, eta=1, sigma=1, A=1, scale_langau=True):
    ''' Returns a Landau convoluded with a Gaus.

    If scale_langau is true the Langau function maximum is at mpv with amplitude A.
    Otherwise the Landau function maximum is at mpv with amplitude A, thus not the resulting Langau. '''
    mpv, eta, sigma, A = _check_parameter(mpv=mpv, eta=eta, sigma=sigma, A=A)

    if scale_langau:
        mpv_scaled, _, _, A_scaled = _scale_to_mpv(mpv, eta, sigma, A=A)
    else:
        mpv_scaled, _, _, A_scaled = _scale_to_mpv(mpv, eta, sigma=0, A=A)

    return langau_pdf(array, mpv_scaled, eta, sigma) * A_scaled  # Numerical scaling maximum to A


def _check_parameter(mpv, eta, sigma, A=1.):
    if eta < 1:
        print 'WARNING: eta < 1 not supported and set to 1! Scale x to fix.'
        eta = 1
    if sigma < 0:
        sigma *= -1
    if sigma > 100 * eta:
        print 'WARNING: sigma > 100 * eta can lead to oszillations. Check result.'
    if not A:
        RuntimeError('A has to be > 0')

    return np.float(mpv), np.float(eta), np.float(sigma), np.float(A)


def _scale_to_mpv(mu, eta, sigma=0., A=None):
    ''' In the original function definition mu != mpv.
    This is fixed here numerically and the amplitude is
    scaled to A. '''

    if sigma > 0:
        res = fmin(lambda x: -langau_pdf(x, mu, eta, sigma), x0=mu, full_output=True, disp=False, xtol=0.000001, ftol=0.000001)
    else:
        res = fmin(lambda x: -landau_pdf(x, mu, eta), x0=mu, full_output=True, disp=False, xtol=0.000001, ftol=0.000001)

    if res[4] != 0:
        raise RuntimeError('Cannot calculate MPV, check function parameters and file bug report!')

    if A:
        A = A / -res[1]
        return mu + (mu - res[0]), eta, sigma, A
    else:
        return mu + (mu - res[0]), eta, sigma
