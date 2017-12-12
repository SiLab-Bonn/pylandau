''' Python interface

    Provides access to compiled library and python
    function for additional features
'''

import numpy as np
from scipy.optimize import fmin

# Import library function
from landaulib import (get_landau_pdf, get_gauss_pdf, get_langau_pdf,
                       landau_pdf, langau_pdf)

# These function add a amplitude parameter A and shift the function that mu = MPV
# This is done numerically very simple, stability might suffer


def get_landau(value, mpv=0, eta=1, A=1):
    mpv, eta, sigma, A = _check_parameter(mpv=mpv, eta=eta, sigma=0., A=A)
    mpv_scaled, eta, sigma, A_scaled = _scale_to_mpv(mpv, eta, sigma=0, A=A)

    # Numerical scaling maximum to A
    return get_landau_pdf(value, eta) * A_scaled


def get_langau(value, mpv=0, eta=1, sigma=1, A=1, scale_langau=True):
    mpv, eta, sigma, A = _check_parameter(mpv=mpv, eta=eta, sigma=sigma, A=A)
    mpv_scaled, eta, sigma, A_scaled = _scale_to_mpv(mpv, eta, sigma, A)

    if scale_langau:
        mpv_scaled, _, _, A_scaled = _scale_to_mpv(mpv, eta, sigma, A=A)
    else:
        mpv_scaled, _, _, A_scaled = _scale_to_mpv(mpv, eta, sigma=0, A=A)

    # Numerical scaling maximum to A
    return get_langau_pdf(value, eta, sigma) * A_scaled


def landau(array, mpv=0, eta=1, A=1):
    if (A == 0.):
        return np.zeros_like(array)

    mpv, eta, sigma, A = _check_parameter(mpv=mpv, eta=eta, sigma=0., A=A)
    mpv_scaled, eta, sigma, A_scaled = _scale_to_mpv(mpv, eta, sigma=0., A=A)

    # Numerical scaling maximum to A
    return landau_pdf(array, mpv_scaled, eta) * A_scaled


def langau(array, mpv=0, eta=1, sigma=1, A=1, scale_langau=True):
    ''' Returns a Landau convoluded with a Gaus.

    If scale_langau is true the Langau function maximum is at mpv with amplitude A.
    Otherwise the Landau function maximum is at mpv with amplitude A, thus not the resulting Langau. '''

    if (A == 0.):
        return np.zeros_like(array)

    mpv, eta, sigma, A = _check_parameter(mpv=mpv, eta=eta, sigma=sigma, A=A)

    if sigma == 0:
        return landau(array, mpv=mpv, eta=eta, A=A)

    if scale_langau:
        mpv_scaled, _, _, A_scaled = _scale_to_mpv(mpv, eta, sigma, A=A)
    else:
        mpv_scaled, _, _, A_scaled = _scale_to_mpv(mpv, eta, sigma=0, A=A)

    # Numerical scaling maximum to A
    return langau_pdf(array, mpv_scaled, eta, sigma) * A_scaled


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


def _scale_to_mpv(mu, eta, sigma=0., A=None):
    ''' In the original function definition mu != mpv.
    This is fixed here numerically. Also the amplitude is
    scaled to A. '''

    if sigma > 0:
        # https://github.com/SiLab-Bonn/pyLandau/issues/11
        if abs(mu) > 1.:
            x0 = mu
        else:
            x0 = eta * np.sign(mu)
        res = fmin(lambda x: -langau_pdf(x, mu, eta, sigma), x0=x0,
                   full_output=True, disp=False, xtol=0.000001, ftol=0.000001)
    else:
        # https://github.com/SiLab-Bonn/pyLandau/issues/11
        if abs(mu) > 1.:
            x0 = mu
        else:
            x0 = eta * np.sign(mu)
        res = fmin(lambda x: -landau_pdf(x, mu, eta), x0=x0,
                   full_output=True, disp=False, xtol=0.000001, ftol=0.000001)

    if res[4] != 0:
        raise RuntimeError(
            'Cannot calculate MPV, check function parameters and file bug report!')

    if A:
        A = A / -res[1]
        return mu + (mu - res[0]), eta, sigma, A
    else:
        return mu + (mu - res[0]), eta, sigma
