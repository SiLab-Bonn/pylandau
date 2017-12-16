''' Python interface

    Provides access to compiled library and python
    function for additional features
'''

from __future__ import print_function
import numpy as np
from scipy import optimize

# Import library function
from pylandau.landaulib import (get_landau_pdf, get_gauss_pdf, get_langau_pdf,
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
    y = landau_pdf(array, mpv_scaled, eta)
    return y * A_scaled


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
    y = langau_pdf(array, mpv_scaled, eta, sigma)
    return y * A_scaled


def _check_parameter(mpv, eta, sigma, A=1.):
    if eta < 1e-9:
        print('WARNING: eta < 1e-9 is not supported. eta set to 1e-9.')
        eta = 1e-9
    if sigma < 0:
        sigma *= -1
    if sigma > 100 * eta:
        print(
            'WARNING: sigma > 100 * eta can lead to oszillations. Check result.')
    if A < 0.:
        raise ValueError('A has to be >= 0')

    return np.float(mpv), np.float(eta), np.float(sigma), np.float(A)


def _scale_to_mpv(mu, eta, sigma=0., A=None):
    ''' In the original function definition mu != mpv.
    This is fixed here numerically. Also the amplitude is
    scaled to A. '''

    # Shift mu to enhance convergence for large mu in numerical optimizers
    # https://github.com/SiLab-Bonn/pylandau/issues/6
    mu_in = mu
    mu = 0.

    if sigma > 0:
        dx = 3. * eta * sigma
        if dx < 1:
            dx = 1.
        res = optimize.minimize_scalar(lambda x, mu, eta, sigma: -get_langau_pdf(x, mu, eta, sigma),
                                       bounds=(mu - dx, mu + dx),
                                       args=(mu, eta, sigma),
                                       method='bounded',
                                       options={'xatol': 1e-09})
    else:
        dx = 3. * eta
        if dx < 1:
            dx = 1.
        res = optimize.minimize_scalar(lambda x, mu, eta: -get_landau_pdf(x, mu, eta),
                                       bounds=(mu - dx, mu + dx),
                                       args=(mu, eta),
                                       method='bounded',
                                       options={'xatol': 1e-09})
    if not res.success or res.fun == 0.:
        raise RuntimeError('Cannot calculate MPV'
                           ', check function parameters and file bug report!')

    if A:
        A = A / -res.fun
        return mu_in - res.x, eta, sigma, A
    else:
        return mu_in - res.x, eta, sigma
