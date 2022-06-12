"""
// Taken from LCG ROOT MathLib
// License info:
// Authors: Andras Zsenei & Lorenzo Moneta   06/2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Langaus authors:
//  Based on a Fortran code by R.Fruehwirth (fruhwirth@hephy.oeaw.ac.at)
//  Adapted for C++/ROOT by H.Pernegger (Heinz.Pernegger@cern.ch) and
//  Markus Friedl (Markus.Friedl@cern.ch)
//
//  Adaption for Python by David-Leon Pohl, pohl@physik.uni-bonn.de
"""


import math
import numpy as np
from numba.pycc import CC  # Allow Ahead-Of-Time Compilation using Numba
from numba import njit


# Define the module our extension will be available in
# FIXME: add extension as submodule of pylandau: Unfortunately "module.submodule" syntax is not allowed
# See https://github.com/numba/numba/issues/8013 for more info
pylandau_numba_ext = CC('pylandau_ext')
pylandau_numba_ext.verbose = True


@njit
@pylandau_numba_ext.export('get_landau_pdf', 'f8(f8, f8, f8)')
def landauPDF(x, x0, xi):

    # Define constant parameters
    p1 = (0.4259894875, -0.1249762550, 0.03984243700, -0.006298287635, 0.001511162253)
    q1 = (1.0, -0.3388260629, 0.09594393323, -0.01608042283, 0.003778942063)


    p2 = (0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411, 0.0001283617211)
    q2 = (1.0, 0.7428795082, 0.3153932961, 0.06694219548, 0.008790609714)


    p3 = (0.1788544503, 0.09359161662, 0.006325387654, 0.00006611667319, -0.000002031049101)
    q3 = (1.0, 0.6097809921, 0.2560616665, 0.04746722384, 0.006957301675)


    p4 = (0.9874054407, 118.6723273, 849.2794360, -743.7792444, 427.0262186)
    q4 = (1.0, 106.8615961, 337.6496214, 2016.712389, 1597.063511)


    p5 = (1.003675074, 167.5702434, 4789.711289, 21217.86767, -22324.94910)
    q5 = (1.0, 156.9424537, 3745.310488, 9834.698876, 66924.28357)

    p6 = (1.000827619, 664.9143136, 62972.92665, 475554.6998, -5743609.109)
    q6 = (1.0, 651.4101098, 56974.73333, 165917.4725, -2815759.939)

    a1 = (0.04166666667, -0.01996527778, 0.02709538966)
    a2 = (-1.845568670, -4.284640743)

    if xi <= 0:
        return 0

    v = (x - x0) / xi

    if v < -5.5:
        u = math.exp(v + 1.0)
        if u < 1e-10:
            return 0.0
        ue = math.exp(-1 / u)
        us = math.sqrt(u)
        denlan = 0.3989422803 * (ue / us) * (1 + (a1[0] + (a1[1] + a1[2] * u) * u) * u)

    elif v < -1:
        u = math.exp(-v - 1)
        denlan = math.exp(-u) * math.sqrt(u) * (p1[0] + (p1[1] + (p1[2] + (p1[3] + p1[4] * v) * v) * v) * v) / (q1[0] + (q1[1] + (q1[2] + (q1[3] + q1[4] * v) * v) * v) * v)

    elif v < 1:
        denlan = (p2[0] + (p2[1] + (p2[2] + (p2[3] + p2[4] * v) * v) * v) * v) / (q2[0] + (q2[1] + (q2[2] + (q2[3] + q2[4] * v) * v) * v) * v)

    elif v < 5:
        denlan = (p3[0] + (p3[1] + (p3[2] + (p3[3] + p3[4] * v) * v) * v) * v) / (q3[0] + (q3[1] + (q3[2] + (q3[3] + q3[4] * v) * v) * v) * v)
    
    elif v < 12:
        u = 1 / v
        denlan = u * u * (p4[0] + (p4[1] + (p4[2] + (p4[3] + p4[4] * u) * u) * u) * u) / (q4[0] + (q4[1] + (q4[2] + (q4[3] + q4[4] * u) * u) * u) * u)

    elif v < 50:
        u = 1 / v
        denlan = u * u * (p5[0] + (p5[1] + (p5[2] + (p5[3] + p5[4] * u) * u) * u) * u) / (q5[0] + (q5[1] + (q5[2] + (q5[3] + q5[4] * u) * u) * u) * u)

    elif v < 300:
        u = 1 / v
        denlan = u * u * (p6[0] + (p6[1] + (p6[2] + (p6[3] + p6[4] * u) * u) * u) * u) / (q6[0] + (q6[1] + (q6[2] + (q6[3] + q6[4] * u) * u) * u) * u)

    else:
        u = 1 / (v - v * math.log(v) / (v + 1))
        denlan = u * u * (1 + (a2[0] + a2[1] * u) * u)

    return denlan / xi


@njit
@pylandau_numba_ext.export('get_gauss_pdf', 'f8(f8, f8, f8)')
def gaussPDF(x, mu, sigma):
    return 0.3989422804014 / sigma * math.exp(- math.pow((x - mu), 2) / (2 * math.pow(sigma, 2)))


@njit
@pylandau_numba_ext.export('get_langau_pdf', 'f8(f8, f8, f8, f8)')
def landauGaussPDF(x, mu, eta, sigma):
    
    mpshift = 0.  # -0.22278298  # Landau maximum location shift in original code is wrong, since the shift does not depend on mu only

    nc_steps = 100  # Number of convolution steps
    sc = 8  # Convolution extends to +-sc Gaussian sigmas

    # Convolution steps have to be increased if sigma > eta * 5 to get stable solution that does not oscillate, addresses #1
    if sigma > 3 * eta:
        nc_steps *= int(sigma / eta / 3.)

    # Do not use too many convolution steps to save time
    if nc_steps > 100000:
        nc_steps = 100000

    # MP shift correction
    mpc = mu - mpshift

    # Range of convolution integral
    x_low = x - sc * sigma
    x_upp = x + sc * sigma

    step = (x_upp - x_low) / float(nc_steps)

    res = 0.

    for i in range(1, int(nc_steps/2 + 1)):

        xx = x_low + (i - 0.5) * step
        fland = landauPDF(xx, mpc, eta) / eta
        res += fland * gaussPDF(x, xx, sigma)

        xx = x_upp - (i - 0.5) * step
        fland = landauPDF(xx, mpc, eta) / eta
        res += fland * gaussPDF(x, xx, sigma)

    return step * res



@pylandau_numba_ext.export('landau_pdf', 'f8[:](f8[:], f8, f8)')
def landau_pdf(array, mu, eta):
    res = np.zeros_like(array)
    for i in range(array.shape[0]):
        res[i] = landauPDF(array[i], mu, eta)
    return res



@pylandau_numba_ext.export('langau_pdf', 'f8[:](f8[:], f8, f8, f8)')
def langau_pdf(array, mu, eta, sigma):
    res = np.zeros_like(array)
    for i in range(array.shape[0]):
        res[i] = landauGaussPDF(array[i], mu, eta, sigma)
    return res


if __name__ == '__main__':
    pylandau_numba_ext.compile()
