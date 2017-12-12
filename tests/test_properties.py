''' Script to check pylandau properties.
'''
import unittest

import numpy as np
from hypothesis import given
import hypothesis.strategies as st

from scipy.integrate import quad as integrate
from scipy.optimize import fmin

import pylandau
from tests import constrains


class TestProperties(unittest.TestCase):

    def test_landau_pdf_integral(self):
        ''' Check that Landau pdf integral is 1 '''
        result, _ = integrate(pylandau.get_landau_pdf,
                              0, 10000, args=(10, 1))
        self.assertAlmostEqual(result, 1, delta=1e-3)

    def test_langau_pdf_integral(self):
        ''' Check that Langau pdf integral is 1 '''
        result, _ = integrate(pylandau.get_langau_pdf,
                              -10000, 10000, args=(10, 1, 3))
        self.assertAlmostEqual(result, 1, delta=1e-3)

    def test_gauss_pdf_integral(self):
        ''' Check that Gauss pdf integral is 1 '''
        result, _ = integrate(pylandau.get_gauss_pdf,
                              0, 10000, args=(10, 3))
        self.assertAlmostEqual(result, 1, delta=1e-3)

    @given(st.floats(constrains.LANDAU_PDF_MIN_MU,
                     constrains.LANDAU_PDF_MAX_MU,
                     allow_nan=False,
                     allow_infinity=False))
    def test_landau_pdf_mu(self, mu):
        ''' Check that Landau pdf mu position '''
        # For the given parameters (mu=mu, eta=1) MPV is expected
        # at x = mu-0.22278
        # See also: https://root.cern.ch/root/html524/TMath.html#TMath:Landau

        # x0 > 1 otherwise scipy.optimize.fmin terminates too early
        if abs(mu) > 1.:
            x0 = mu
        else:
            x0 = 1. * np.sign(mu)

        mpv = fmin(lambda x: -pylandau.get_landau_pdf(x, mu=mu, eta=1.),
                   x0=x0, full_output=True, disp=False,
                   xtol=0.000001, ftol=0.000001)[0][0]
        self.assertAlmostEqual(mpv, mu - 0.22278, delta=1e9)

    @given(st.tuples(
        # mpv
        st.floats(constrains.LANDAU_MIN_MPV,
                  constrains.LANDAU_MAX_MPV,
                  allow_nan=False,
                  allow_infinity=False),
        # eta
        st.floats(constrains.LANDAU_MIN_ETA,
                  constrains.LANDAU_MAX_ETA,
                  allow_nan=False,
                  allow_infinity=False),
        # A
        st.floats(constrains.LANDAU_MIN_A,
                  constrains.LANDAU_MAX_A,
                  allow_nan=False,
                  allow_infinity=False),)
           )
    def test_landau_fallback(self, pars):
        ''' Check if langau is landau for sigma = 0 '''
        (mpv, eta, A) = pars
        x = np.linspace(mpv - 5 * eta, mpv + 5 * eta, 1000)
        y_1 = pylandau.landau(x, mpv=mpv, eta=eta, A=A)
        y_2 = pylandau.langau(x, mpv=mpv, eta=eta, sigma=0., A=A)
        self.assertTrue(np.all(y_1 == y_2))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestProperties)
    unittest.TextTestRunner(verbosity=2).run(suite)
