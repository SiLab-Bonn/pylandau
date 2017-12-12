''' Check pylandau with random inputs
'''
import unittest

from hypothesis import given, assume
import hypothesis.strategies as st
import numpy as np

import pylandau
from tests import constrains


class TestFuzzing(unittest.TestCase):

    @given(st.floats(constrains.LANDAU_MIN_MPV,
                     constrains.LANDAU_MAX_MPV,
                     allow_nan=False,
                     allow_infinity=False))
    def test_landau_mpv(self, mpv):
        ''' Check Landau MPV position '''
        x = np.linspace(mpv - 10, mpv + 10, 1000)
        y = pylandau.landau(x, mpv=mpv, eta=1., A=1.)
        delta = x[1] - x[0]
        self.assertAlmostEqual(x[np.argmax(y)], mpv, delta=delta)

    @given(st.floats(constrains.LANGAU_MIN_MPV,
                     constrains.LANGAU_MAX_MPV,
                     allow_nan=False,
                     allow_infinity=False))
    def test_langau_mpv(self, mpv):
        ''' Check Langau MPV position '''
        x = np.linspace(mpv - 10, mpv + 10, 1000)
        y = pylandau.langau(x, mpv=mpv, eta=1., sigma=1., A=1.)
        delta = x[1] - x[0]
        self.assertAlmostEqual(x[np.argmax(y)], mpv, delta=delta)

    @given(st.floats(constrains.LANDAU_MIN_A,
                     constrains.LANDAU_MAX_A,
                     allow_nan=False,
                     allow_infinity=False))
    def test_landau_A(self, A):
        ''' Check Landau amplitude '''
        mpv = 1.
        x = np.linspace(mpv - 10, mpv + 10, 1000)
        y = pylandau.landau(x, mpv=mpv, eta=1., A=A)
        self.assertAlmostEqual(y.max(), A, delta=1e-4 * A)

    @given(st.floats(constrains.LANGAU_MIN_A,
                     constrains.LANGAU_MAX_A,
                     allow_nan=False,
                     allow_infinity=False))
    def test_langau_A(self, A):
        ''' Check Langau amplitude '''
        mpv = 1.
        x = np.linspace(mpv - 10, mpv + 10, 1000)
        y = pylandau.langau(x, mpv=mpv, eta=1., sigma=1., A=A)
        self.assertAlmostEqual(y.max(), A, delta=1e-4 * A)

    @given(st.floats(constrains.LANDAU_MIN_MPV,
                     constrains.LANDAU_MAX_MPV,
                     allow_nan=False,
                     allow_infinity=False),
           st.floats(constrains.LANDAU_MIN_ETA,
                     constrains.LANDAU_MAX_ETA,
                     allow_nan=False,
                     allow_infinity=False),
           st.floats(constrains.LANDAU_MIN_A,
                     constrains.LANDAU_MAX_A,
                     allow_nan=False,
                     allow_infinity=False))
    def test_landau_stability(self, mpv, eta, A):
        ''' Check Landau outputs for same input parameters '''
        x = np.linspace(mpv - 5 * eta, mpv + 5 * eta, 1000)
        y_1 = pylandau.landau(x, mpv=mpv, eta=eta, A=A)
        y_2 = pylandau.landau(x, mpv=mpv, eta=eta, A=A)
        self.assertTrue(np.all(y_1 == y_2))

    @given(st.floats(constrains.LANGAU_MIN_MPV,
                     constrains.LANGAU_MAX_MPV,
                     allow_nan=False,
                     allow_infinity=False),
           st.floats(constrains.LANGAU_MIN_ETA,
                     constrains.LANGAU_MAX_ETA,
                     allow_nan=False,
                     allow_infinity=False),
           st.floats(constrains.LANGAU_MIN_SIGMA,
                     constrains.LANGAU_MAX_SIGMA,
                     allow_nan=False,
                     allow_infinity=False),
           st.floats(constrains.LANGAU_MIN_A,
                     constrains.LANGAU_MAX_A,
                     allow_nan=False,
                     allow_infinity=False)
           )
    def test_langau_stability(self, mpv, eta, sigma, A):
        ''' Check Langau outputs for same input parameters '''
        # Correct input to avoid oscillations
        if sigma > 100 * eta:
            sigma = eta
        assume(sigma * eta < constrains.LANGAU_MAX_ETA_SIGMA)
        x = np.linspace(mpv - 5 * sigma * eta, mpv + 5 * sigma * eta, 1000)
        y_1 = pylandau.langau(x, mpv=mpv, eta=eta, sigma=sigma, A=A)
        y_2 = pylandau.langau(x, mpv=mpv, eta=eta, sigma=sigma, A=A)
        self.assertTrue(np.all(y_1 == y_2))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFuzzing)
    unittest.TextTestRunner(verbosity=2).run(suite)
