''' Script to check pylandau.
'''
import unittest

from hypothesis import given, seed
import hypothesis.extra.numpy as nps
import hypothesis.strategies as st
from hypothesis.extra.numpy import unsigned_integer_dtypes
import numpy as np
from scipy.integrate import quad as integrate

import pylandau


def drange(start, stop, step):  # range with floats
    r = start
    while r < stop:
        yield r
        r += step


def F(function, x, args=()):
    return (function(x_i, *args) for x_i in x)


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):  # remove created files
        pass

    def test_landau_pdf_integral(self):
        ''' Check that Landau pdf integral is 1 '''
        result, _ = integrate(
            pylandau.get_landau_pdf, 0, 10000, args=(10, 1))
        self.assertAlmostEqual(result, 1, delta=1e-3)

    def test_langau_pdf_integral(self):
        ''' Check that Langau pdf integral is 1 '''
        result, _ = integrate(
            pylandau.get_langau_pdf, -10000, 10000, args=(10, 1, 3))
        self.assertAlmostEqual(result, 1, delta=1e-3)

    def test_gauss_pdf_integral(self):
        ''' Check that Gauss pdf integral is 1 '''
        result, _ = integrate(
            pylandau.get_gauss_pdf, 0, 10000, args=(10, 3))
        self.assertAlmostEqual(result, 1, delta=1e-3)

    @given(st.floats(1, 1e5, allow_nan=False, allow_infinity=False))
    def test_landau_mpv_pos(self, mpv):
        x = np.linspace(mpv - 10, mpv + 10, 1000)
        y = pylandau.landau(x, mpv=mpv, eta=1., A=1.)
        delta = x[1] - x[0]
        self.assertAlmostEqual(x[np.argmax(y)], mpv, delta=delta)

    @given(st.floats(-1e5, -1, allow_nan=False, allow_infinity=False))
    def test_landau_mpv_neg(self, mpv):
        x = np.linspace(mpv - 10, mpv + 10, 1000)
        y = pylandau.landau(x, mpv=mpv, eta=1., A=1.)
        delta = x[1] - x[0]
        self.assertAlmostEqual(x[np.argmax(y)], mpv, delta=delta)

    @given(st.floats(1, 1e5, allow_nan=False, allow_infinity=False))
    def test_langau_mpv_pos(self, mpv):
        x = np.linspace(mpv - 10, mpv + 10, 1000)
        y = pylandau.langau(x, mpv=mpv, eta=1., sigma=1., A=1.)
        delta = x[1] - x[0]
        self.assertAlmostEqual(x[np.argmax(y)], mpv, delta=delta)

    @given(st.floats(-1e5, -1, allow_nan=False, allow_infinity=False))
    def test_langau_mpv_neg(self, mpv):
        x = np.linspace(mpv - 10, mpv + 10, 1000)
        y = pylandau.langau(x, mpv=mpv, eta=1., sigma=1., A=1.)
        delta = x[1] - x[0]
        self.assertAlmostEqual(x[np.argmax(y)], mpv, delta=delta)

    @given(st.floats(np.nextafter(0, 1), 1e5, allow_nan=False, allow_infinity=False))
    def test_landau_A(self, A):
        mpv = 1.
        x = np.linspace(mpv - 10, mpv + 10, 1000)
        y = pylandau.landau(x, mpv=mpv, eta=1., A=A)
        self.assertAlmostEqual(y.max(), A, delta=1e-4 * A)

    @given(st.floats(np.nextafter(0, 1), 1e5, allow_nan=False, allow_infinity=False))
    def test_langau_A(self, A):
        mpv = 1.
        x = np.linspace(mpv - 10, mpv + 10, 1000)
        y = pylandau.langau(x, mpv=mpv, eta=1., sigma=1., A=A)
        self.assertAlmostEqual(y.max(), A, delta=1e-4 * A)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner(verbosity=2).run(suite)
