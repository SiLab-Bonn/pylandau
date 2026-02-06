''' Check pylandau with random inputs
'''
import pytest

import numpy as np
import pylandau

from hypothesis import given, assume
import hypothesis.strategies as st

from tests import constraints


class TestFuzzing():

    @given(st.floats(constraints.LANDAU_MIN_MPV,
                     constraints.LANDAU_MAX_MPV,
                     allow_nan=False,
                     allow_infinity=False))
    def test_landau_mpv(self, mpv):
        ''' Check Landau MPV position '''
        x = np.linspace(mpv - 10, mpv + 10, 1000)
        y = pylandau.landau(x, mpv=mpv, eta=1., A=1.)
        delta = x[1] - x[0]
        assert float(x[np.argmax(y)]) == pytest.approx(mpv, abs=delta)

    @given(st.floats(constraints.LANGAU_MIN_MPV,
                     constraints.LANGAU_MAX_MPV,
                     allow_nan=False,
                     allow_infinity=False))
    def test_langau_mpv(self, mpv):
        ''' Check Langau MPV position '''
        x = np.linspace(mpv - 10, mpv + 10, 1000)
        y = pylandau.langau(x, mpv=mpv, eta=1., sigma=1., A=1.)
        delta = x[1] - x[0]
        assert x[np.argmax(y)] == pytest.approx(mpv, abs=delta)

    @given(st.floats(constraints.LANDAU_MIN_A,
                     constraints.LANDAU_MAX_A,
                     allow_nan=False,
                     allow_infinity=False))
    def test_landau_A(self, A):
        ''' Check Landau amplitude '''
        mpv = 1.
        x = np.linspace(mpv - 10, mpv + 10, 1000)
        y = pylandau.landau(x, mpv=mpv, eta=1., A=A)
        assert y.max() == pytest.approx(A, abs=1e-4 * A)

    @given(st.floats(constraints.LANGAU_MIN_A,
                     constraints.LANGAU_MAX_A,
                     allow_nan=False,
                     allow_infinity=False))
    def test_langau_A(self, A):
        ''' Check Langau amplitude '''
        mpv = 1.
        x = np.linspace(mpv - 10, mpv + 10, 1000)
        y = pylandau.langau(x, mpv=mpv, eta=1., sigma=1., A=A)
        assert y.max() == pytest.approx(A, abs=1e-4 * A)

    @given(st.floats(constraints.LANDAU_MIN_MPV,
                     constraints.LANDAU_MAX_MPV,
                     allow_nan=False,
                     allow_infinity=False),
           st.floats(constraints.LANDAU_MIN_ETA,
                     constraints.LANDAU_MAX_ETA,
                     allow_nan=False,
                     allow_infinity=False),
           st.floats(constraints.LANDAU_MIN_A,
                     constraints.LANDAU_MAX_A,
                     allow_nan=False,
                     allow_infinity=False))
    
    def test_landau_stability(self, mpv, eta, A):
        ''' Check Landau outputs for same input parameters '''
        x = np.linspace(mpv - 5 * eta, mpv + 5 * eta, 1000)
        y_1 = pylandau.landau(x, mpv=mpv, eta=eta, A=A)
        y_2 = pylandau.landau(x, mpv=mpv, eta=eta, A=A)
        assert np.all(y_1 == y_2) == True

    @given(st.floats(constraints.LANGAU_MIN_MPV,
                     constraints.LANGAU_MAX_MPV,
                     allow_nan=False,
                     allow_infinity=False),
           st.floats(constraints.LANGAU_MIN_ETA,
                     constraints.LANGAU_MAX_ETA,
                     allow_nan=False,
                     allow_infinity=False),
           st.floats(constraints.LANGAU_MIN_SIGMA,
                     constraints.LANGAU_MAX_SIGMA,
                     allow_nan=False,
                     allow_infinity=False),
           st.floats(constraints.LANGAU_MIN_A,
                     constraints.LANGAU_MAX_A,
                     allow_nan=False,
                     allow_infinity=False)
           )
    
    def test_langau_stability(self, mpv, eta, sigma, A):
        ''' Check Langau outputs for same input parameters '''
        # Correct input to avoid oscillations
        if sigma > 100 * eta:
            sigma = eta
        assume(sigma * eta < constraints.LANGAU_MAX_ETA_SIGMA)
        x = np.linspace(mpv - 5 * sigma * eta, mpv + 5 * sigma * eta, 1000)
        y_1 = pylandau.langau(x, mpv=mpv, eta=eta, sigma=sigma, A=A)
        y_2 = pylandau.langau(x, mpv=mpv, eta=eta, sigma=sigma, A=A)
        assert np.all(y_1 == y_2) == True

if __name__ == '__main__':
    pytest.main()
