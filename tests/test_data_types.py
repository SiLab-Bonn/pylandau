''' Script to check pylandau data type handling.
'''
import unittest

import numpy as np

from hypothesis import given, seed, assume
import hypothesis.extra.numpy as nps
import hypothesis.strategies as st
from scipy.integrate import quad as integrate

from tests import constrains

import pylandau


@st.defines_strategy
def scalar_dtypes():
    """Return a strategy that can return any non-flexible scalar dtype."""
    return

# @st.composite


def python_number(draw, min_val, max_val):
    return draw(st.one_of(st.floats(min_val,
                                    max_val,
                                    allow_nan=False,
                                    allow_infinity=False),
                          st.integers(min_val,
                                      max_val)))

# @st.composite


def numpy_number(draw, min_val, max_val):
    dtype = draw(st.one_of(nps.integer_dtypes(),
                           nps.unsigned_integer_dtypes(),
                           nps.floating_dtypes()))

    if 'f' in dtype.str:
        if min_val < np.finfo(dtype).min:
            min_val = np.finfo(dtype).min
        if max_val > np.finfo(dtype).max:
            max_val = np.finfo(dtype).max
        number = draw(st.floats(min_val, max_val, allow_nan=False,
                                allow_infinity=False))
    else:
        if min_val < np.iinfo(dtype).min:
            min_val = np.iinfo(dtype).min
        if max_val > np.iinfo(dtype).max:
            max_val = np.iinfo(dtype).max
        number = draw(st.integers(min_val, max_val))

    return np.array([number], dtype)[0]


@st.composite
def A_landau(draw):
    return draw(st.sampled_from([numpy_number(draw,
                                              constrains.LANDAU_MIN_A,
                                              constrains.LANDAU_MAX_A),
                                 python_number(draw,
                                               constrains.LANDAU_MIN_A,
                                               constrains.LANDAU_MAX_A)]))


@st.composite
def MPV_landau(draw):
    return draw(st.sampled_from([numpy_number(draw,
                                              constrains.LANDAU_MIN_MPV,
                                              constrains.LANDAU_MAX_MPV),
                                 python_number(draw,
                                               constrains.LANDAU_MIN_MPV,
                                               constrains.LANDAU_MAX_MPV)]))


@st.composite
def eta_landau(draw):
    return draw(st.sampled_from([numpy_number(draw,
                                              constrains.LANDAU_MIN_ETA,
                                              constrains.LANDAU_MAX_ETA),
                                 python_number(draw,
                                               constrains.LANDAU_MIN_ETA,
                                               constrains.LANDAU_MAX_ETA)]))


@st.composite
def A_langau(draw):
    return draw(st.sampled_from([numpy_number(draw,
                                              constrains.LANGAU_MIN_A,
                                              constrains.LANGAU_MAX_A),
                                 python_number(draw,
                                               constrains.LANGAU_MIN_A,
                                               constrains.LANGAU_MAX_A)]))


@st.composite
def MPV_langau(draw):
    return draw(st.sampled_from([numpy_number(draw,
                                              constrains.LANGAU_MIN_MPV,
                                              constrains.LANGAU_MAX_MPV),
                                 python_number(draw,
                                               constrains.LANGAU_MIN_MPV,
                                               constrains.LANGAU_MAX_MPV)]))


@st.composite
def eta_langau(draw):
    return draw(st.sampled_from([numpy_number(draw,
                                              constrains.LANGAU_MIN_ETA,
                                              constrains.LANGAU_MAX_ETA),
                                 python_number(draw,
                                               constrains.LANGAU_MIN_ETA,
                                               constrains.LANGAU_MAX_ETA)]))


@st.composite
def sigma_langau(draw):
    return draw(st.sampled_from([numpy_number(draw,
                                              constrains.LANGAU_MIN_SIGMA,
                                              constrains.LANGAU_MAX_SIGMA),
                                 python_number(draw,
                                               constrains.LANGAU_MIN_SIGMA,
                                               constrains.LANGAU_MAX_SIGMA)]))


@st.composite
def x(draw):
    return draw(st.sampled_from([numpy_number(draw,
                                              constrains.MIN_X,
                                              constrains.MAX_X),
                                 python_number(draw,
                                               constrains.MIN_X,
                                               constrains.MAX_X)]))


class TestDatatypes(unittest.TestCase):

    @given(x(), MPV_landau(), eta_landau())
    def test_landau_pdf_inputs(self, x, mpv, eta):
        ''' Check Landau PDF result for different input parameter dtypes'''
        self.assertTrue(isinstance(pylandau.get_landau_pdf(x, mu=mpv,
                                                           eta=eta),
                                   float))

    @given(x(), MPV_landau(), eta_landau(), sigma_langau())
    def test_langau_pdf_inputs(self, x, mpv, eta, sigma):
        ''' Check Langau PDF result for different input parameter dtypes'''
        self.assertTrue(isinstance(pylandau.get_langau_pdf(x, mu=mpv,
                                                           eta=eta,
                                                           sigma=sigma),
                                   float))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDatatypes)
    unittest.TextTestRunner(verbosity=2).run(suite)
