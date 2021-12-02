''' Script to check pylandau data type handling.
'''
import unittest

import numpy as np
import pylandau

from hypothesis import given
import hypothesis.extra.numpy as nps
import hypothesis.strategies as st

from tests import constraints


def python_number(draw, min_val, max_val):
    return draw(st.one_of(st.floats(min_val,
                                    max_val,
                                    allow_nan=False,
                                    allow_infinity=False),
                          st.integers(np.ceil(min_val),
                                      np.floor(max_val))))


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
        min_val, max_val = np.ceil(min_val), np.floor(max_val)
        if min_val < np.iinfo(dtype).min:
            min_val = np.iinfo(dtype).min
        if max_val > np.iinfo(dtype).max:
            max_val = np.iinfo(dtype).max
        number = draw(st.integers(min_val, max_val))

    return np.array([number], dtype)[0]


@st.composite
def A_landau(draw):
    return draw(st.sampled_from([numpy_number(draw,
                                              constraints.LANDAU_MIN_A,
                                              constraints.LANDAU_MAX_A),
                                 python_number(draw,
                                               constraints.LANDAU_MIN_A,
                                               constraints.LANDAU_MAX_A)]))


@st.composite
def MPV_landau(draw):
    return draw(st.sampled_from([numpy_number(draw,
                                              constraints.LANDAU_MIN_MPV,
                                              constraints.LANDAU_MAX_MPV),
                                 python_number(draw,
                                               constraints.LANDAU_MIN_MPV,
                                               constraints.LANDAU_MAX_MPV)]))


@st.composite
def eta_landau(draw):
    return draw(st.sampled_from([numpy_number(draw,
                                              constraints.LANDAU_MIN_ETA,
                                              constraints.LANDAU_MAX_ETA),
                                 python_number(draw,
                                               constraints.LANDAU_MIN_ETA,
                                               constraints.LANDAU_MAX_ETA)]))


@st.composite
def A_langau(draw):
    return draw(st.sampled_from([numpy_number(draw,
                                              constraints.LANGAU_MIN_A,
                                              constraints.LANGAU_MAX_A),
                                 python_number(draw,
                                               constraints.LANGAU_MIN_A,
                                               constraints.LANGAU_MAX_A)]))


@st.composite
def MPV_langau(draw):
    return draw(st.sampled_from([numpy_number(draw,
                                              constraints.LANGAU_MIN_MPV,
                                              constraints.LANGAU_MAX_MPV),
                                 python_number(draw,
                                               constraints.LANGAU_MIN_MPV,
                                               constraints.LANGAU_MAX_MPV)]))


@st.composite
def eta_langau(draw):
    return draw(st.sampled_from([numpy_number(draw,
                                              constraints.LANGAU_MIN_ETA,
                                              constraints.LANGAU_MAX_ETA),
                                 python_number(draw,
                                               constraints.LANGAU_MIN_ETA,
                                               constraints.LANGAU_MAX_ETA)]))


@st.composite
def sigma_langau(draw):
    return draw(st.sampled_from([numpy_number(draw,
                                              constraints.LANGAU_MIN_SIGMA,
                                              constraints.LANGAU_MAX_SIGMA),
                                 python_number(draw,
                                               constraints.LANGAU_MIN_SIGMA,
                                               constraints.LANGAU_MAX_SIGMA)]))


@st.composite
def x(draw):
    return draw(st.sampled_from([numpy_number(draw,
                                              constraints.MIN_X,
                                              constraints.MAX_X),
                                 python_number(draw,
                                               constraints.MIN_X,
                                               constraints.MAX_X)]))


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
