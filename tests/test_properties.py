''' Script to check pylandau properties.
'''
import unittest

from scipy.integrate import quad as integrate

import pylandau


class Test(unittest.TestCase):

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


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner(verbosity=2).run(suite)
