''' Script to check the module.
'''
import unittest
from scipy import integrate
from pyLandau import landau


def approx_Equal(x, y, tolerance=0.001):
    return abs(x - y) <= 0.5 * tolerance * (x + y)


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):  # remove created files
        pass

    def test_check_pdf_integral(self):  # a pdf integral has to be 1
        mu, eta, sigma = 10, 1, 3
        y = integrate.quad(landau.get_landau_pdf, 0, 10000, args=(mu, eta))[0]
        self.assertTrue(approx_Equal(y, 1))
        y = integrate.quad(landau.get_gauss_pdf, 0, 10000, args=(mu, sigma))[0]
        self.assertTrue(approx_Equal(y, 1))
        y = integrate.quad(landau.get_langau_pdf, -10000, 10000, args=(mu, eta, sigma))[0]
        self.assertTrue(approx_Equal(y, 1))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner(verbosity=2).run(suite)
