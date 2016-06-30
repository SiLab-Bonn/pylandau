''' Script to check pyLandau.
'''
import unittest
import pylandau


def approx_Equal(x, y, tolerance=0.001):  # check similarity of values within an allowed tolerance
    return abs(x - y) <= 0.5 * tolerance * (x + y)


def drange(start, stop, step):  # range with floats
    r = start
    while r < stop:
        yield r
        r += step


def F(function, x, args=()):
    return (function(x_i, *args) for x_i in x)


def simple_integrate(function, a, b, args=(), dx=1.):  # integration with sum, substitutes scipy.integrate.quad
    x = drange(a, b, dx)
    y = F(function, x, args)
    return sum(y) * dx, x


try:  # use scipy integrate function if available, otherwise self defined simple_integrate
    from scipy.integrate import quad as integrate
except ImportError:
    integrate = simple_integrate


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):  # remove created files
        pass

    def test_check_pdf_integral(self):  # a pdf integral has to be 1
        mu, eta, sigma = 10, 1, 3
        result, _ = integrate(pylandau.get_landau_pdf, 0, 10000, args=(mu, eta))
        self.assertTrue(approx_Equal(result, 1))
        result, _ = integrate(pylandau.get_gauss_pdf, 0, 10000, args=(mu, sigma))
        self.assertTrue(approx_Equal(result, 1))
        result, _ = integrate(pylandau.get_langau_pdf, -10000, 10000, args=(mu, eta, sigma))
        self.assertTrue(approx_Equal(result, 1))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner(verbosity=2).run(suite)
