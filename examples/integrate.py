import pylandau
# Check the integral of PDF ~ 1 with scipy numerical integration
mu, eta, sigma, A = 10, 1, 3, 1
from scipy import integrate
y, err = integrate.quad(pylandau.get_landau_pdf, 0, 10000, args=(mu, eta))
print('Integral of Landau PDF:', y)
y, err = integrate.quad(pylandau.get_gauss_pdf, 0, 10000, args=(mu, sigma))
print('Integral of Gauss PDF:', y)
y, err = integrate.quad(pylandau.get_langau_pdf, -10000, 10000, args=(mu, eta, sigma))
print('Integral of Landau + Gauss (Langau) PDF:', y)
