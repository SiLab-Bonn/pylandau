import numpy as np
from pyLandau import landau

# Plot the functions using matplotlib
import matplotlib.pyplot as plt
x = np.arange(0, 100, 0.01)

for A, eta, mu in ((1, 1, 10), (1, 2, 30), (0.5, 5, 50)):
    plt.plot(x, landau.landau(x, mu, eta, A), label='A=%d, eta=%d, mu=%d' % (A, eta, mu))
plt.legend(loc=0)
plt.show()

# Check the integral of PDF ~ 1 with scipy numerical integration
mu, eta, sigma, A = 10, 1, 3, 1
from scipy import integrate
y, err = integrate.quad(landau.get_landau_pdf, 0, 10000, args=(mu, eta))
print 'Integral of Landau PDF:', y
y, err = integrate.quad(landau.get_gauss_pdf, 0, 10000, args=(mu, sigma))
print 'Integral of Gauss PDF:', y
y, err = integrate.quad(landau.get_langau_pdf, -10000, 10000, args=(mu, eta, sigma))
print 'Integral of Landau + Gauss (Langau) PDF:', y

# Plot the Landau Gauss convolution and cross check with convolution with scipy
from scipy.ndimage.filters import gaussian_filter1d
x = np.arange(0, 50, 0.01)
y = landau.landau(x, mu=mu, eta=eta, A=A)
y_gconv = gaussian_filter1d(y, sigma=sigma / 0.01)
y_gconv_2 = landau.langau(x, mu, eta, sigma, A)
plt.plot(x, y, label='Landau')
plt.plot(x, y_gconv_2, label='Langau')
plt.plot(x, y_gconv / np.amax(y_gconv), '--', label='Langau Scipy')
plt.legend(loc=0)
plt.show()