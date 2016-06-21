import numpy as np
from pyLandau import landau

# Plot the functions using matplotlib
import matplotlib.pyplot as plt
x = np.arange(0, 100, 0.01)

for A, eta, mu in ((1, 1, 10), (1, 2, 30), (0.5, 5, 50)):
    plt.plot(x, landau.landau(x, mu, eta, A), label='A=%d, eta=%d, mu=%d' % (A, eta, mu))
plt.legend(loc=0)
plt.show()

# Plot the Landau Gauss convolution and cross check with convolution with scipy
from scipy.ndimage.filters import gaussian_filter1d
mu, eta, sigma, A = 10, 1, 3, 1
x = np.arange(0, 50, 0.01)
y = landau.landau(x, mu=mu, eta=eta, A=A)
y_gconv = gaussian_filter1d(y, sigma=sigma / 0.01)
y_gconv_2 = landau.langau(x, mu, eta, sigma, A)
plt.plot(x, y, label='Landau')
plt.plot(x, y_gconv_2, label='Langau')
plt.plot(x, y_gconv / np.amax(y_gconv), '--', label='Langau Scipy')
plt.legend(loc=0)
plt.show()

# Plot Landau distributions with FWHM and MPV
from scipy.interpolate import splrep, sproot


def fwhm(x, y, k=10):  # http://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak
    """
    Determine full-with-half-maximum of a peaked set of points, x and y.

    Assumes that there is only one peak present in the datasset.  The function
    uses a spline interpolation of order k.
    """

    class MultiplePeaks(Exception):
        pass

    class NoPeaksFound(Exception):
        pass

    half_max = np.amax(y) / 2.0
    s = splrep(x, y - half_max)
    roots = sproot(s)

    if len(roots) > 2:
        raise MultiplePeaks("The dataset appears to have multiple peaks, and "
                            "thus the FWHM can't be determined.")
    elif len(roots) < 2:
        raise NoPeaksFound("No proper peaks were found in the data set; likely "
                           "the dataset is flat (e.g. all zeros).")
    else:
        return roots[0], roots[1]

x = np.arange(0, 100, 0.01)
for A, eta, mu in ((1, 1, 10), (1, 2, 30), (0.5, 5, 50)):
    y = landau.landau(x, mu, eta, A)
    plt.plot(x, y, label='A=%d, eta=%d, mu=%d' % (A, eta, mu))
    x_fwhm_1, x_fwhm_2 = fwhm(x, y)
    plt.plot([x_fwhm_1, x_fwhm_2], [np.max(y) / 2., np.max(y) / 2.], label='FWHM: %1.1f' % np.abs(x_fwhm_1 - x_fwhm_2))
    x_mpv = x[np.argmax(y)]
    plt.plot([x_mpv, x_mpv], [0., np.max(y)], label='MPV: %1.1f' % x_mpv)
plt.legend(loc=0)
plt.show()
