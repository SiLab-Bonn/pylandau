# Plot Landau distributions with FWHM and MPV
import numpy as np
import matplotlib.pyplot as plt
import pylandau
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
for A, eta, mpv in ((1, 1, 10), (1, 2, 30), (0.5, 5, 50)):
    y = pylandau.landau(x, mpv, eta, A)
    plt.plot(x, y, label='A=%d, mpv=%d, eta=%d' % (A, mpv, eta))
    x_fwhm_1, x_fwhm_2 = fwhm(x, y)
    plt.plot([x_fwhm_1, x_fwhm_2], [np.max(y) / 2., np.max(y) / 2.], label='FWHM: %1.1f' % np.abs(x_fwhm_1 - x_fwhm_2))
    plt.plot([mpv, mpv], [0., np.max(y)], label='MPV: %1.1f' % mpv)
plt.legend(loc=0)
plt.show()
