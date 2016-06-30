# Plot the functions using matplotlib

import numpy as np
import pylandau
import matplotlib.pyplot as plt

x = np.arange(0, 100, 0.01)
for A, eta, mpv in ((1, 1, 10), (1, 2, 30), (0.5, 5, 50)):
    # Use the function that calculates y values when given array
    plt.plot(x, pylandau.landau(x, mpv, eta, A), '-', label='mu=%1.1f, eta=%1.1f, A=%1.1f' % (mpv, eta, A))

    # Use the function that calculates the y value given a x value, (e.g. needed for minimizers)
    y = np.array([pylandau.get_landau(x_value, mpv, eta, A) for x_value in x])
    plt.plot(x, y, 'r--', label='mu=%1.1f, eta=%1.1f, A=%1.1f' % (mpv, eta, A))

    # Use the function that calculates the y value given a x value, (e.g. needed for minimizers)
    sigma = 10
    plt.plot(x, pylandau.langau(x, mpv, eta, sigma, A), '--', label='mu=%1.1f, eta=%1.1f, sigma=%1.1f, A=%1.1f' % (A, eta, sigma, mpv))
plt.legend(loc=0)
plt.show()
