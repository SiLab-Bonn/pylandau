# Plot the functions using matplotlib

import numpy as np
import pylandau
import matplotlib.pyplot as plt

x = np.arange(0, 100, 0.01)

for A, eta, mu in ((1, 1, 10), (1, 2, 30), (0.5, 5, 50)):
    # Use the function that calculates y values when given array
    plt.plot(x, pylandau.landau(x, mu, eta, A), '-', label='A=%1.1f, eta=%1.1f, mu=%1.1f' % (A, eta, mu))

    # Use the function that calculates the y value given a x value, (e.g. needed for minimizers)
    y = np.array([pylandau.get_landau(x_value, mu, eta, A) for x_value in x])
    plt.plot(x, y, '--', label='A=%1.1f, eta=%1.1f, mu=%1.1f' % (A, eta, mu))
plt.legend(loc=0)
plt.show()
