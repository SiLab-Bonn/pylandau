''' Simple fit without fit error estimation.
For correct fit errors check the advanced example.
Bound should be defined. Eta has to be > 1.
'''

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import pylandau

# Create fake data with possion error
mpv, eta, sigma, A = 30, 5, 4, 1000
x = np.arange(0, 100, 0.5)
y = pylandau.langau(x, mpv, eta, sigma, A)
yerr = np.random.normal(np.zeros_like(x), np.sqrt(y))
yerr[y < 1] = 1
y += yerr

# Fit with constrains
coeff, pcov = curve_fit(pylandau.langau, x, y,
                        sigma=yerr,
                        absolute_sigma=True,
                        p0=(mpv, eta, sigma, A),
                        bounds=(1, 10000))

# Plot
plt.errorbar(x, y, np.sqrt(pylandau.langau(x, *coeff)), fmt=".")
plt.plot(x, pylandau.langau(x, *coeff), "-")
plt.show()
