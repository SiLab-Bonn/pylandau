from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from pyLandau import landau

x = np.arange(0, 100, 0.5)
mu, eta, sigma, A = 10, 1, 3, 1
y = landau.langau(x, mu, eta, sigma, A) + np.random.normal(0, 0.05, 200)

coeff, pcov = curve_fit(landau.langau, x, y, p0=(mu, eta, sigma, A))
plt.plot(x, y, "o")
plt.plot(x, landau.langau(x, *coeff), "-")
plt.show()
