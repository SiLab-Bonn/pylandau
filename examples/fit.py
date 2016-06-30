from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import pylandau

x = np.arange(0, 100, 0.5)
mpv, eta, sigma, A = 10, 1, 3, 1
y = pylandau.langau(x, mpv, eta, sigma, A) + np.random.normal(0, 0.05, 200)

coeff, pcov = curve_fit(pylandau.langau, x, y, p0=(mpv, eta, sigma, A))
plt.plot(x, y, "o")
plt.plot(x, pylandau.langau(x, *coeff), "-")
plt.show()
