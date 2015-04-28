import numpy as np
import matplotlib.pyplot as plt
from pyLandau import landau

x = np.arange(0, 100, 0.01)

for A, eta, mu in ((1, 1, 10), (1, 2, 30), (0.5, 5, 50)):
    plt.plot(x, landau.landau(x, A, eta, mu), label='A=%d, eta=%d, mu=%d' % (A, eta, mu))

plt.legend(loc=0)
plt.show()
