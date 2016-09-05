''' Advanced fitting example that calculated the MPV errors correctly. 
For this iminuit is used plus an additional step with minos to get
asymmetric correct errors.

Thus iminuit has to be installed (pip install iminuit).

Further info on minuit and the original implementation:
http://seal.web.cern.ch/seal/documents/minuit/mnusersguide.pdf
'''

import numpy as np
import iminuit

from matplotlib import pyplot as plt

from pylandau import langau


def fit_landau_migrad(x, y, p0, limit_mpv, limit_eta, limit_sigma, limit_A):
    def minimizeMe(mpv, eta, sigma, A):
        chi2 = np.sum(np.square(y - langau(x, mpv, eta, sigma, A).astype(float)) / np.square(yerr.astype(float)))
        return chi2 / (x.shape[0] - 5)  # devide by NDF

    # Prefit to get correct errors
    yerr = np.sqrt(y)  # Assume error from measured data
    yerr[y < 1] = 1
    m = iminuit.Minuit(minimizeMe,
                       mpv=p0[0],
                       limit_mpv=limit_mpv,
                       error_mpv=1,
                       eta=p0[1],
                       error_eta=0.1,
                       limit_eta=limit_eta,
                       sigma=p0[2],
                       error_sigma=0.1,
                       limit_sigma=limit_sigma,
                       A=p0[3],
                       error_A=1,
                       limit_A=limit_A,
                       errordef=1,
                       print_level=2)
    m.migrad()

    if not m.get_fmin().is_valid:
        raise RuntimeError('Fit did not converge')

    # Main fit with model errors
    yerr = np.sqrt(langau(x,
                          mpv=m.values['mpv'],
                          eta=m.values['eta'],
                          sigma=m.values['sigma'],
                          A=m.values['A']))  # Assume error from measured data
    yerr[y < 1] = 1

    m = iminuit.Minuit(minimizeMe,
                       mpv=m.values['mpv'],
                       limit_mpv=limit_mpv,
                       error_mpv=1,
                       eta=m.values['eta'],
                       error_eta=0.1,
                       limit_eta=limit_eta,
                       sigma=m.values['sigma'],
                       error_sigma=0.1,
                       limit_sigma=limit_sigma,
                       A=m.values['A'],
                       error_A=1,
                       limit_A=limit_A,
                       errordef=1,
                       print_level=2)
    m.migrad()

    fit_values = m.values

    values = np.array([fit_values['mpv'],
                       fit_values['eta'],
                       fit_values['sigma'],
                       fit_values['A']])

    m.hesse()

    m.minos()
    minos_errors = m.get_merrors()

    if not minos_errors['mpv'].is_valid:
        print('Warning: MPV error determination with Minos failed! You can still use Hesse errors.')

    errors = np.array([(minos_errors['mpv'].lower, minos_errors['mpv'].upper),
                       (minos_errors['eta'].lower, minos_errors['eta'].upper),
                       (minos_errors['sigma'].lower, minos_errors['sigma'].upper),
                       (minos_errors['A'].lower, minos_errors['A'].upper)])

    return values, errors, m

if __name__ == '__main__':
    # Fake counting experiment with Landgaus distribution
    x = np.arange(100).astype(np.float)
    y = langau(x,
               mpv=30.,
               eta=5.,
               sigma=4.,
               A=1000.)
    # Add poisson error
    y += np.random.normal(np.zeros_like(y), np.sqrt(y))

    # Fit the data with numerical exact error estimation
    # Taking into account correlations
    values, errors, m = fit_landau_migrad(x,
                                          y,
                                          p0=[x.shape[0] / 2., 10., 10., np.max(y)],
                                          limit_mpv=(10., 100.),
                                          limit_eta=(2., 20.),
                                          limit_sigma=(2., 20.),
                                          limit_A=(500., 1500.))

    # Plot fit result
    yerr = np.sqrt(langau(x, *values))
    plt.errorbar(x, y, yerr, fmt='.')
    plt.plot(x, langau(x, *values), '-', label='Fit:')
    plt.legend(loc=0)
    plt.show()

    # Show Chi2 as a function of the mpv parameter
    # The cut of the parabola at dy = 1 defines the error
    # https://github.com/iminuit/iminuit/blob/master/tutorial/tutorial.ipynb
    plt.clf()
    m.draw_mnprofile('mpv', subtract_min=False)
    plt.show()

    # Show contour plot, is somewhat correlation / fit confidence
    # of these two parameter. The off regions are a hint for
    # instabilities of the langau function
    plt.clf()
    m.draw_mncontour('mpv', 'sigma', nsigma=3)
    plt.show()
