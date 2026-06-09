# Functions with a common API that can be imported and used by
# primitives in the CrossDispersed class

import numpy as np
from astropy.modeling import models

from gempy.library import astromodels as am


wavelength_solutions = {
    'H': {'coeffs': [[ 1.62719405e+03, -1.85186657e+02,  1.10465558e+01, -6.50278922e-01],
                     [ 1.10299670e+01, -1.30398607e+00,  4.44519995e-02, -2.13691072e-04],
                     [-3.57456411e-01,  3.78580897e-02,  3.12939396e-04, -1.13459310e-03],
                     [ 2.64703094e-02, -6.07315045e-03, -1.84590437e-03, -7.88264266e-04],
                     [ 2.24830871e-04, -3.20559501e-03,  3.82718956e-04, -1.09889164e-03]],
          'orders': (98, 124),  # min and max (inclusive); used as the domain
          },
    'K': {'coeffs': [[ 2.15707804e+03, -3.19003333e+02,  2.42046495e+01, -1.82517934e+00],
                     [ 1.48318568e+01, -2.12191657e+00,  8.83729661e-02, -2.96548006e-03],
                     [-4.79808155e-01,  7.00062319e-02,  3.34163888e-03,  3.28515399e-03],
                     [ 2.37935095e-03, -7.05616369e-03, -6.86993275e-03, -2.83719655e-03],
                     [ 2.70230942e-03,  7.57069167e-03,  7.32808274e-03,  3.44550519e-03]],
          'orders': (71, 96),
          },
}


def initial_wave_models(ad):
    """
    Yields initial wavelength solution models for each extension in the input

    Parameters
    ----------
    ad: AstroData
        the AstroData object under consideration

    Yields
    ------
        a Chebyshev1D model describing the wavelength solution for each
        extension in the input AstroData
    """
    try:
        wvlsol = wavelength_solutions[ad.band()]
    except KeyError:
        raise ValueError(f"Band '{ad.band()}' not recognized.")

    carray, orders = np.asarray(wvlsol['coeffs']), wvlsol['orders']
    x_degree, y_degree = carray.shape[0] - 1, carray.shape[1] - 1
    coeffs = {f"c{i}_{j}": carray[i][j] for j in range(y_degree+1)
              for i in range(x_degree+1)}
    m = models.Chebyshev2D(x_degree=x_degree, y_degree=y_degree, **coeffs,
                           x_domain=(0, 2047), y_domain=orders)
    for ext in ad:
        spec_order = ext.SLITEDGE["specorder"][0]
        yield am.reduce_dimensionality(m, y=spec_order)
