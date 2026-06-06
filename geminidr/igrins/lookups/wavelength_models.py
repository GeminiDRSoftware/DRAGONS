# Provides a function to create an initial 2D wavelength solution for IGRINS data.
# The data are stored in a dict, and retrieved using a function, which allows for
# easier updates since the structure of the data can be changed.

import numpy as np
from astropy.modeling import models


wavelength_solutions = {
    'H': {'coeffs': [[ 1.62719405e+03, -1.85186657e+02,  1.10465558e+01, -6.50278922e-01],
                     [ 1.10299670e+01, -1.30398607e+00,  4.44519995e-02, -2.13691072e-04],
                     [-3.57456411e-01,  3.78580897e-02,  3.12939396e-04, -1.13459310e-03],
                     [ 2.64703094e-02, -6.07315045e-03, -1.84590437e-03, -7.88264266e-04],
                     [ 2.24830871e-04, -3.20559501e-03,  3.82718956e-04, -1.09889164e-03]],
          'orders': range(98, 125),  # i.e., domain is [98, 124]
          },
    'K': {'coeffs': [[ 2.15707804e+03, -3.19003333e+02,  2.42046495e+01, -1.82517934e+00],
                     [ 1.48318568e+01, -2.12191657e+00,  8.83729661e-02, -2.96548006e-03],
                     [-4.79808155e-01,  7.00062319e-02,  3.34163888e-03,  3.28515399e-03],
                     [ 2.37935095e-03, -7.05616369e-03, -6.86993275e-03, -2.83719655e-03],
                     [ 2.70230942e-03,  7.57069167e-03,  7.32808274e-03,  3.44550519e-03]],
          'orders': range(71, 97),  # i.e., domain is [71, 96]
          },
}


def get_initial_wavelength_solution(band):
    """
    Retrieve the initial wavelength solution for the given band.

    Parameters
    ----------
    band : str
        The band for which to retrieve the wavelength solution.
        Should be either 'H' or 'K'.

    Returns
    -------
    models.Chebyshev2D
        The initial wavelength solution as a 2D Chebyshev model, a function
        of pixel position and spectral order.
    """
    if band not in wavelength_solutions:
        raise ValueError(f"Band '{band}' not recognized.")

    wvlsol = wavelength_solutions[band]
    carray, orders = np.asarray(wvlsol['coeffs']), wvlsol['orders']
    x_degree, y_degree = carray.shape[0] - 1, carray.shape[1] - 1
    coeffs = {f"c{i}_{j}": carray[i][j] for j in range(y_degree+1)
              for i in range(x_degree+1)}
    m = models.Chebyshev2D(x_degree=x_degree, y_degree=y_degree, **coeffs,
                           x_domain=(0, 2047), y_domain=(min(orders), max(orders)))
    return m
