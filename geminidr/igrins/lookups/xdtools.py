# Functions with a common API that can be imported and used by
# primitives in the CrossDispersed class

import numpy as np
from astropy.modeling import models

from gempy.library import astromodels as am


# Chebyshev2D coefficients for the product of wavelength (in nm) and spectral order
# These coefficients come from the test dataset I used in development
wavelength_solutions_from_data = {
    'K': {'coeffs': [[ 1.78095299e+05,  4.25074583e+02,  1.61294404e+01, -3.92109393e-01],
                     [ 1.21937439e+03,  1.07871451e+01, -5.04098648e+00,  2.29315668e-01],
                     [-3.98015723e+01, -1.28642216e+00,  1.31774478e-02, -3.05626312e-01],
                     [ 5.39157407e-01, -4.58463186e-01, -6.90676137e-02, -2.49087382e-01],
                     [-7.82344343e-02, -3.61791401e-01, -5.13339735e-02, -1.32550482e-01]],
          'orders': (70, 93),
          },
}

# These coefficients come from the 20240721 echellogram files in IGRINSDR/ref_data
wavelength_solutions = {
    'H': {'coeffs': [[ 1.79415296e+05,  6.69726798e+02,  1.95996272e+01, -9.17334128e-02],
                     [ 1.21585058e+03, -1.05999824e+00, -3.54267563e+00,  2.74611357e-01],
                     [-3.94307645e+01, -4.28312043e-01,  2.75832867e-01, -8.97743149e-02],
                     [ 2.89929824e+00, -3.38279105e-01, -2.47831431e-01, -9.06292807e-02],
                     [ 4.91360620e-03, -3.41473279e-01,  1.68211813e-02, -9.82160237e-02]],
          'orders': (98, 124),
          },
    'K': {'coeffs': [[ 1.78123452e+05,  4.78356949e+02,  1.95082469e+01, -2.01979603e-01],
                     [ 1.22520002e+03,  8.77112566e+00, -5.89552393e+00,  3.06210566e-01],
                     [-3.96286136e+01, -1.31881496e-01,  7.30621562e-01,  2.93537102e-01],
                     [ 1.56450282e-01, -6.01793129e-01, -6.29879555e-01, -2.78410642e-01],
                     [ 2.70682043e-01,  7.11013537e-01,  6.73953571e-01,  3.31760748e-01]],
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
        cheb1d = am.reduce_dimensionality(m, y=spec_order)
        cheb1d.inverse = am.make_inverse_chebyshev1d(cheb1d, max_deviation=0.01)
        yield cheb1d | models.Scale(1. / spec_order)
