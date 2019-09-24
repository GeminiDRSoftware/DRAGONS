# pytest suite

"""
Tests for the matching module.

This is a suite of tests to be run with pytest.

To run:
   1) py.test -v   (must in gemini_python or have it in PYTHONPATH)
"""

import numpy as np
import pytest

from astropy.modeling import models
from gempy.library import matching, astromodels
from gempy.library.transform import Transform


SEED = 0  # random number seed
RANGE = 2000  # "size" of detector
NUM_SOURCES = 20  # number of sources to put in catalog
SCATTER = 0.5  # rms scatter

@pytest.fixture
def chebyshev1d():

    coefficients = [550., 80., 3.2, 1.6, 1.6]
    coefficients_dict = {'c{}'.format(i): c for i, c in enumerate(coefficients)}

    model = models.Chebyshev1D(degree=4, domain=[0, 3200], **coefficients_dict)

    return model


def test_KDTreeFitter_can_fit_a_chebyshev1d_function(chebyshev1d):
    np.random.seed(SEED)

    x = np.random.uniform(low=0, high=3200, size=40)

    y = chebyshev1d(x)

    input_model = models.Chebyshev1D(degree=chebyshev1d.degree, domain=[0, 3200])
    input_model.c0 = chebyshev1d.c0
    input_model.c1 = chebyshev1d.c1

    # We're starting with a linear model guess, so scale of mismatch is similar
    # to the scale of the quadratic term in the real model
    kdsigma = chebyshev1d.c2

    fitter = matching.KDTreeFitter(sigma=kdsigma)
    fitted_model = fitter(input_model, x, y)

    np.testing.assert_allclose(fitted_model.parameters, chebyshev1d.parameters, atol=1)


@pytest.fixture
def make_catalog():
    np.random.seed(SEED)
    x = np.random.uniform(low=0.05*RANGE, high=0.95*RANGE, size=NUM_SOURCES)
    y = np.random.uniform(low=0.05*RANGE, high=0.95*RANGE, size=NUM_SOURCES)
    return x, y


def transform_coords(coords, model):
    """
    Transform input coordinates to output coordinates using the
    given model, and add Gaussian scatter of given rms
    """
    np.random.seed(SEED)
    xin, yin = coords
    xout, yout = model(xin, yin)
    xout += np.random.normal(loc=0.0, scale=SCATTER, size=len(xout))
    yout += np.random.normal(loc=0.0, scale=SCATTER, size=len(yout))
    return xout, yout

def test_fit_model(make_catalog):
    xshift, yshift = 5.0, 10.0
    incoords = make_catalog
    real_model = astromodels.Shift2D(xshift, yshift)
    refcoords = transform_coords(incoords, real_model)
    in_model = real_model.copy()
    for p in in_model.param_names:
        setattr(in_model, p, 0.0)
    model = matching.fit_model(in_model, incoords, refcoords, brute=True)
    for p in model.param_names:
        assert (abs(getattr(model, p) - getattr(real_model, p)) <
                2 * SCATTER)

def test_align_catalogs(make_catalog):
    tol = 0.01
    xshift, yshift = 5.0, 10.0
    incoords = make_catalog
    real_model = astromodels.Shift2D(xshift, yshift)
    transform = Transform([astromodels.Shift2D()])
    refcoords = transform_coords(incoords, real_model)
    model = matching.align_catalogs(incoords, refcoords,
                                    transform, tolerance=tol).asModel()
    for p in model.param_names:
        assert (abs(getattr(model, p) - getattr(real_model, p)) <
                max(SCATTER, tol))

def test_match_sources():
    yin, xin = np.mgrid[0:5, 0:5]
    xref = np.array([2.1, 3.1])
    yref = np.array([3.9, 0.1])
    matched = matching.match_sources((xin.ravel(), yin.ravel()),
                                     (xref, yref))
    assert matched[22] == 0
    assert matched[3] == 1
