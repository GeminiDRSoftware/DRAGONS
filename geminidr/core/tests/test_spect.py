#!/usr/bin/env python
"""
Tests applied to primitives_spect.py

Notes
-----

    For extraction tests, your input wants to be a 2D image with an `APERTURE`
    table attached. You'll see what happens if you take a spectrophotometric
    standard and run it through the standard reduction recipe, but the
    `APERTURE` table has one row per aperture with the following columns:

    - number : sequential list of aperture number

    - ndim, degree, domain_start, domain_end, c0, [c1, c2, c3...] : standard
    Chebyshev1D definition of the aperture centre (in pixels) as a function of
    pixel in the dispersion direction

    - aper_lower : location of bottom of aperture relative to centre (always
    negative)

    - aper_upper : location of top of aperture relative to centre (always
    positive)

    The ndim column will always be 1 since it's always 1D Chebyshev, but the
    `model_to_dict()` and `dict_to_model()` functions that convert the Model
    instance to a dict create/require this.
"""
import pytest
import numpy as np

from astropy import table
from astropy.io import fits
from scipy import ndimage, optimize

from geminidr.core import primitives_spect


# noinspection PyPep8Naming
def test_QESpline_optimization():
    """
    Test the optimization of the QESpline. This defines 3 regions, each of a
    different constant value, with gaps between them. The spline optimization
    should determine the relative offsets.
    """
    from geminidr.core.primitives_spect import QESpline

    gap = 20
    data_length = 300
    real_coeffs = [0.5, 1.2]

    # noinspection PyTypeChecker
    data = np.array([1] * data_length +
                    [0] * gap +
                    [real_coeffs[0]] * data_length +
                    [0] * gap +
                    [real_coeffs[1]] * data_length)

    masked_data = np.ma.masked_where(data == 0, data)
    xpix = np.arange(len(data))
    weights = np.where(data > 0, 1., 0.)
    boundaries = (data_length, 2*data_length+gap)

    coeffs = np.ones((2,))
    order = 10

    result = optimize.minimize(
        QESpline, coeffs,
        args=(xpix, masked_data, weights, boundaries, order),
        tol=1e-7,
        method='Nelder-Mead'
    )

    np.testing.assert_allclose(real_coeffs, 1./result.x, atol=0.01)


@pytest.fixture(scope="module")
def fake_data():

    astrofaker = pytest.importorskip("astrofaker")

    data = np.zeros((100, 200))
    data[50] = 10.

    hdu = fits.ImageHDU()
    hdu.header['CCDSUM'] = "1 1"
    hdu.data = data

    aperture = table.Table(
        [[1],  # Number
         [1],  # ndim
         [0],  # degree
         [0],  # domain_start
         [data.shape[1] - 1],  # domain_end
         [50],  # c0
         [-3],  # aper_lower
         [3],  # aper_upper
         ],
        names=[
            'number',
            'ndim',
            'degree',
            'domain_start',
            'domain_end',
            'c0',
            'aper_lower',
            'aper_upper'],
    )

    ad = astrofaker.create('GMOS-S')
    ad.add_extension(hdu, pixel_scale=1.0)
    ad[0].APERTURE = aperture

    return ad



def test_fake_star_has_expected_integrated_flux():

    data = np.zeros(100)
    data[50] = 10.
    data = ndimage.gaussian_filter(data, sigma=5)

    np.testing.assert_almost_equal(data.sum(), 10.)


def test_find_apertures(fake_data):
    _p = primitives_spect.Spect([])
    _p.findSourceApertures(fake_data)


def test_sky_correct_from_slit(fake_data):
    _p = primitives_spect.Spect([])
    ade = _p.skyCorrectFromSlit(fake_data)[0]


def test_extract_1d_spectra(fake_data):
    _p = primitives_spect.Spect([])
    ade = _p.extract1DSpectra(fake_data)[0]

    np.testing.assert_equal(ade[0].shape[0], fake_data[0].data.shape[1])
    np.testing.assert_equal(ade[0].data, fake_data[0].data[50])


if __name__ == '__main__':
    pytest.main()
