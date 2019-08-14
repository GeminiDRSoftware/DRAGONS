#!/usr/bin/python

import pytest
import astrodata
import numpy as np

from astropy.io import fits
from scipy import ndimage

from geminidr import dormantViewer
from geminidr.core.primitives_spect import Spect

import astrofaker


def create_1d_spectrum(width, n_lines, max_weight):
    """
    Generates a 1D NDArray with the sky spectrum.

    Parameters
    ----------
    width : int
        Number of array elements.
    n_lines : int
        Number of artificial lines.
    max_weight : float
        Maximum weight (or flux, or intensity) of the lines.

    Returns
    -------
    sky_1d_spectrum : numpy.ndarray

    """
    lines = np.random.randint(low=0, high=width, size=n_lines)
    weights = max_weight * np.random.random(size=n_lines)

    spectrum = np.zeros(width)
    spectrum[lines] = weights

    return spectrum


@pytest.fixture
def fake_spectrum():
    """
    Generates a fake 2D spectrum to be used in the tests.
    """
    np.random.seed(0)

    width = 4000
    height = 2000
    snr = 0.1

    obj_max_weight = 300.
    obj_continnum = 300. + 0.01 * np.arange(width)

    sky = create_1d_spectrum(width, int(0.01 * width), 100.)
    obj = create_1d_spectrum(width, int(0.1 * width), obj_max_weight) + \
          obj_continnum

    obj_pos = np.random.randint(low=height // 2 - int(0.1 * height),
                                high=height // 2 + int(0.1 * height))

    spec = np.repeat(sky[np.newaxis, :], height, axis=0)
    spec[obj_pos] += obj
    spec = ndimage.gaussian_filter(spec, sigma=(7, 3))

    spec += snr * obj_max_weight * np.random.random(spec.shape)

    ad = astrofaker.create('GMOS-N')
    ad.dispersion_axis = [1]
    ad.add_extension(data=spec, pixel_scale=0.08)

    return ad


def test_determine_distortion(fake_spectrum):

    assert isinstance(fake_spectrum, astrodata.AstroData)

    # p = Spect(fake_spectrum)
    #
    # # ToDo - Remove hard dependency on displays unless strictly necessary
    # p.viewer = dormantViewer(p, None)
    #
    # ad = p.determineDistortion()


if __name__ == '__main__':
    pytest.main()
