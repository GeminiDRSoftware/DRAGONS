#!/usr/bin/env python
"""
Tests for GMOS Spect LS Extraction.

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

import os
import pytest

import astrofaker
import astrodata
import geminidr
import numpy as np

from astropy.io import fits
from astrodata import testing
from gempy.adlibrary import dataselect
from gempy.utils import logutils
from geminidr.gmos import primitives_gmos_spect
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals

try:
    import astrofaker

    HAS_ASTROFAKER = True
except ImportError:
    HAS_ASTROFAKER = False


test_cases = [

    # GMOS-N B600 at 0.600 um ---
    ('GMOS/GN-2018A-Q-302-56', [
        'N20180304S0121.fits',  # Standard
        'N20180304S0122.fits',  # Standard
        'N20180304S0123.fits',  # Standard
        'N20180304S0124.fits',  # Standard
        'N20180304S0125.fits',  # Standard
        'N20180304S0126.fits',  # Standard
        'N20180304S0204.fits',  # Bias
        'N20180304S0205.fits',  # Bias
        'N20180304S0206.fits',  # Bias
        'N20180304S0207.fits',  # Bias
        'N20180304S0208.fits',  # Bias
        'N20180304S0122.fits',  # Flat
        'N20180304S0123.fits',  # Flat
        'N20180304S0126.fits',  # Flat
        'N20180302S0397.fits',  # Arc
    ]),

    # # GMOS-S R400 at 0.700 um ---
    # ('GMOS/GS-2017B-LP-14', [
    #     'S20171014S0223.fits',  # Bias
    #     'S20171014S0224.fits',  # Bias
    #     'S20171014S0225.fits',  # Bias
    #     'S20171014S0226.fits',  # Bias
    #     'S20171014S0227.fits',  # Bias
    #     'S20171015S0053.fits',  # Flat
    #     'S20171016S0008.fits',  # Arc
    #     'S20171015S0052.fits',  # Spectro-photometric Standard Feige110
    # ])

]


@pytest.mark.remote_data
@pytest.mark.parametrize("path, files", test_cases)
def test_reduce(path, files, tmp_path_factory):

    tmp_path = tmp_path_factory.mktemp('dragons_tests_')
    tmp_path = tmp_path / path
    tmp_path.mkdir(parents=True, exist_ok=True)
    os.chdir(tmp_path)

    logutils.config(mode="standard", file_name=tmp_path / "my_log.log")

    files = [testing.download_from_archive(f, path=path) for f in files]

    bias_list = dataselect.select_data(files, tags=['BIAS'])
    r = Reduce()
    r.files.extend(bias_list)
    r.runr()
    master_bias = r.output_filenames[0]

    cals = ["processed_bias:{:s}".format(master_bias)]

    flat_list = dataselect.select_data(files, tags=['FLAT'])
    r = Reduce()
    r.files.extend(flat_list)
    r.ucals = normalize_ucals(r.files, cals)
    r.runr()
    master_flat = r.output_filenames[0]

    cals.append("processed_flat:{:s}".format(master_flat))

    arc_list = dataselect.select_data(files, tags=['ARC'])
    r = Reduce()
    r.files.extend(arc_list)
    r.ucals = normalize_ucals(r.files, cals)
    r.runr()
    master_arc = r.output_filenames[0]

    cals.append("processed_arc:{:s}".format(master_arc))

    sci_list = dataselect.select_data(files, xtags=['CAL'])

    # Reduce the standard -----------
    _p = primitives_gmos_spect.GMOSSpect([astrodata.open(f) for f in sci_list])

    _p.prepare()
    _p.addDQ(static_bpm=None)
    _p.addVAR(read_noise=True)
    _p.overscanCorrect()
    _p.biasCorrect(bias=master_bias)
    _p.ADUToElectrons()
    _p.addVAR(poisson_noise=True)
    _p.flatCorrect(flat=master_flat)
    _p.distortionCorrect(arc=master_arc)
    _p.writeOutputs()
    _p.findSourceApertures(max_apertures=1)
    _p.writeOutputs()
    # _p.skyCorrectFromSlit()
    # _p.traceApertures()
    # _p.extract1DSpectra()
    # _p.linearizeSpectra()
    # _p.calculateSensitivity()
    # _p.writeOutputs()


@pytest.mark.skipif("not HAS_ASTROFAKER")
def test_find_apertures():

    hdu = fits.ImageHDU()
    hdu.data = np.zeros((200, 100))
    hdu.data[100] = 10.

    ad = astrofaker.create('GMOS-S')
    ad.add_extension(hdu, pixel_scale=1.0)

    _p = primitives_gmos_spect.GMOSSpect([ad])
    _p.findSourceApertures()



@pytest.fixture(scope='session')
def path_to_outputs(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('dragons_tests_')

    def _output_dir(path):
        _dir = tmp_path / path
        _dir.mkdir(parents=True, exist_ok=True)
        os.chdir(_dir)

        return _dir

    return _output_dir


if __name__ == '__main__':
    pytest.main()
