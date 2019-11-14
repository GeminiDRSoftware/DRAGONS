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

from astrodata import testing
from gempy.adlibrary import dataselect

from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals

test_cases = [

    # GMOS-N B600 at 0.520 um ---
    ('GMOS/GN-2016B-Q-10-7', [
        'N20170121S0347.fits',  # Bias
        'N20170121S0348.fits',  # Bias
        'N20170121S0349.fits',  # Bias
        'N20170121S0350.fits',  # Bias
        'N20170121S0351.fits',  # Bias
        'N20170121S0286.fits',  # Flat
        'N20170122S0221.fits',  # Arc
        'N20170121S0285.fits',  # Spectro-photometric Standard Feige66
    ]),

    # GMOS-S R400 at 0.700 um ---
    ('GMOS/GS-2017B-LP-14', [
        'S20171014S0223.fits',  # Bias
        'S20171014S0224.fits',  # Bias
        'S20171014S0225.fits',  # Bias
        'S20171014S0226.fits',  # Bias
        'S20171014S0227.fits',  # Bias
        'S20171015S0053.fits',  # Flat
        'S20171016S0008.fits',  # Arc
        'S20171015S0052.fits',  # Spectro-photomeetric Standard Feige110
    ])

]


@pytest.fixture(scope='session')
def path_to_outputs(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('dragons_tests_')

    def _output_dir(path):
        _dir = tmp_path / path
        _dir.mkdir(parents=True, exist_ok=True)
        os.chdir(_dir)

        return _dir

    return _output_dir


def _reduce(files, calibrations=None, recipe_name="_default"):
    calibrations = calibrations if calibrations is not None else []

    r = Reduce()
    r.files.extend(files)
    r.recipename = recipe_name
    r.ucals = calibrations
    r.runr()

    return r.output_filenames[0]


@pytest.mark.remote_data
@pytest.mark.parametrize("path, files", test_cases)
def test_extract_aperture(path, files, path_to_outputs):
    calibrations = []

    output_dir = path_to_outputs(path)
    all_files = [testing.download_from_archive(f, path) for f in files]

    bias_files = dataselect.select_data(all_files, tags=['BIAS'])
    master_bias = _reduce(bias_files)
    calibrations.append('processed_bias:{:s}'.format(master_bias))

    flat_files = dataselect.select_data(all_files, tags=['FLAT'])
    master_flat = _reduce(flat_files, calibrations=calibrations)
    calibrations.append('processed_flat:{:s}'.format(master_flat))

    print("... Current working dir: {}".format(output_dir))


if __name__ == '__main__':
    pytest.main()
