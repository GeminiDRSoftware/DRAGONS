#!/usr/bin/env python
"""
Regression tests for GMOS LS extraction1D. These tests run on real data to ensure
that the output is always the same. Further investigation is needed to check if
these outputs are scientifically relevant.
"""

import os

import pytest
from astropy import table

import astrodata
import geminidr
import numpy as np
from astrodata import testing
from geminidr.gmos import primitives_gmos_spect
from gempy.utils import logutils

# Test parameters --------------------------------------------------------------
trace_apertures_parameters = {
    "trace_order": 2,
    "nsum": 20,
    "step": 10,
    "max_shift": 0.09,
    "max_missed": 5,
    "debug": False
}

input_datasets = [
    # Input Filename                                                  Aperture Center
    # ("process_arcs/GMOS/N20180508S0021_mosaicWithApertureTable.fits", 244),  # B600 720 - todo: won't pass
    ("process_arcs/GMOS/N20180509S0010_mosaicWithApertureTable.fits", 259),  # R400 900
    ("process_arcs/GMOS/N20180516S0081_mosaicWithApertureTable.fits", 255),  # R600 860
    ("process_arcs/GMOS/N20190201S0163_mosaicWithApertureTable.fits", 255),  # B600 530
    ("process_arcs/GMOS/N20190313S0114_mosaicWithApertureTable.fits", 254),  # B600 482
    ("process_arcs/GMOS/N20190427S0123_mosaicWithApertureTable.fits", 260),  # R400 525
    ("process_arcs/GMOS/N20190427S0126_mosaicWithApertureTable.fits", 259),  # R400 625
    ("process_arcs/GMOS/N20190427S0127_mosaicWithApertureTable.fits", 258),  # R400 725
    ("process_arcs/GMOS/N20190427S0141_mosaicWithApertureTable.fits", 264),  # R150 660
]

ref_datasets = [
    "_".join(f[0].split("_")[:-1]) + "_aperturesTraced.fits"
    for f in input_datasets
]

# Local Fixtures and Helper Functions ------------------------------------------
@pytest.fixture(scope='module')
def ad(request, ad_factory, path_to_outputs):
    """
    Loads existing input FITS files as AstroData objects, runs the
    `traceApertures` primitive on it, and return the output object containing
    a `.APERTURE` table.

    This makes tests more efficient because the primitive is run only once,
    instead of N x Number of tests.

    If the input file does not exist, this fixture raises a IOError.

    If the input file does not exist and PyTest is called with the
    `--force-preprocess-data`, this fixture looks for cached raw data and
    process it. If the raw data does not exist, it is then cached via download
    from the Gemini Archive.

    Parameters
    ----------
    request : fixture
        PyTest's built-in fixture with information about the test itself.
    ad_factory : fixture
        Custom fixture defined in the `conftest.py` file that loads cached data,
        or download and/or process it if needed.
    path_to_outputs : fixture
        Custom fixture defined in `astrodata.testing` containing the path to the
        output folder.

    Returns
    -------
    AstroData
        Object containing Wavelength Solution table.
    """
    fname, ap_center = request.param

    p = primitives_gmos_spect.GMOSSpect([])
    p.viewer = geminidr.dormantViewer(p, None)

    print('\n\n Running test inside folder:\n  {}'.format(path_to_outputs))

    _ad = ad_factory(fname, preprocess_recipe, **{'center': ap_center})
    ad_out = p.traceApertures([_ad], **trace_apertures_parameters)[0]

    tests_failed_before_module = request.session.testsfailed
    yield ad_out

    if request.session.testsfailed > tests_failed_before_module:
        _dir = os.path.join(path_to_outputs, os.path.dirname(fname))
        os.makedirs(_dir, exist_ok=True)

        fname_out = os.path.join(_dir, ad_out.filename)
        ad_out.write(filename=fname_out, overwrite=True)
        print('\n Saved file to:\n  {}\n'.format(fname_out))

    del ad_out


def preprocess_recipe(ad, path, center):
    """
    Recipe used to generate input data for `traceAperture` tests.

    Parameters
    ----------
    ad : AstroData
        Input raw arc data loaded as AstroData.
    path : str
        Path that points to where the input data is cached.
    center : int
        Aperture center.

    Returns
    -------
    AstroData
        Pre-processed arc data.
    """
    _p = primitives_gmos_spect.GMOSSpect([ad])

    _p.prepare()
    _p.addDQ(static_bpm=None)
    _p.addVAR(read_noise=True)
    _p.overscanCorrect()
    _p.ADUToElectrons()
    _p.addVAR(poisson_noise=True)
    _p.mosaicDetectors(suffix="_mosaicWithApertureTable")
    ad = _p.makeIRAFCompatible()[0]

    width = ad[0].shape[1]

    aperture = table.Table(
        [[1],  # Number
         [1],  # ndim
         [0],  # degree
         [0],  # domain_start
         [width - 1],  # domain_end
         [center],  # c0
         [-10],  # aper_lower
         [10],  # aper_upper
         ],
        names=[
            'number',
            'ndim',
            'degree',
            'domain_start',
            'domain_end',
            'c0',
            'aper_lower',
            'aper_upper']
    )

    ad[0].APERTURE = aperture
    ad.write(os.path.join(path, ad.filename))

    return ad


@pytest.fixture(scope="session", autouse=True)
def setup_log(path_to_outputs):
    """
    Fixture that setups DRAGONS' logging system to avoid duplicated outputs.

    Parameters
    ----------
    path_to_outputs : fixture
        Custom fixture defined in `astrodata.testing` containing the path to the
        output folder.
    """
    log_file = "{}.log".format(os.path.splitext(os.path.basename(__file__))[0])
    log_file = os.path.join(path_to_outputs, log_file)

    logutils.config(mode="standard", file_name=log_file)


# Tests Definitions ------------------------------------------------------------
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad, ad_ref", zip(input_datasets, ref_datasets), indirect=True)
def test_trace_apertures_is_stable(ad, ad_ref):

    input_table = ad[0].APERTURE
    reference_table = ad_ref[0].APERTURE

    assert input_table['aper_lower'][0] <= 0
    assert input_table['aper_upper'][0] >= 0

    keys = ad[0].APERTURE.colnames

    actual = np.array([input_table[k] for k in keys])
    desired = np.array([reference_table[k] for k in keys])

    np.testing.assert_allclose(desired, actual, atol=0.05)
