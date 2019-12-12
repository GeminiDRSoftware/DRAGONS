#!/usr/bin/env python
"""
Regression tests for GMOS LS `skyCorrectFromSlit`. These tests run on real data
to ensure that the output is always the same. Further investigation is needed to
check if these outputs are scientifically relevant.
"""

import os

import numpy as np
import pytest
from astropy import table

import geminidr
from geminidr.gmos import primitives_gmos_spect
from gempy.utils import logutils


# Test parameters --------------------------------------------------------------
test_datasets = [
    # Input Filename
    ("process_arcs/GMOS/N20180508S0021_aperturesTraced.fits",
     "process_arcs/GMOS/N20180615S0409_distortionDetermined.fits", 244),  # B600 720
    ("process_arcs/GMOS/N20180509S0010_aperturesTraced.fits",
     "process_arcs/GMOS/N20180509S0080_distortionDetermined.fits", 259),  # R400 900
    ("process_arcs/GMOS/N20180516S0081_aperturesTraced.fits",
     "process_arcs/GMOS/N20180516S0214_distortionDetermined.fits", 255),  # R600 860
    # ("process_arcs/GMOS/N20190201S0163_aperturesTraced.fits",
    #  "process_arcs/GMOS/N20190201S0176_distortionDetermined.fits", 255),  # B600 530
    ("process_arcs/GMOS/N20190313S0114_aperturesTraced.fits",
     "process_arcs/GMOS/N20190313S0132_distortionDetermined.fits", 254),  # B600 482
    ("process_arcs/GMOS/N20190427S0123_aperturesTraced.fits",
     "process_arcs/GMOS/N20190427S0266_distortionDetermined.fits", 260),  # R400 525
    ("process_arcs/GMOS/N20190427S0126_aperturesTraced.fits",
     "process_arcs/GMOS/N20190427S0267_distortionDetermined.fits", 259),  # R400 625
    ("process_arcs/GMOS/N20190427S0127_aperturesTraced.fits",
     "process_arcs/GMOS/N20190427S0268_distortionDetermined.fits", 258),  # R400 725
    ("process_arcs/GMOS/N20190427S0141_aperturesTraced.fits",
     "process_arcs/GMOS/N20190427S0270_distortionDetermined.fits", 264),  # R150 660
]

ref_datasets = [
    "_".join(f[0].split("_")[:-1]) + "_skyCorrected.fits"
    for f in test_datasets
]


# Local Fixtures and Helper Functions ------------------------------------------
@pytest.fixture(scope='module')
def ad(request, ad_factory, path_to_outputs):
    """
    Loads existing input FITS files as AstroData objects, runs the
    `skyCorrectFromSlit` primitive on it, and return the output object containing
    the extracted 1d spectrum. This makes tests more efficient because the
    primitive is run only once, instead of N x Numbes of tests.

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
    fname, arc_name, ap_center = request.param

    p = primitives_gmos_spect.GMOSSpect([])
    p.viewer = geminidr.dormantViewer(p, None)

    print('\n\n Running test inside folder:\n  {}'.format(path_to_outputs))

    arc_ad = ad_factory(arc_name, preprocess_arc_recipe)

    _ad = ad_factory(fname, preprocess_recipe, center=ap_center, arc=arc_ad)
    ad_out = p.skyCorrectFromSlit([_ad], order=5, grow=0)[0]

    tests_failed_before_module = request.session.testsfailed

    yield ad_out

    if request.session.testsfailed > tests_failed_before_module:
        _dir = os.path.join(path_to_outputs, os.path.dirname(fname))
        os.makedirs(_dir, exist_ok=True)

        fname_out = os.path.join(_dir, ad_out.filename)
        ad_out.write(filename=fname_out, overwrite=True)
        print('\n Saved file to:\n  {}\n'.format(fname_out))

    del ad_out


def preprocess_arc_recipe(ad, path):
    """
    Recipe used to generate input data for `distortionCorrect` tests.

    Parameters
    ----------
    ad : AstroData
        Input raw arc data loaded as AstroData.
    path : str
        Path that points to where the input data is cached.

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
    _p.mosaicDetectors()
    _p.makeIRAFCompatible()

    ad = _p.determineDistortion(
        spatial_order=3, spectral_order=4, id_only=False, min_snr=5.,
        fwidth=None, nsum=10, max_shift=0.05, max_missed=5)[0]

    _p.writeOutputs(outfilename=os.path.join(path, ad.filename))

    return ad


def preprocess_recipe(ad, path, center, arc):
    """
    Recipe used to generate input data for `skyCorrectFromSlit` tests.

    Parameters
    ----------
    ad : AstroData
        Input raw data loaded as AstroData.
    path : str
        Path that points to where the input data is cached.
    center : int
        Aperture center.
    arc : AstroData
        Distortion corrected arc loaded as AstroData.

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
    _p.mosaicDetectors()
    _p.distortionCorrect(arc=arc)
    _ad = _p.makeIRAFCompatible()[0]

    width = _ad[0].shape[1]

    aperture = table.Table(
        [[1],  # Number
         [1],  # ndim
         [0],  # degree
         [0],  # domain_start
         [width - 1],  # domain_end
         [center],  # c0
         [-5],  # aper_lower
         [5],  # aper_upper
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

    _ad[0].APERTURE = aperture

    ad = _p.traceApertures(
        [_ad], trace_order=2, nsum=20, step=10, max_shift=0.09, max_missed=5)[0]

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
@pytest.mark.parametrize("ad, ad_ref", zip(test_datasets, ref_datasets), indirect=True)
def test_extract_1d_spectra_is_stable(ad, ad_ref):
    np.testing.assert_allclose(ad[0].data, ad_ref[0].data)
