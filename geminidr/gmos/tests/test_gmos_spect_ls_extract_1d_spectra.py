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
from .test_gmos_spect_ls_trace_aperture import trace_apertures_parameters

extract_1d_spectra_parameters = {
    "method": "standard",
    "width": None,
    "grow": 10,
    "debug": False
}

test_datasets = [
    # Input Filename
    ("process_arcs/GMOS/N20180508S0021_aperturesTraced.fits", 244),  # B600 720
    ("process_arcs/GMOS/N20180509S0010_aperturesTraced.fits", 259),  # R400 900
    ("process_arcs/GMOS/N20180516S0081_aperturesTraced.fits", 255),  # R600 860
    ("process_arcs/GMOS/N20190201S0163_aperturesTraced.fits", 255),  # B600 530
    ("process_arcs/GMOS/N20190313S0114_aperturesTraced.fits", 254),  # B600 482
    ("process_arcs/GMOS/N20190427S0123_aperturesTraced.fits", 260),  # R400 525
    ("process_arcs/GMOS/N20190427S0126_aperturesTraced.fits", 259),  # R400 625
    ("process_arcs/GMOS/N20190427S0127_aperturesTraced.fits", 258),  # R400 725
    ("process_arcs/GMOS/N20190427S0141_aperturesTraced.fits", 264),  # R150 660
]

ref_datasets = [
    "_".join(f[0].split("_")[:-1]) + "_extracted.fits"
    for f in test_datasets
]


@pytest.fixture(scope='module')
def ad(request, path_to_inputs, path_to_outputs):
    """
    Loads existing input FITS files as AstroData objects, runs the
    `extract1DSpectra` primitive on it, and return the output object containing
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
    path_to_inputs : fixture
        Custom fixture defined in `astrodata.testing` containing the path to the
        cached input files.
    path_to_outputs : fixture
        Custom fixture defined in `astrodata.testing` containing the path to the
        output folder.

    Returns
    -------
    AstroData
        Object containing Wavelength Solution table.

    Raises
    ------
    IOError
        If the input file does not exist and if --force-preprocess-data is False.
    """
    force_preprocess = request.config.getoption("--force-preprocess-data")
    fname, ap_center = request.param

    full_fname = os.path.join(path_to_inputs, request.param[0])

    if os.path.exists(full_fname):
        print("\n Loading existing input file:\n  {:s}\n".format(full_fname))
        _ad = astrodata.open(full_fname)

    elif force_preprocess:

        print("\n Pre-processing input file:\n  {:s}\n".format(full_fname))
        subpath, basename = os.path.split(full_fname)
        basename, extension = os.path.splitext(basename)
        basename = basename.split('_')[0] + extension

        raw_fname = testing.download_from_archive(basename, path=subpath)

        _ad = astrodata.open(raw_fname)
        _ad = preprocess_data(_ad, subpath, ap_center)

    else:
        raise IOError("Cannot find input file:\n {:s}".format(full_fname))

    p = primitives_gmos_spect.GMOSSpect([])
    p.viewer = geminidr.dormantViewer(p, None)

    ad_out = p.extract1DSpectra([_ad], **extract_1d_spectra_parameters)[0]

    tests_failed_before_module = request.session.testsfailed

    yield ad_out

    if request.session.testsfailed > tests_failed_before_module:
        _dir = os.path.join(path_to_outputs, os.path.dirname(fname))
        os.makedirs(_dir, exist_ok=True)

        fname_out = os.path.join(_dir, ad_out.filename)
        ad_out.write(filename=fname_out, overwrite=True)
        print('\n Saved file to:\n  {}\n'.format(fname_out))

    del ad_out


def preprocess_data(ad, path, center):
    """
    Recipe used to generate input data for Wavelength Calibration tests. It is
    called only if the input data do not exist and if `--force-preprocess-data`
    is used in the command line.

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
    _p.mosaicDetectors(suffix="_aperturesTraced")
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

    ad = _p.traceApertures([_ad], **trace_apertures_parameters)[0]
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


@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad, ad_ref", zip(test_datasets, ref_datasets), indirect=True)
def test_extract_1d_spectra_is_stable(ad, ad_ref):
    assert ad[0].data.ndim == 1
    np.testing.assert_allclose(ad[0].data, ad_ref[0].data, atol=1e-3)
