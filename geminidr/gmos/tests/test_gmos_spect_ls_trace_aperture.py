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
from astrodata import testing
from geminidr.gmos import primitives_gmos_spect
from gempy.utils import logutils

test_datasets = [
    # Input Filename                                Aperture Center
    ("N20180508S0021_mosaicWithApertureTable.fits", 244),  # B600 720
    ("N20180509S0010_mosaicWithApertureTable.fits", 259),  # R400 900
    ("N20180516S0081_mosaicWithApertureTable.fits", 255),  # R600 860
    ("N20190201S0163_mosaicWithApertureTable.fits", 255),  # B600 530
    ("N20190313S0114_mosaicWithApertureTable.fits", 254),  # B600 482
    ("N20190427S0123_mosaicWithApertureTable.fits", 260),  # R400 525
    ("N20190427S0126_mosaicWithApertureTable.fits", 259),  # R400 625
    ("N20190427S0127_mosaicWithApertureTable.fits", 258),  # R400 725
    ("N20190427S0141_mosaicWithApertureTable.fits", 264),  # R150 660
]


@pytest.fixture(scope='module')
def ad_factory(request, path_to_inputs, path_to_outputs, path_to_refs):
    """
    Returns an function that can be used to load an existing input file or to
    create it if needed.

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
    path_to_refs : fixture
        Custom fixture defined in `astrodata.testing` containing the path to the
        cached reference files.

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

    p = primitives_gmos_spect.GMOSSpect([])
    p.viewer = geminidr.dormantViewer(p, None)

    def _astrodata_factory(filename, center):

        fname = os.path.join(path_to_inputs, filename)

        print('\n\n Running test inside folder:\n  {}'.format(path_to_outputs))

        if os.path.exists(fname):
            print("\n Loading existing input file:\n  {:s}\n".format(fname))
            _ad = astrodata.open(fname)

        elif force_preprocess:

            print("\n Pre-processing input file:\n  {:s}\n".format(fname))
            subpath, basename = os.path.split(fname)
            basename, extension = os.path.splitext(basename)
            basename = basename.split('_')[0] + extension

            raw_fname = testing.download_from_archive(basename, path=subpath)

            _ad = astrodata.open(raw_fname)
            _ad = preprocess_data(_ad, path_to_inputs, center)

        else:
            raise IOError("Cannot find input file:\n {:s}".format(fname))

        return _ad

    return _astrodata_factory


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


@pytest.mark.parametrize("input_fname, ap_center", test_datasets)
def test_can_run_trace_standard_star_aperture(ad_factory, input_fname, ap_center):
    ad = ad_factory(input_fname, ap_center)
    p = primitives_gmos_spect.GMOSSpect([])

    p.traceApertures(
        [ad],
        trace_order=2,
        nsum=20,
        step=10,
        max_shift=0.09,
        max_missed=5,
        debug=False,
    )
