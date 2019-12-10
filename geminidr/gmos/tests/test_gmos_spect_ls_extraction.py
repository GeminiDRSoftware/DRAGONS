#!/usr/bin/env python
"""
Regression tests for GMOS LS extraction1D. These tests run on real data to ensure
that the output is always the same. Further investigation is needed to check if
these outputs are scientifically relevant.
"""

import os
import pytest

import astrodata
import gemini_instruments
import geminidr
from astrodata import testing
from astropy import table
from geminidr.gmos import primitives_gmos_spect

from .test_gmos_spect_ls_wavelength_calibration import preprocess_data

test_datasets = [
    # science
    ("N20180508S0021_mosaic.fits", 244, -20, 20),  # B600 720
    # ("N20180509S0010.fits", "N20180509S0011.fits", "N20180509S0080.fits"),  # R400 900
    # ("N20180516S0081.fits", "N20180516S0082.fits", "N20180516S0214.fits"),  # R600 860
    # ("N20190201S0163.fits", "N20190201S0161.fits", "N20190201S0176.fits"),  # B600 530
    # ("N20190313S0114.fits", "N20190313S0115.fits", "N20190313S0132.fits"),  # B600 482
    # ("N20190427S0123.fits", "N20190427S0124.fits", "N20190427S0266.fits"),  # R400 525
    # ("N20190427S0126.fits", "N20190427S0125.fits", "N20190427S0267.fits"),  # R400 625
    # ("N20190427S0127.fits", "N20190427S0128.fits", "N20190427S0268.fits"),  # R400 725
    # ("N20190427S0141.fits", "N20190427S0139.fits", "N20190427S0270.fits"),  # R150 660
]


@pytest.fixture(scope='module')
def sci(request, path_to_inputs, path_to_outputs, path_to_refs):
    """
    Loads existing input FITS files as AstroData objects

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

    fname, ap_center, ap_lower, ap_upper = request.param

    fname = os.path.join(path_to_inputs, fname)

    p = primitives_gmos_spect.GMOSSpect([])
    p.viewer = geminidr.dormantViewer(p, None)

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
        _ad = preprocess_data(_ad, path_to_inputs, ap_center, ap_lower, ap_upper)





    else:
        raise IOError("Cannot find input file:\n {:s}".format(fname))

    return request.param


def preprocess_data(ad, path, center, lower, upper):
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
    ad = _p.makeIRAFCompatible()[0]

    ad[0].APERTURE = table.Table(
        [[1],  # Number
         [1],  # ndim
         [0],  # degree
         [0],  # domain_start
         [ad[0].shape[0] - 1],  # domain_end
         [center],  # c0
         [lower],  # aper_lower
         [upper],  # aper_upper
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

    _p.writeOutputs([ad], outfilename=os.path.join(path, ad.filename))

    return ad


@pytest.mark.parametrize("sci", test_datasets, indirect=True)
def test_extraction_of_std_star(sci):
    assert True
