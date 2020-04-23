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

# Test parameters --------------------------------------------------------------
input_datasets = [
    # (Input Filename, Aperture Center)
    # ("N20180508S0021.fits", 244),  # B600 720 - todo: won't pass
    ("N20180509S0010.fits", 259),  # R400 900
    # ("N20180516S0081.fits", 255),  # R600 860
    # ("N20190201S0163.fits", 255),  # B600 530
    # ("N20190313S0114.fits", 254),  # B600 482
    # ("N20190427S0123.fits", 260),  # R400 525
    # ("N20190427S0126.fits", 259),  # R400 625
    # ("N20190427S0127.fits", 258),  # R400 725
    # ("N20190427S0141.fits", 264),  # R150 660
]

fixed_test_parameters_for_determine_distortion = {
    "debug": False,
    "max_missed": 5,
    "max_shift": 0.09,
    "nsum": 20,
    "step": 10,
    "trace_order": 2,
}


# Tests Definitions ------------------------------------------------------------
@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("preprocessed_ad", input_datasets, indirect=True)
def test_regression_trace_apertures(preprocessed_ad, output_path, reference_ad):

    with output_path():
        p = primitives_gmos_spect.GMOSSpect([preprocessed_ad])
        p.viewer = geminidr.dormantViewer(p, None)
        p.traceApertures()
        aperture_traced_ad = p.writeOutputs().pop()

    ref_ad = reference_ad(aperture_traced_ad.filename)

    for ext, ref_ext in zip(aperture_traced_ad, ref_ad):
        input_table = ext.APERTURE
        reference_table = ref_ext.APERTURE

        assert input_table['aper_lower'][0] <= 0
        assert input_table['aper_upper'][0] >= 0

        keys = ext.APERTURE.colnames
        actual = np.array([input_table[k] for k in keys])
        desired = np.array([reference_table[k] for k in keys])
        np.testing.assert_allclose(desired, actual, atol=0.05)


# Local Fixtures and Helper Functions ------------------------------------------
@pytest.fixture(scope='function')
def preprocessed_ad(request, cache_path, new_path_to_inputs, output_path,
                    reduce_data):
    """
    Runs `traceApertures` primitive on a pre-processed data and return the
    output object containing a `.APERTURE` table.

    Parameters
    ----------
    request : fixture
        PyTest's built-in fixture with information about the test itself.
    cache_path : pytest.fixture
        Path to where the data will be temporarily cached.
    output_path : pytest.fixture
        Fixture containing a custom context manager used to enter and leave the
        output folder easily.
    new_path_to_inputs : pytest.fixture
        Path to the permanent local input files.
    reduce_data : pytest.fixture
        Recipe to reduce the data up to the step before
        `determineWavelengthSolution`.

    Returns
    -------
    AstroData
        Aperture-traced data.
    """
    basename, center = request.param
    should_preprocess = request.config.getoption("--force-preprocess-data")

    input_fname = basename.replace('.fits', '_mosaic.fits')
    input_path = os.path.join(new_path_to_inputs, input_fname)

    if os.path.exists(input_path):
        input_ad = astrodata.open(input_path)

    elif should_preprocess:
        filename = cache_path(basename)
        ad = astrodata.open(filename)
        input_ad = reduce_data(ad, center)

    else:
        raise IOError(
            'Could not find input file:\n' +
            '  {:s}\n'.format(input_path) +
            '  Run pytest with "--force-preprocess-data" to get it')

    return input_ad


@pytest.fixture(scope='module')
def reduce_data(output_path):
    """
    Recipe used to generate input data for `traceAperture` tests.

    Parameters
    ----------
    output_path : pytest.fixture
        Fixture containing a custom context manager used to enter and leave the
        output folder easily.

    Returns
    -------
    AstroData
        Pre-processed arc data.
    """
    def _reduce_data(ad, center):
        with output_path():
            p = primitives_gmos_spect.GMOSSpect([ad])
            p.prepare()
            p.addDQ(static_bpm=None)
            p.addVAR(read_noise=True)
            p.overscanCorrect()
            p.ADUToElectrons()
            p.addVAR(poisson_noise=True)
            p.mosaicDetectors()
            ad = p.makeIRAFCompatible()[0]

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
            ad.write()

        return ad
    return _reduce_data
