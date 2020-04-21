#!/usr/bin/env python
"""
Regression tests for GMOS LS `extract1DSpectra`. These tests run on real data to
ensure that the output is always the same. Further investigation is needed to
check if these outputs are scientifically relevant.
"""
import numpy as np
import os
import pytest

import astrodata
import gemini_instruments
import geminidr
from astrodata import testing
from astropy.table import Table
from geminidr.gmos import primitives_gmos_spect

# Test parameters --------------------------------------------------------------
# Each test input filename contains the original input filename with
# "_skyCorrected" suffix.
test_datasets = [
    ("N20180508S0021.fits", 244),  # B600 720
    #  "process_arcs/GMOS/N20180615S0409_distortionDetermined.fits", 244),
    # ("process_arcs/GMOS/N20180509S0010_skyCorrected.fits",
    #  "process_arcs/GMOS/N20180509S0080_distortionDetermined.fits", 259),  # R400 900
    # ("process_arcs/GMOS/N20180516S0081_skyCorrected.fits",
    #  "process_arcs/GMOS/N20180516S0214_distortionDetermined.fits", 255),  # R600 860
    # # ("process_arcs/GMOS/N20190201S0163_skyCorrected.fits",
    # #  "process_arcs/GMOS/N20190201S0176_distortionDetermined.fits", 255),  # B600 530
    # ("process_arcs/GMOS/N20190313S0114_skyCorrected.fits",
    #  "process_arcs/GMOS/N20190313S0132_distortionDetermined.fits", 254),  # B600 482
    # ("process_arcs/GMOS/N20190427S0123_skyCorrected.fits",
    #  "process_arcs/GMOS/N20190427S0266_distortionDetermined.fits", 260),  # R400 525
    # ("process_arcs/GMOS/N20190427S0126_skyCorrected.fits",
    #  "process_arcs/GMOS/N20190427S0267_distortionDetermined.fits", 259),  # R400 625
    # ("process_arcs/GMOS/N20190427S0127_skyCorrected.fits",
    #  "process_arcs/GMOS/N20190427S0268_distortionDetermined.fits", 258),  # R400 725
    # ("process_arcs/GMOS/N20190427S0141_skyCorrected.fits",
    #  "process_arcs/GMOS/N20190427S0270_distortionDetermined.fits", 264),  # R150 660
]


# Tests Definitions ------------------------------------------------------------
@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize('preprocessed_ad', test_datasets, indirect=True)
def test_regression_on_extract_1d_spectra(preprocessed_ad, reference_ad, output_path):

    with output_path():
        p = primitives_gmos_spect.GMOSSpect([preprocessed_ad])
        p.viewer = geminidr.dormantViewer(p, None)
        p.extract1DSpectra(method="standard", width=None, grow=10)
        extracted_ad = p.writeOutputs().pop()

    ref_ad = reference_ad(extracted_ad.filename)

    for ext, ref_ext in zip(extracted_ad, ref_ad):
        assert ext.data.ndim == 1
        np.testing.assert_allclose(ext.data, ref_ext.data, atol=1e-3)


# Local Fixtures and Helper Functions ------------------------------------------
def add_aperture_table(ad, center):
    """
    Adds a fake aperture table to the `AstroData` object.

    Parameters
    ----------
    ad : AstroData
    center : int

    Returns
    -------
    AstroData : the input data with an `.APERTURE` table attached to it.
    """
    width = ad[0].shape[1]

    aperture = Table(
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

    ad[0].APERTURE = aperture
    return ad


@pytest.fixture(scope='function')
def preprocessed_ad(request, cache_path, new_path_to_inputs, reduce_arc,
                    reduce_data):
    """
    Reads the input data or cache/process it in a temporary folder.

    Parameters
    ----------
    request : pytest.fixture
        Fixture that contains information this fixture's parent.
    cache_path : pytest.fixture
        Path to where the data will be temporarily cached.
    new_path_to_inputs : pytest.fixture
        Path to the permanent local input files.
    reduce_arc : pytest.fixture
        Recipe used to reduce ARC data.
    reduce_data : pytest.fixture
        Recipe to reduce the data up to the step before
        `determineWavelengthSolution`.

    Returns
    -------
    AstroData
        The extracted spectrum.

    Raises
    ------
    IOError : if the input file does not exist.
    """
    basename, center = request.param
    should_preprocess = request.config.getoption("--force-preprocess-data")

    input_fname = basename.replace('.fits', '_skyCorrected.fits')
    input_path = os.path.join(new_path_to_inputs, input_fname)

    if os.path.exists(input_path):
        input_data = astrodata.open(input_path)

    elif should_preprocess:
        filename = cache_path(basename)
        ad = astrodata.open(filename)
        cals = testing.get_associated_calibrations(basename)
        cals = [cache_path(c)
                for c in cals[cals.caltype.str.contains('arc')].filename.values]
        master_arc = reduce_arc(ad.data_label(), cals)
        input_data = reduce_data(ad, center, master_arc)

    else:
        raise IOError(
            'Could not find input file:\n' +
            '  {:s}\n'.format(input_path) +
            '  Run pytest with "--force-preprocess-data" to get it')

    return input_data


@pytest.fixture(scope="module")
def reduce_data(output_path):
    """
    Recipe used to generate input data for `extract1DSpectra` tests.

    Parameters
    ----------
    output_path : pytest.fixture
        Fixture containing a custom context manager used to enter and leave the
        output folder easily.

    Returns
    -------
    function : factory that will read the standard star file, process them
        using a custom recipe and return an AstroData object.
    """
    def _reduce_data(ad, center, arc):
        with output_path():
            p = primitives_gmos_spect.GMOSSpect([ad])
            p.prepare()
            p.addDQ(static_bpm=None)
            p.addVAR(read_noise=True)
            p.overscanCorrect()
            p.ADUToElectrons()
            p.addVAR(poisson_noise=True)
            p.mosaicDetectors()
            p.distortionCorrect(arc=arc)

            ad = p.makeIRAFCompatible()[0]
            ad = add_aperture_table(ad, center)

            p = primitives_gmos_spect.GMOSSpect([ad])
            p.traceApertures(trace_order=2, nsum=20, step=10, max_shift=0.09, max_missed=5)
            p.skyCorrectFromSlit(order=5, grow=0)

            ad = p.writeOutputs()[0]
        return ad
    return _reduce_data
