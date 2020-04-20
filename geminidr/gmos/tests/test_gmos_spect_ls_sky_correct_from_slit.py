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

import astrodata
import gemini_instruments
import geminidr
from astrodata import testing
from geminidr.gmos import primitives_gmos_spect


# Test parameters --------------------------------------------------------------
# Each test input filename contains the original input filename with
# "_aperturesTraced" suffix
test_datasets = [
    ("N20180508S0021.fits", 244),  # B600 720
    # ("N20180509S0010.fits", 259),  # R400 900
    # ("N20180516S0081.fits", 255),  # R600 860
    # ("N20190201S0163.fits", 255),  # B600 530
    # ("N20190313S0114.fits", 254),  # B600 482
    # ("N20190427S0123.fits", 260),  # R400 525
    # ("N20190427S0126.fits", 259),  # R400 625
    # ("N20190427S0127.fits", 258),  # R400 725
    # ("N20190427S0141.fits", 264),  # R150 660
    
    # Input Filename
    # ("process_arcs/GMOS/N20180508S0021_aperturesTraced.fits",
    #  "process_arcs/GMOS/N20180615S0409_distortionDetermined.fits", 244),  # B600 720
    # ("process_arcs/GMOS/N20180509S0010_aperturesTraced.fits",
    #  "process_arcs/GMOS/N20180509S0080_distortionDetermined.fits", 259),  # R400 900
    # ("process_arcs/GMOS/N20180516S0081_aperturesTraced.fits",
    #  "process_arcs/GMOS/N20180516S0214_distortionDetermined.fits", 255),  # R600 860
    # # ("process_arcs/GMOS/N20190201S0163_aperturesTraced.fits",
    # #  "process_arcs/GMOS/N20190201S0176_distortionDetermined.fits", 255),  # B600 530
    # ("process_arcs/GMOS/N20190313S0114_aperturesTraced.fits",
    #  "process_arcs/GMOS/N20190313S0132_distortionDetermined.fits", 254),  # B600 482
    # ("process_arcs/GMOS/N20190427S0123_aperturesTraced.fits",
    #  "process_arcs/GMOS/N20190427S0266_distortionDetermined.fits", 260),  # R400 525
    # ("process_arcs/GMOS/N20190427S0126_aperturesTraced.fits",
    #  "process_arcs/GMOS/N20190427S0267_distortionDetermined.fits", 259),  # R400 625
    # ("process_arcs/GMOS/N20190427S0127_aperturesTraced.fits",
    #  "process_arcs/GMOS/N20190427S0268_distortionDetermined.fits", 258),  # R400 725
    # ("process_arcs/GMOS/N20190427S0141_aperturesTraced.fits",
    #  "process_arcs/GMOS/N20190427S0270_distortionDetermined.fits", 264),  # R150 660
]


# Tests Definitions ------------------------------------------------------------
@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_regression_extract_1d_spectra(sky_subtracted_ad, reference_ad):
    ref_ad = reference_ad(sky_subtracted_ad.filename)
    for ext, ref_ext in zip(sky_subtracted_ad, ref_ad):
        np.testing.assert_allclose(ext.data, ref_ext.data)


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

    ad[0].APERTURE = aperture
    return ad


# def preprocess_arc_recipe(ad, path):
#     """
#     Recipe used to generate arc data for `skyCorrectFromSlit` tests.
#
#     Parameters
#     ----------
#     ad : AstroData
#         Input raw arc data loaded as AstroData.
#     path : str
#         Path that points to where the input data is cached.
#
#     Returns
#     -------
#     AstroData
#         Pre-processed arc data.
#     """
#     _p = primitives_gmos_spect.GMOSSpect([ad])
#
#     _p.prepare()
#     _p.addDQ(static_bpm=None)
#     _p.addVAR(read_noise=True)
#     _p.overscanCorrect()
#     _p.ADUToElectrons()
#     _p.addVAR(poisson_noise=True)
#     _p.mosaicDetectors()
#     _p.makeIRAFCompatible()
#
#     ad = _p.determineDistortion(
#         spatial_order=3, spectral_order=4, id_only=False, min_snr=5.,
#         fwidth=None, nsum=10, max_shift=0.05, max_missed=5)[0]
#
#     _p.writeOutputs(outfilename=os.path.join(path, ad.filename))
#
#     return ad


@pytest.fixture(scope='module')
def get_input_ad(cache_path, new_path_to_inputs, reduce_arc, reduce_data):
    """
    Reads the input data or cache/process it in a temporary folder.

    Parameters
    ----------
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
    function : factory that reads existing input data or call the data
        reduction recipe.
    """
    def _get_input_ad(basename, center, should_preprocess):
        assert isinstance(should_preprocess, bool)
        assert isinstance(center, int)
        input_fname = basename.replace('.fits', '_aperturesTraced.fits')
        input_path = os.path.join(new_path_to_inputs, input_fname)

        if should_preprocess:
            filename = cache_path(basename)
            ad = astrodata.open(filename)
            cals = testing.get_associated_calibrations(basename)
            cals = [cache_path(c)
                    for c in cals[cals.caltype.str.contains('arc')].filename.values]

            master_arc = reduce_arc(ad.data_label(), cals)
            input_data = reduce_data(ad, center, master_arc)

        elif os.path.exists(input_path):
            input_data = astrodata.open(input_path)

        else:
            raise IOError(
                'Could not find input file:\n' +
                '  {:s}\n'.format(input_path) +
                '  Run pytest with "--force-preprocess-data" to get it')

        return input_data
    return _get_input_ad


@pytest.fixture(scope='module')
def reduce_data(output_path):
    """
    Recipe used to generate input data for `skyCorrectFromSlit` tests.

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
            ad = p.writeOutputs()[0]
        return ad
    return _reduce_data


@pytest.fixture(scope='module', params=test_datasets)
def sky_subtracted_ad(request, get_input_ad, output_path):
    """
    Runs the `skyCorrectFromSlit` primitive on input data.

    Parameters
    ----------
    request : fixture
        PyTest's built-in fixture with information about the test itself.
    get_input_ad : pytest.fixture
        Fixture that reads the input data or cache/process it in a temporary
        folder.
    output_path : pytest.fixture
        Fixture containing a custom context manager used to enter and leave the
        output folder easily.

    Returns
    -------
    AstroData
        Sky-subtracted data.
    """
    filename, center = request.param
    pre_process = request.config.getoption("--force-preprocess-data")
    input_ad = get_input_ad(filename, center, pre_process)

    with output_path():
        p = primitives_gmos_spect.GMOSSpect([input_ad])
        p.viewer = geminidr.dormantViewer(p, None)
        p.skyCorrectFromSlit(order=5, grow=0)
        _sky_subtracted_ad = p.writeOutputs().pop()

    return _sky_subtracted_ad
