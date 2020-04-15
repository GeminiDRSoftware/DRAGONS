#!/usr/bin/env python
"""
Tests related to GMOS Long-slit Spectroscopy Arc primitives. `input_files` is a
list of tuples which contains:

 - the input filename,
 - the full-width-at-half-maximum measured using IRAF's splot,
 - the wavelength solution order guessed based on resudials (usually between 2 and 4),
 - the minimum signal to noise for detection, based on splot analysis.

The input data can be cached from the archive and/or processed using the
--force-preprocess-data command line option.

Notes
-----
- The `indirect` argument on `@pytest.mark.parametrize` fixture forces the
  `ad` and `ad_ref` fixtures to be called and the AstroData object returned.

  @bquint:
    It seems that the matching process depends heavily on the FWHM. Because of
    that, the FWHM was measured using `splot` (keys h, c) manually for each
    file. It basically tells how far the KDTreeFitter should look for a match.

    The fitting order was picked up after running the test and analysing the
    shape of the residuals.

    Finally, the min_snr was semi-arbitrary. It had an opposite effect from what
    I, expected. Sometimes, raising this number caused more peaks to be detected.
"""
import os
from warnings import warn

import numpy as np
import pytest

import astrodata
import geminidr

from astrodata import testing
from geminidr.gmos import primitives_gmos_spect
from gempy.library import astromodels
from gempy.adlibrary import dataselect

from .plots_gmos_spect_longslit_arcs import PlotGmosSpectLongslitArcs

# Test parameters --------------------------------------------------------------
determine_wavelength_solution_parameters = {
    'center': None,
    'nsum': 10,
    'linelist': None,
    'weighting': 'natural',
    'nbright': 0
}

input_files = [

    # Process Arcs: GMOS-N ---
    # (Input File, fwidth, order, min_snr)
    ("N20100115S0346.fits", 6., 2, 5.),  # B600:0.500 EEV
    # ("N20130112S0390.fits", 3., 2, 5.),  # B600:0.500 E2V
    # ("N20170609S0173.fits", 5., 2, 5.),  # B600:0.500 HAM
    # ("N20170403S0452.fits", 5., 2, 3.),  # B600:0.590 HAM Full Frame 1x1
    # ("N20170415S0255.fits", 5., 3, 3.),  # B600:0.590 HAM Central Spectrum 1x1
    # ("N20171016S0010.fits", 5., 2, 5.),  # B600:0.500 HAM, ROI="Central Spectrum", bin=1x2
    # ("N20171016S0127.fits", 5., 2, 5.),  # B600:0.500 HAM, ROI="Full Frame", bin=1x2
    # ("N20100307S0236.fits", 4., 2, 3.),  # B1200:0.445 EEV
    # ("N20130628S0290.fits", 5., 2, 3.),  # B1200:0.420 E2V - Looks Good
    # ("N20170904S0078.fits", 3., 3, 3.),  # B1200:0.440 HAM
    # ("N20170627S0116.fits", 2.5, 3, 10.),  # B1200:0.520 HAM
    # ("N20100830S0594.fits", 2.5, 2, 3.),  # R150:0.500 EEV - todo: is that strong line in the blue real?
    # ("N20100702S0321.fits", 2.5, 2, 3.),  # R150:0.700 EEV
    # ("N20130606S0291.fits", 5., 2, 3.),  # R150:0.550 E2V
    # ("N20130112S0574.fits", 4.5, 3, 3.),  # R150:0.700 E2V
    # ("N20130809S0337.fits", 3, 2, 3.),  # R150:0.700 E2V
    # ("N20140408S0218.fits", 3, 4, 3.),  # R150:0.700 E2V
    # ("N20180119S0232.fits", 5, 2, 10.),  # R150:0.520 HAM - todo: won't pass
    # ("N20180516S0214.fits", 3.5, 3, 5.),  # R150:0.610 HAM ROI="Central Spectrum", bin=2x2
    # ("N20171007S0439.fits", 3, 2, 10.),  # R150:0.650 HAM
    # ("N20171007S0441.fits", 6, 2, 5.),  # R150:0.650 HAM
    # ("N20101212S0213.fits", 5.5, 2, 3.),  # R400:0.550 EEV
    # ("N20100202S0214.fits", 6, 2, 3.),  # R400:0.700 EEV
    # ("N20130106S0194.fits", 6, 2, 3.),  # R400:0.500 E2V
    # ("N20130422S0217.fits", 4.5, 3, 3.),  # R400:0.700 E2V
    # ("N20170108S0210.fits", 6, 3, 3.),  # R400:0.660 HAM
    # ("N20171113S0135.fits", 5.5, 2, 3.),  # R400:0.750 HAM
    # ("N20100427S1276.fits", 5.5, 2, 3.),  # R600:0.675 EEV
    # # ("N20180120S0417.fits", 8, 3, 5.),  # R600:0.860 HAM - todo: won't pass
    # ("N20100212S0143.fits", 5.5, 3, 5.),  # R831:0.450 EEV
    # ("N20100720S0247.fits", 3.5, 3, 3.),  # R831:0.850 EEV
    # ("N20130808S0490.fits", 4., 3, 5.),  # R831:0.571 E2V
    # ("N20130830S0291.fits", 3.5, 3, 5.),  # R831:0.845 E2V
    # # ("N20170910S0009.fits", 4.5, 2, 3.),  # R831:0.653 HAM- todo: won't pass
    # ("N20170509S0682.fits", 4.5, 3, 3.),  # R831:0.750 HAM
    # ("N20181114S0512.fits", 4, 3, 15.),  # R831:0.865 HAM - todo: passes *only* with fwhm=4??
    # # ("N20170416S0058.fits", 6., 2, 5.),  # R831:0.865 HAM - todo: won't pass
    # # ("N20170416S0081.fits", 4, 2, 3.),  # R831:0.865 HAM - todo: won't pass
    # # ("N20180120S0315.fits", 3, 2, 15.),  # R831:0.865 HAM - todo: won't pass
    #
    # # Process Arcs: GMOS-S ---
    # ("S20130218S0126.fits", 5., 2, 10),  # B600:0.500 EEV
    # ("S20130111S0278.fits", 6, 3, 5.),  # B600:0.520 EEV
    # ("S20130114S0120.fits", 3, 2, 5.),  # B600:0.500 EEV
    # ("S20130216S0243.fits", 3, 2, 3.),  # B600:0.480 EEV
    # ("S20130608S0182.fits", 6, 3, 3.),  # B600:0.500 EEV
    # ("S20131105S0105.fits", 3, 2, 5.),  # B600:0.500 EEV
    # ("S20140504S0008.fits", 6, 3, 10.),  # B600:0.500 EEV
    # ("S20170103S0152.fits", 7, 2, 10.),  # B600:0.600 HAM
    # ("S20170108S0085.fits", 5.5, 2, 10.),  # B600:0.500 HAM - todo: detector partially empty
    # ("S20130510S0103.fits", 2.5, 2, 5.),  # B1200:0.450 EEV - todo: region without matches
    # ("S20130629S0002.fits", 7, 6, 5.),  # B1200:0.525 EEV - todo: order = 6!!
    # ("S20131123S0044.fits", 4, 2, 3.),  # B1200:0.595 EEV
    # ("S20170116S0189.fits", 5, 2, 3.),  # B1200:0.440 HAM
    # ("S20170103S0149.fits", 7, 2, 3.),  # B1200:0.440 HAM
    # ("S20170730S0155.fits", 3.5, 2, 3.),  # B1200:0.440 HAM
    # # ("S20171219S0117.fits", 4, 2, 3.),  # B1200:0.440 HAM - todo: won't pass
    # ("S20170908S0189.fits", 3, 2, 3.),  # B1200:0.550 HAM
    # ("S20131230S0153.fits", 3, 2, 10.),  # R150:0.550 EEV
    # ("S20130801S0140.fits", 6, 2, 15.),  # R150:0.700 EEV
    # ("S20170430S0060.fits", 3, 2, 15.),  # R150:0.717 HAM
    # # ("S20170430S0063.fits", 6, 2, 15.),  # R150:0.727 HAM - todo: not stable
    # ("S20171102S0051.fits", 6, 2, 5.),   # R150:0.950 HAM
    # ("S20130114S0100.fits", 6, 4, 15.),  # R400:0.620 EEV
    # ("S20130217S0073.fits", 4, 2, 5.),  # R400:0.800 EEV
    # ("S20170108S0046.fits", 3, 2, 3.),  # R400:0.550 HAM
    # ("S20170129S0125.fits", 3, 2, 3.),  # R400:0.685 HAM
    # ("S20170703S0199.fits", 5, 3, 3.),  # R400:0.800 HAM
    # ("S20170718S0420.fits", 5, 2, 3.),  # R400:0.910 HAM
    # ("S20100306S0460.fits", 6, 2, 15.),  # R600:0.675 EEV
    # ("S20101218S0139.fits", 6, 2, 10.),  # R600:0.675 EEV
    # ("S20110306S0294.fits", 6, 2, 5.),  # R600:0.675 EEV
    # ("S20110720S0236.fits", 6, 2, 5.),  # R600:0.675 EEV
    # ("S20101221S0090.fits", 4, 2, 3.),  # R600:0.690 EEV
    # ("S20120322S0122.fits", 5, 2, 3.),  # R600:0.900 EEV
    # ("S20130803S0011.fits", 2, 2, 3.),  # R831:0.576 EEV
    # ("S20130414S0040.fits", 4, 2, 10.),  # R831:0.845 EEV
    # ("S20170214S0059.fits", 2, 2, 10.),  # R831:0.440 HAM - todo: the numbers says it is fine but I can't tell by the plots
    # ("S20170703S0204.fits", 3, 2, 3.),  # R831:0.600 HAM
    # ("S20171018S0048.fits", 5, 2, 3.)  # R831:0.865 HAM - todo: the numbers says it is fine but I can't tell by the plots
]


# Tests Definitions ------------------------------------------------------------
@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_reduced_arcs_contain_wavelength_solution_model_with_expected_rms(wlength_calibrated_ad):
    """
    Make sure that the WAVECAL model was fitted with an RMS smaller than half of
    the slit size in pixels.

    todo: this test must change with the slit size. While checking that, I found
        out that the `ad[0].slit()` descriptor returns nothing. I could use the
        existing `ad[0].focal_plane_mask()` descriptor for now but it is
        counter-intuitive.

    """
    table = wlength_calibrated_ad[0].WAVECAL
    coefficients = table["coefficients"]
    rms = coefficients[table["name"] == "rms"]

    pixel_scale = wlength_calibrated_ad[0].pixel_scale()  # arcsec / px
    slit_size_in_arcsec = float(wlength_calibrated_ad[0].focal_plane_mask().replace('arcsec', ''))
    slit_size_in_px = slit_size_in_arcsec / pixel_scale  # px
    dispersion = abs(wlength_calibrated_ad[0].dispersion(asNanometers=True))  # nm / px

    required_rms = dispersion * slit_size_in_px

    np.testing.assert_array_less(rms, required_rms)


@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_reduced_arcs_contains_stable_wavelength_solution(
        wlength_calibrated_ad, reference_ad):
    """
    Make sure that the wavelength solution gives same results on different
    runs.
    """
    ref_ad = reference_ad(wlength_calibrated_ad.filename)
    table = wlength_calibrated_ad[0].WAVECAL
    table_ref = ref_ad[0].WAVECAL

    model = astromodels.dict_to_chebyshev(
        dict(zip(table["name"], table["coefficients"])))

    ref_model = astromodels.dict_to_chebyshev(
        dict(zip(table_ref["name"], table_ref["coefficients"])))

    x = np.arange(wlength_calibrated_ad[0].shape[1])
    wavelength = model(x)
    ref_wavelength = ref_model(x)

    pixel_scale = wlength_calibrated_ad[0].pixel_scale()  # arcsec / px
    slit_size_in_arcsec = float(wlength_calibrated_ad[0].focal_plane_mask().replace('arcsec', ''))
    slit_size_in_px = slit_size_in_arcsec / pixel_scale
    dispersion = abs(wlength_calibrated_ad[0].dispersion(asNanometers=True))  # nm / px

    tolerance = 0.5 * (slit_size_in_px * dispersion)
    np.testing.assert_allclose(wavelength, ref_wavelength, rtol=tolerance)


# Local Fixtures and Helper Functions ------------------------------------------
@pytest.fixture(scope='module', params=input_files)
def wlength_calibrated_ad(request, get_input_ad, output_path):
    """
    Loads existing input FITS files as AstroData objects, runs the
    `determineWavelengthSolution` on it and return the output object with a
    `.WAVECAL` table.

    This makes tests more efficient because the primitive is run only once,
    instead of x n_tests.

    If the input file does not exist, this fixture raises a IOError.

    If the input file does not exist and PyTest is called with the
    `--force-preprocess-data`, this fixture looks for cached raw data and
    process it. If the raw data does not exist, it is then cached via download
    from the Gemini Archive.

    Parameters
    ----------
    request
    get_input_ad
    output_path

    Returns
    -------

    """
    filename, fwidth, order, min_snr = request.param
    pre_process = request.config.getoption("--force-preprocess-data")

    input_ad = get_input_ad(filename, pre_process)

    with output_path():
        p = primitives_gmos_spect.GMOSSpect([input_ad])
        p.viewer = geminidr.dormantViewer(p, None)

        p.determineWavelengthSolution(
            order=order, min_snr=min_snr, fwidth=fwidth,
            **determine_wavelength_solution_parameters)

        wcalibrated_ad = p.writeOutputs().pop()

        if request.config.getoption("--do-plots"):
            do_plots(wcalibrated_ad, "./")

    return wcalibrated_ad


@pytest.fixture(scope='module')
def get_input_ad(cache_path, new_path_to_inputs, reduce_data):
    """
    Reads the input data or cache/process it in a temporary folder.

    Parameters
    ----------
    cache_path : pytest.fixture
        Path to where the data will be temporarily cached.
    new_path_to_inputs : pytest.fixture
        Path to the permanent local input files.
    reduce_data : pytest.fixture
        Recipe to reduce the data up to the step before
        `determineWavelengthSolution`.

    Returns
    -------
    flat_corrected_ad : AstroData
        Bias and flat corrected data.
    master_arc : AstroData
        Master arc data.
    """
    def _get_input_ad(basename, should_preprocess):
        input_fname = basename.replace('.fits', '_mosaic.fits')
        input_path = os.path.join(new_path_to_inputs, input_fname)

        if should_preprocess:
            filename = cache_path(basename)
            input_data = reduce_data(astrodata.open(filename))

        elif os.path.exists(input_path):
            input_data = astrodata.open(input_path)

        else:
            raise IOError(
                'Could not find input file:\n' +
                '  {:s}\n'.format(input_path) +
                '  Run pytest with "--force-preprocess-data" to get it')

        return input_data
    return _get_input_ad


def do_plots(ad, output_dir):
    """
    Generate diagnostic plots.

    Parameters
    ----------
    ad : astrodata
    output_dir : str
    """
    try:
        p = PlotGmosSpectLongslitArcs(ad, output_dir)
        p.wavelength_calibration_plots()
        p.close_all()

    except ImportError:
        warn("Could not generate plots")


@pytest.fixture(scope='module')
def reduce_data(output_path):
    """
    Recipe used to generate input data for `determineWavelengthSolution` tests.

    Parameters
    ----------
    output_path : pytest.fixture
        Context manager used to write reduced data to a temporary folder.

    Returns
    -------
    function : A function that will read the standard star file, process them
        using a custom recipe and return an AstroData object.
    """
    def _reduce_data(ad):
        with output_path():
            _p = primitives_gmos_spect.GMOSSpect([ad])
            _p.prepare()
            _p.addDQ(static_bpm=None)
            _p.addVAR(read_noise=True)
            _p.overscanCorrect()
            _p.ADUToElectrons()
            _p.addVAR(poisson_noise=True)
            _p.mosaicDetectors()
            _p.makeIRAFCompatible()
            ad = _p.writeOutputs()[0]
        return ad
    return _reduce_data

