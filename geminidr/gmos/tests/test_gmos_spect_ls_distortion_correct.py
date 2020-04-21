#!/usr/bin/env python
"""
Tests related to GMOS Long-slit Spectroscopy Arc primitives.

Notes
-----
- The `indirect` argument on `@pytest.mark.parametrize` fixture forces the
  `ad` and `ad_ref` fixtures to be called and the AstroData object returned.
"""
import numpy as np
import os
import pytest
from copy import deepcopy
from warnings import warn

import astrodata
import geminidr
from astrodata import testing
from geminidr.gmos import primitives_gmos_spect, primitives_gmos_longslit
from gempy.utils import logutils

# Test parameters --------------------------------------------------------------
# Each test input filename contains the original input filename with
# "_distortionDetermined" suffix
original_inputs = [
    # Process Arcs: GMOS-N ---
    # (Input File, fwidth, order, min_snr)
    "N20100115S0346.fits",  # B600:0.500 EEV
    # "N20130112S0390.fits",  # B600:0.500 E2V
    # "N20170609S0173.fits",  # B600:0.500 HAM
    # "N20170403S0452.fits",  # B600:0.590 HAM Full Frame 1x1
    # "N20170415S0255.fits",  # B600:0.590 HAM Central Spectrum 1x1
    # "N20171016S0010.fits",  # B600:0.500 HAM, ROI="Central Spectrum", bin=1x2
    # "N20171016S0127.fits",  # B600:0.500 HAM, ROI="Full Frame", bin=1x2
    # "N20100307S0236.fits",  # B1200:0.445 EEV
    # "N20130628S0290.fits",  # B1200:0.420 E2V
    # "N20170904S0078.fits",  # B1200:0.440 HAM
    # "N20170627S0116.fits",  # B1200:0.520 HAM
    # "N20100830S0594.fits",  # R150:0.500 EEV
    # "N20100702S0321.fits",  # R150:0.700 EEV
    # "N20130606S0291.fits",  # R150:0.550 E2V
    # "N20130112S0574.fits",  # R150:0.700 E2V
    # "N20130809S0337.fits",  # R150:0.700 E2V
    # "N20140408S0218.fits",  # R150:0.700 E2V
    # "N20180119S0232.fits",  # R150:0.520 HAM
    # "N20180516S0214.fits",  # R150:0.610 HAM ROI="Central Spectrum", bin=2x2
    # # "N20171007S0439.fits",  # R150:0.650 HAM - todo: won't pass
    # "N20171007S0441.fits",  # R150:0.650 HAM
    # "N20101212S0213.fits",  # R400:0.550 EEV
    # "N20100202S0214.fits",  # R400:0.700 EEV
    # "N20130106S0194.fits",  # R400:0.500 E2V
    # "N20130422S0217.fits",  # R400:0.700 E2V
    # "N20170108S0210.fits",  # R400:0.660 HAM
    # "N20171113S0135.fits",  # R400:0.750 HAM
    # "N20100427S1276.fits",  # R600:0.675 EEV
    # "N20180120S0417.fits",  # R600:0.860 HAM
    # "N20100212S0143.fits",  # R831:0.450 EEV
    # "N20100720S0247.fits",  # R831:0.850 EEV
    # "N20130808S0490.fits",  # R831:0.571 E2V
    # "N20130830S0291.fits",  # R831:0.845 E2V
    # "N20170910S0009.fits",  # R831:0.653 HAM
    # "N20170509S0682.fits",  # R831:0.750 HAM
    # "N20181114S0512.fits",  # R831:0.865 HAM
    # "N20170416S0058.fits",  # R831:0.865 HAM
    # "N20170416S0081.fits",  # R831:0.865 HAM
    # "N20180120S0315.fits",  # R831:0.865 HAM
    # # Process Arcs: GMOS-S ---
    # # "S20130218S0126.fits",  # B600:0.500 EEV - todo: won't pass
    # "S20130111S0278.fits",  # B600:0.520 EEV
    # "S20130114S0120.fits",  # B600:0.500 EEV
    # "S20130216S0243.fits",  # B600:0.480 EEV
    # "S20130608S0182.fits",  # B600:0.500 EEV
    # "S20131105S0105.fits",  # B600:0.500 EEV
    # "S20140504S0008.fits",  # B600:0.500 EEV
    # "S20170103S0152.fits",  # B600:0.600 HAM
    # "S20170108S0085.fits",  # B600:0.500 HAM
    # "S20130510S0103.fits",  # B1200:0.450 EEV
    # "S20130629S0002.fits",  # B1200:0.525 EEV
    # "S20131123S0044.fits",  # B1200:0.595 EEV
    # "S20170116S0189.fits",  # B1200:0.440 HAM
    # "S20170103S0149.fits",  # B1200:0.440 HAM
    # "S20170730S0155.fits",  # B1200:0.440 HAM
    # "S20171219S0117.fits",  # B1200:0.440 HAM
    # "S20170908S0189.fits",  # B1200:0.550 HAM
    # "S20131230S0153.fits",  # R150:0.550 EEV
    # "S20130801S0140.fits",  # R150:0.700 EEV
    # "S20170430S0060.fits",  # R150:0.717 HAM
    # # "S20170430S0063.fits",  # R150:0.727 HAM - todo: won't pass
    # "S20171102S0051.fits",  # R150:0.950 HAM
    # "S20130114S0100.fits",  # R400:0.620 EEV
    # "S20130217S0073.fits",  # R400:0.800 EEV
    # "S20170108S0046.fits",  # R400:0.550 HAM
    # "S20170129S0125.fits",  # R400:0.685 HAM
    # "S20170703S0199.fits",  # R400:0.800 HAM
    # "S20170718S0420.fits",  # R400:0.910 HAM
    # # "S20100306S0460.fits",  # R600:0.675 EEV - todo: won't pass
    # # "S20101218S0139.fits",  # R600:0.675 EEV - todo: won't pass
    # "S20110306S0294.fits",  # R600:0.675 EEV
    # "S20110720S0236.fits",  # R600:0.675 EEV
    # "S20101221S0090.fits",  # R600:0.690 EEV
    # "S20120322S0122.fits",  # R600:0.900 EEV
    # "S20130803S0011.fits",  # R831:0.576 EEV
    # "S20130414S0040.fits",  # R831:0.845 EEV
    # "S20170214S0059.fits",  # R831:0.440 HAM
    # "S20170703S0204.fits",  # R831:0.600 HAM
    # "S20171018S0048.fits",  # R831:0.865 HAM
]

fixed_test_parameters_for_determine_distortion = {
    "fwidth": None,
    "id_only": False,
    "max_missed": 5,
    "max_shift": 0.05,
    "min_snr": 5.,
    "nsum": 10,
    "spatial_order": 3,
    "spectral_order": 4,
}


# Tests Definitions ------------------------------------------------------------
@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("preprocessed_ad", original_inputs, indirect=True)
def test_regression_in_distortion_correct(output_path, preprocessed_ad,
                                          reference_ad):
    """
    Runs the `distortionCorrect` primitive on a preprocessed data and compare
    its model with the one in the reference file.
    """
    with output_path():
        p = primitives_gmos_longslit.GMOSLongslit([deepcopy(preprocessed_ad)])
        p.viewer = geminidr.dormantViewer(p, None)
        p.distortionCorrect(arc=deepcopy(preprocessed_ad), order=3, subsample=1)
        ad = p.writeOutputs().pop()

    ref_ad = reference_ad(ad.filename)
    data = np.ma.masked_invalid(ad[0].data)
    ref_data = np.ma.masked_invalid(ref_ad[0].data)
    np.testing.assert_allclose(data, ref_data, atol=1)


@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("preprocessed_ad", original_inputs, indirect=True)
def test_full_frame_distortion_works_on_smaller_region(output_path, preprocessed_ad):
    """
    Takes a full-frame arc and self-distortion-corrects it. It then fakes
    subregions of this and corrects those using the full-frame distortion to
    confirm that the result is the same as the appropriate region of the
    distortion-corrected full-frame image. There's no need to do this more
    than once for a given binning, so we loop within the function, keeping
    track of binnings we've already processed.
    """
    ad = preprocessed_ad

    NSUB = 4  # we're going to take combos of horizontal quadrants
    completed_binnings = []

    xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
    if ad.detector_roi_setting() != "Full Fame" or (xbin, ybin) in completed_binnings:
        return

    p = primitives_gmos_longslit.GMOSLongslit([ad])
    p.viewer.viewer_name = None
    ad_out = p.distortionCorrect([deepcopy(ad)], arc=ad, order=1)[0]

    for start in range(NSUB):
        for end in range(start + 1, NSUB + 1):
            ad_copy = deepcopy(ad)
            y1b = start * ad[0].shape[0] // NSUB
            y2b = end * ad[0].shape[0] // NSUB
            y1, y2 = y1b * ybin, y2b * ybin  # unbinned pixels

            # Fake the section header keywords and set the SCI and DQ
            # to the appropriate sub-region
            for ext in ad_copy:
                arrsec = ext.array_section()
                detsec = ext.detector_section()
                ext.hdr['CCDSEC'] = '[{}:{},{}:{}]'.format(arrsec.x1 + 1,
                                                           arrsec.x2, y1 + 1, y2)
                ext.hdr['DETSEC'] = '[{}:{},{}:{}]'.format(detsec.x1 + 1,
                                                           detsec.x2, y1 + 1, y2)
                ext.data = ext.data[y1b:y2b]
                ext.mask = ext.mask[y1b:y2b]
                ext.hdr['DATASEC'] = '[1:{},1:{}]'.format(ext.shape[1], y2b - y1b)

            ad2 = p.distortionCorrect(
                [ad_copy], arc=p.streams['mosaic'][0], order=1)[0]

        # It's GMOS LS so the offset between this AD and the full-frame
        # will be the same as the DETSEC offset, but the width may be
        # smaller so we need to shuffle the smaller image within the
        # larger one to look for a match
        ny, nx = ad2[0].shape
        xsizediff = ad_out[0].shape[1] - nx

        for xoffset in range(xsizediff + 1):
            np.testing.assert_allclose(
                np.ma.masked_array(ad2[0].data, mask=ad2[0].mask),
                ad_out[0].data[y1b:y1b + ny, xoffset:xoffset + nx],
                atol=0.01,
                err_msg="Problem with {} {}:{}".format(ad.filename, start, end))

        completed_binnings.append((xbin, ybin))


# Local Fixtures and Helper Functions ------------------------------------------
@pytest.fixture(scope="module")
def do_plots(ad, output_path, reference_path):
    """
    Generate diagnostic plots.

    Parameters
    ----------
    ad : AstroData
    output_path : str or Path
    reference_path : str or Path
    """
    try:
        from .plots_gmos_spect_longslit_arcs import PlotGmosSpectLongslitArcs
    except ImportError:
        warn("Could not generate plots")
        return

    ad_ref = astrodata.open(os.path.join(reference_path, ad.filename))

    p = PlotGmosSpectLongslitArcs(ad, output_folder=output_path, ref_folder=reference_path)
    p.show_distortion_map(ad)
    p.show_distortion_model_difference(ad, ad_ref)
    p.close_all()


@pytest.fixture(scope="module")
def preprocessed_ad(request, cache_path, new_path_to_inputs, reduce_data):
    """
    Reads the preprocessed input data or cache/process it in a temporary folder.

    Parameters
    ----------
    request : pytest.fixture
        Fixture that contains information this fixture's parent.
    cache_path : pytest.fixture
        Path to where the data will be temporarily cached.
    new_path_to_inputs : pytest.fixture
        Path to the permanent local input files.
    reduce_data : pytest.fixture
        Recipe to reduce the data up to the step before
        `determineWavelengthSolution`.

    Returns
    -------
    AstroData
        The distortion corrected object.

    Raises
    ------
    IOError : if the input file does not exist.
    """
    basename = request.param
    should_preprocess = request.config.getoption("--force-preprocess-data")

    input_fname = basename.replace('.fits', '_distortionDetermined.fits')
    input_path = os.path.join(new_path_to_inputs, input_fname)

    if os.path.exists(input_path):
        input_data = astrodata.open(input_path)

    elif should_preprocess:
        filename = cache_path(basename)
        input_data = reduce_data(astrodata.open(filename))

    else:
        raise IOError(
            'Could not find input file:\n' +
            '  {:s}\n'.format(input_path) +
            '  Run pytest with "--force-preprocess-data" to get it')

    return input_data


@pytest.fixture(scope="module")
def reduce_data(output_path):
    """
    Recipe used to generate input data for `distortionCorrect` tests.

    Parameters
    ----------
    output_path : pytest.fixture
        Fixture containing a custom context manager used to enter and leave the
        output folder easily.

    Returns
    -------
    function : A function that will read the standard star file, process them
        using a custom recipe and return an AstroData object.
    """
    def _reduce_data(ad):
        with output_path():
            # Use config to prevent outputs when running Reduce via API
            logutils.config(file_name='log_{}.txt'.format(ad.data_label()))

            _p = primitives_gmos_spect.GMOSSpect([ad])
            _p.prepare()
            _p.addDQ(static_bpm=None)
            _p.addVAR(read_noise=True)
            _p.overscanCorrect()
            _p.ADUToElectrons()
            _p.addVAR(poisson_noise=True)
            _p.mosaicDetectors()
            _p.makeIRAFCompatible()
            _p.determineDistortion(**fixed_test_parameters_for_determine_distortion)
            ad = _p.writeOutputs().pop()
        return ad
    return _reduce_data
