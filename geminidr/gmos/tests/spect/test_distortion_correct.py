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

import astrodata
import geminidr

from astrodata.testing import compare_models
from geminidr.gmos import primitives_gmos_longslit
from gempy.utils import logutils
from recipe_system.testing import ref_ad_factory

# Test parameters --------------------------------------------------------------
datasets = [
    # Process Arcs: GMOS-N ---
    "N20100115S0346_distortionDetermined.fits",  # B600:0.500 EEV
    # "N20130112S0390_distortionDetermined.fits",  # B600:0.500 E2V
    # "N20170609S0173_distortionDetermined.fits",  # B600:0.500 HAM
    # "N20170403S0452_distortionDetermined.fits",  # B600:0.590 HAM Full Frame 1x1
    # "N20170415S0255_distortionDetermined.fits",  # B600:0.590 HAM Central Spectrum 1x1
    # "N20171016S0010_distortionDetermined.fits",  # B600:0.500 HAM, ROI="Central Spectrum", bin=1x2
    # "N20171016S0127_distortionDetermined.fits",  # B600:0.500 HAM, ROI="Full Frame", bin=1x2
    # "N20100307S0236_distortionDetermined.fits",  # B1200:0.445 EEV
    # "N20130628S0290_distortionDetermined.fits",  # B1200:0.420 E2V
    # "N20170904S0078_distortionDetermined.fits",  # B1200:0.440 HAM
    # "N20170627S0116_distortionDetermined.fits",  # B1200:0.520 HAM
    # "N20100830S0594_distortionDetermined.fits",  # R150:0.500 EEV
    # "N20100702S0321_distortionDetermined.fits",  # R150:0.700 EEV
    # "N20130606S0291_distortionDetermined.fits",  # R150:0.550 E2V
    # "N20130112S0574_distortionDetermined.fits",  # R150:0.700 E2V
    # "N20130809S0337_distortionDetermined.fits",  # R150:0.700 E2V
    # "N20140408S0218_distortionDetermined.fits",  # R150:0.700 E2V
    # "N20180119S0232_distortionDetermined.fits",  # R150:0.520 HAM
    # "N20180516S0214_distortionDetermined.fits",  # R150:0.610 HAM ROI="Central Spectrum", bin=2x2
    # # "N20171007S0439_distortionDetermined.fits",  # R150:0.650 HAM - todo: won't pass
    # "N20171007S0441_distortionDetermined.fits",  # R150:0.650 HAM
    # "N20101212S0213_distortionDetermined.fits",  # R400:0.550 EEV
    # "N20100202S0214_distortionDetermined.fits",  # R400:0.700 EEV
    # "N20130106S0194_distortionDetermined.fits",  # R400:0.500 E2V
    # "N20130422S0217_distortionDetermined.fits",  # R400:0.700 E2V
    # "N20170108S0210_distortionDetermined.fits",  # R400:0.660 HAM
    # "N20171113S0135_distortionDetermined.fits",  # R400:0.750 HAM
    # "N20100427S1276_distortionDetermined.fits",  # R600:0.675 EEV
    # "N20180120S0417_distortionDetermined.fits",  # R600:0.860 HAM
    # "N20100212S0143_distortionDetermined.fits",  # R831:0.450 EEV
    # "N20100720S0247_distortionDetermined.fits",  # R831:0.850 EEV
    # "N20130808S0490_distortionDetermined.fits",  # R831:0.571 E2V
    # "N20130830S0291_distortionDetermined.fits",  # R831:0.845 E2V
    # "N20170910S0009_distortionDetermined.fits",  # R831:0.653 HAM
    # "N20170509S0682_distortionDetermined.fits",  # R831:0.750 HAM
    # "N20181114S0512_distortionDetermined.fits",  # R831:0.865 HAM
    # "N20170416S0058_distortionDetermined.fits",  # R831:0.865 HAM
    # "N20170416S0081_distortionDetermined.fits",  # R831:0.865 HAM
    # "N20180120S0315_distortionDetermined.fits",  # R831:0.865 HAM
    # # Process Arcs: GMOS-S ---
    # # "S20130218S0126_distortionDetermined.fits",  # B600:0.500 EEV - todo: won't pass
    # "S20130111S0278_distortionDetermined.fits",  # B600:0.520 EEV
    # "S20130114S0120_distortionDetermined.fits",  # B600:0.500 EEV
    # "S20130216S0243_distortionDetermined.fits",  # B600:0.480 EEV
    # "S20130608S0182_distortionDetermined.fits",  # B600:0.500 EEV
    # "S20131105S0105_distortionDetermined.fits",  # B600:0.500 EEV
    # "S20140504S0008_distortionDetermined.fits",  # B600:0.500 EEV
    # "S20170103S0152_distortionDetermined.fits",  # B600:0.600 HAM
    # "S20170108S0085_distortionDetermined.fits",  # B600:0.500 HAM
    # "S20130510S0103_distortionDetermined.fits",  # B1200:0.450 EEV
    # "S20130629S0002_distortionDetermined.fits",  # B1200:0.525 EEV
    # "S20131123S0044_distortionDetermined.fits",  # B1200:0.595 EEV
    # "S20170116S0189_distortionDetermined.fits",  # B1200:0.440 HAM
    # "S20170103S0149_distortionDetermined.fits",  # B1200:0.440 HAM
    # "S20170730S0155_distortionDetermined.fits",  # B1200:0.440 HAM
    # "S20171219S0117_distortionDetermined.fits",  # B1200:0.440 HAM
    # "S20170908S0189_distortionDetermined.fits",  # B1200:0.550 HAM
    # "S20131230S0153_distortionDetermined.fits",  # R150:0.550 EEV
    # "S20130801S0140_distortionDetermined.fits",  # R150:0.700 EEV
    # "S20170430S0060_distortionDetermined.fits",  # R150:0.717 HAM
    # # "S20170430S0063_distortionDetermined.fits",  # R150:0.727 HAM - todo: won't pass
    # "S20171102S0051_distortionDetermined.fits",  # R150:0.950 HAM
    # "S20130114S0100_distortionDetermined.fits",  # R400:0.620 EEV
    # "S20130217S0073_distortionDetermined.fits",  # R400:0.800 EEV
    # "S20170108S0046_distortionDetermined.fits",  # R400:0.550 HAM
    # "S20170129S0125_distortionDetermined.fits",  # R400:0.685 HAM
    # "S20170703S0199_distortionDetermined.fits",  # R400:0.800 HAM
    # "S20170718S0420_distortionDetermined.fits",  # R400:0.910 HAM
    # # "S20100306S0460_distortionDetermined.fits",  # R600:0.675 EEV - todo: won't pass
    # # "S20101218S0139_distortionDetermined.fits",  # R600:0.675 EEV - todo: won't pass
    # "S20110306S0294_distortionDetermined.fits",  # R600:0.675 EEV
    # "S20110720S0236_distortionDetermined.fits",  # R600:0.675 EEV
    # "S20101221S0090_distortionDetermined.fits",  # R600:0.690 EEV
    # "S20120322S0122_distortionDetermined.fits",  # R600:0.900 EEV
    # "S20130803S0011_distortionDetermined.fits",  # R831:0.576 EEV
    # "S20130414S0040_distortionDetermined.fits",  # R831:0.845 EEV
    # "S20170214S0059_distortionDetermined.fits",  # R831:0.440 HAM
    # "S20170703S0204_distortionDetermined.fits",  # R831:0.600 HAM
    # "S20171018S0048_distortionDetermined.fits",  # R831:0.865 HAM
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
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("ad", datasets, indirect=True)
def test_regression_in_distortion_correct(ad, change_working_dir, ref_ad_factory):
    """
    Runs the `distortionCorrect` primitive on a preprocessed data and compare
    its model with the one in the reference file.
    """
    with change_working_dir():

        logutils.config(
            file_name='log_regression_{:s}.txt'.format(ad.data_label()))

        p = primitives_gmos_longslit.GMOSLongslit([deepcopy(ad)])
        p.viewer = geminidr.dormantViewer(p, None)
        p.distortionCorrect(order=3, subsample=1)
        dist_corrected_ad = p.writeOutputs()[0]

    ref_ad = ref_ad_factory(dist_corrected_ad.filename)
    for ext, ext_ref in zip(dist_corrected_ad, ref_ad):
        data = np.ma.masked_invalid(ext.data)
        ref_data = np.ma.masked_invalid(ext_ref.data)
        np.testing.assert_allclose(data, ref_data, atol=1)
        # Compare output WCS as well as pixel values (by evaluating it at the
        # ends of the ranges, since there are multiple ways of constructing an
        # equivalent WCS, eg. depending on the order of various Shift models):
        for f in ext_ref.wcs.available_frames:
            assert repr(getattr(ext_ref.wcs, f)) == repr(getattr(ext.wcs, f))
        corner1, corner2 = (0, 0), tuple(v-1 for v in ext_ref.shape[::-1])
        world1, world2 = ext_ref.wcs(*corner1), ext_ref.wcs(*corner2)
        np.testing.assert_allclose(ext.wcs(*corner1), world1, rtol=1e-6)
        np.testing.assert_allclose(ext.wcs(*corner2), world2, rtol=1e-6)
        np.testing.assert_allclose(ext.wcs.invert(*world1),
                                   corner1, atol=1e-3)
        np.testing.assert_allclose(ext.wcs.invert(*world2),
                                   corner2, atol=1e-3)


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad", datasets, indirect=True)
def test_full_frame_distortion_works_on_smaller_region(ad, change_working_dir):
    """
    Takes a full-frame arc and self-distortion-corrects it. It then fakes
    sub-regions of this and corrects those using the full-frame distortion to
    confirm that the result is the same as the appropriate region of the
    distortion-corrected full-frame image. There's no need to do this more
    than once for a given binning, so we loop within the function, keeping
    track of binnings we've already processed.
    """
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
@pytest.fixture(scope='function')
def ad(path_to_inputs, request):
    """
    Returns the pre-processed spectrum file.

    Parameters
    ----------
    path_to_inputs : pytest.fixture
        Fixture defined in :mod:`astrodata.testing` with the path to the
        pre-processed input file.
    request : pytest.fixture
        PyTest built-in fixture containing information about parent test.

    Returns
    -------
    AstroData
        Input spectrum processed up to right before the `distortionCorrect`
        primitive.
    """
    filename = request.param
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        ad = astrodata.open(path)
    else:
        raise FileNotFoundError(path)

    return ad


# -- Recipe to create pre-processed data ---------------------------------------
def create_inputs_recipe():
    """
    Creates input data for tests using pre-processed standard star and its
    calibration files.

    The raw files will be downloaded and saved inside the path stored in the
    `$DRAGONS_TEST/raw_inputs` directory. Processed files will be stored inside
    a new folder called "dragons_test_inputs". The sub-directory structure
    should reflect the one returned by the `path_to_inputs` fixture.
    """
    import os
    from astrodata.testing import download_from_archive
    from geminidr.gmos.tests.spect import CREATED_INPUTS_PATH_FOR_TESTS

    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("inputs/", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    for filename in datasets:
        print('Downloading files...')
        basename = filename.split("_")[0] + ".fits"
        sci_path = download_from_archive(basename)
        sci_ad = astrodata.open(sci_path)
        data_label = sci_ad.data_label()

        print('Reducing pre-processed data:')
        logutils.config(file_name='log_{}.txt'.format(data_label))

        p = primitives_gmos_longslit.GMOSLongslit([sci_ad])
        p.prepare()
        p.addDQ(static_bpm=None, user_bpm=None, add_illum_mask=False)
        p.addVAR(read_noise=True, poisson_noise=False)
        p.overscanCorrect(function="spline", high_reject=3., low_reject=3.,
                          nbiascontam=0, niterate=2, order=None)
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True, read_noise=False)
        p.mosaicDetectors()
        p.makeIRAFCompatible()
        p.determineDistortion(**fixed_test_parameters_for_determine_distortion)

        os.chdir("inputs")
        processed_ad = p.writeOutputs().pop()
        os.chdir("../../")
        print('Wrote pre-processed file to:\n'
              '    {:s}'.format(processed_ad.filename))


if __name__ == '__main__':
    import sys

    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main()
