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
associated_calibrations = {
    "N20100115S0312.fits": {              # GN B600:0.500 EEV
        'bias': ["N20100113S0137.fits",
                 "N20100113S0138.fits",
                 "N20100113S0139.fits",
                 "N20100113S0140.fits",
                 "N20100113S0141.fits"],
        'flat': ["N20100115S0311.fits"],
        'arcs': ["N20100115S0346.fits"],
    },
    "S20200116S0104.fits": {              # GS R400:0.850 HAM CS
        'bias': ["S20200116S0425.fits",
                 "S20200116S0426.fits",
                 "S20200116S0427.fits",
                 "S20200116S0428.fits",
                 "S20200116S0429.fits",],
        'flat': ["S20200116S0103.fits"],
        'arcs': ["S20200116S0357.fits"],
    },
}
sci_datasets = [key.replace('.fits', '_QECorrected.fits')
                for key in associated_calibrations]
arc_datasets = [cals['arcs'][0].replace('.fits', '_arc.fits')
                for cals in associated_calibrations.values()]
datasets = arc_datasets + sci_datasets


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


def compare_frames(frame1, frame2):
    """Compare the important stuff of two CoordinateFrame instances"""
    for attr in ("naxes", "axes_type", "axes_order", "unit", "axes_names"):
        assert getattr(frame1, attr) == getattr(frame2, attr)


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
        p.distortionCorrect(interpolant="spline3", subsample=1)
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
            compare_frames(getattr(ext_ref.wcs, f), getattr(ext.wcs, f))
        corner1, corner2 = (0, 0), tuple(v-1 for v in ext_ref.shape[::-1])
        world1, world2 = ext_ref.wcs(*corner1), ext_ref.wcs(*corner2)
        np.testing.assert_allclose(ext.wcs(*corner1), world1, rtol=1e-6)
        np.testing.assert_allclose(ext.wcs(*corner2), world2, rtol=1e-6)
        # The inverse is not highly accurate and transforming back & forth can
        # just exceed even a tolerance of 0.01 pix, but it's best to compare
        # with the true corner values rather than the same results from the
        # reference, otherwise it would be too easy to overlook a problem with
        # the inverse when someone checks the reference file.
        np.testing.assert_allclose(ext.wcs.invert(*world1),
                                   corner1, atol=0.02)
        np.testing.assert_allclose(ext.wcs.invert(*world2),
                                   corner2, atol=0.02)


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


# -- Recipe to create pre-processed data (no wave cal, just distortion) -------
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
    from recipe_system.reduction.coreReduce import Reduce
    from recipe_system.utils.reduce_utils import normalize_ucals

    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("inputs/", exist_ok=True)
    print('Current working directory:\n    {:s}'.format(os.getcwd()))

    # This wants refactoring a bit so there's less duplication WRT QECorrect
    # and any variants of the distortionCorrect tests.
    for filename, cals in associated_calibrations.items():
        print(filename)
        # print(cals)

        sci_path = download_from_archive(filename)
        bias_paths = [download_from_archive(f) for f in cals['bias']]
        flat_paths = [download_from_archive(f) for f in cals['flat']]
        arc_paths = [download_from_archive(f) for f in cals['arcs']]

        sci_ad = astrodata.open(sci_path)
        arc_ad = astrodata.open(arc_paths[0])
        data_label = sci_ad.data_label()
        # is_ham = sci_ad.detector_name(pretty=True).startswith('Hamamatsu')

        logutils.config(file_name='log_bias_{}.txt'.format(data_label))
        bias_reduce = Reduce()
        bias_reduce.files.extend(bias_paths)
        bias_reduce.runr()
        bias_master = bias_reduce.output_filenames.pop()
        calibration_files = ['processed_bias:{}'.format(bias_master)]
        del bias_reduce

        logutils.config(file_name='log_flat_{}.txt'.format(data_label))
        flat_reduce = Reduce()
        flat_reduce.files.extend(flat_paths)
        flat_reduce.ucals = normalize_ucals(calibration_files)
        flat_reduce.runr()
        flat_master = flat_reduce.output_filenames.pop()
        calibration_files.append('processed_flat:{}'.format(flat_master))
        del flat_reduce

        logutils.config(file_name='log_arc_{}.txt'.format(data_label))

        p = primitives_gmos_longslit.GMOSLongslit([arc_ad])
        p.prepare()
        p.addDQ()
        p.addVAR(read_noise=True, poisson_noise=False)
        p.overscanCorrect()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True, read_noise=False)
        p.mosaicDetectors()
        p.makeIRAFCompatible()
        p.determineDistortion(**fixed_test_parameters_for_determine_distortion)
        p.storeProcessedArc()

        os.chdir("inputs")
        processed_arc = p.writeOutputs().pop()
        os.chdir("../")
        print('Wrote pre-processed arc to:\n'
              '    {:s}'.format(processed_arc.filename))

        logutils.config(file_name='log_{}.txt'.format(data_label))
        p = primitives_gmos_longslit.GMOSLongslit([sci_ad])
        p.prepare()
        p.addDQ()
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.biasCorrect(bias=bias_master)
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        p.attachWavelengthSolution(arc=processed_arc)
        p.flatCorrect(flat=flat_master)
        p.QECorrect()  # requires QECorrect bug fix

        os.chdir("inputs")
        processed_sci = p.writeOutputs().pop()
        os.chdir("../")
        print('Wrote pre-processed sci to:\n'
              '    {:s}'.format(processed_sci.filename))


if __name__ == '__main__':
    import sys

    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
    else:
        pytest.main()
