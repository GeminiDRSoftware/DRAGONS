#!/usr/bin/env python
"""
Tests related to GMOS Long-slit Spectroscopy Arc primitives.

Notes
-----
- The `indirect` argument on `@pytest.mark.parametrize` fixture forces the
  `ad` and `ad_ref` fixtures to be called and the AstroData object returned.
"""
import os
from warnings import warn

import numpy as np
import pytest

import astrodata
import geminidr
from astrodata import testing
from copy import deepcopy
from geminidr.gmos import primitives_gmos_spect, primitives_gmos_longslit


# Test parameters --------------------------------------------------------------
input_files = [
    # Process Arcs: GMOS-N ---
    # (Input File, fwidth, order, min_snr)
    "process_arcs/GMOS/N20100115S0346_distortionDetermined.fits",  # B600:0.500 EEV
    "process_arcs/GMOS/N20130112S0390_distortionDetermined.fits",  # B600:0.500 E2V
    "process_arcs/GMOS/N20170609S0173_distortionDetermined.fits",  # B600:0.500 HAM
    "process_arcs/GMOS/N20170403S0452_distortionDetermined.fits",  # B600:0.590 HAM Full Frame 1x1
    "process_arcs/GMOS/N20170415S0255_distortionDetermined.fits",  # B600:0.590 HAM Central Spectrum 1x1
    "process_arcs/GMOS/N20171016S0010_distortionDetermined.fits",  # B600:0.500 HAM, ROI="Central Spectrum", bin=1x2
    "process_arcs/GMOS/N20171016S0127_distortionDetermined.fits",  # B600:0.500 HAM, ROI="Full Frame", bin=1x2
    "process_arcs/GMOS/N20100307S0236_distortionDetermined.fits",  # B1200:0.445 EEV
    "process_arcs/GMOS/N20130628S0290_distortionDetermined.fits",  # B1200:0.420 E2V
    "process_arcs/GMOS/N20170904S0078_distortionDetermined.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/N20170627S0116_distortionDetermined.fits",  # B1200:0.520 HAM
    "process_arcs/GMOS/N20100830S0594_distortionDetermined.fits",  # R150:0.500 EEV
    "process_arcs/GMOS/N20100702S0321_distortionDetermined.fits",  # R150:0.700 EEV
    "process_arcs/GMOS/N20130606S0291_distortionDetermined.fits",  # R150:0.550 E2V
    "process_arcs/GMOS/N20130112S0574_distortionDetermined.fits",  # R150:0.700 E2V
    "process_arcs/GMOS/N20130809S0337_distortionDetermined.fits",  # R150:0.700 E2V
    "process_arcs/GMOS/N20140408S0218_distortionDetermined.fits",  # R150:0.700 E2V
    "process_arcs/GMOS/N20180119S0232_distortionDetermined.fits",  # R150:0.520 HAM
    "process_arcs/GMOS/N20180516S0214_distortionDetermined.fits",  # R150:0.610 HAM ROI="Central Spectrum", bin=2x2
    # "process_arcs/GMOS/N20171007S0439_distortionDetermined.fits",  # R150:0.650 HAM - todo: won't pass
    "process_arcs/GMOS/N20171007S0441_distortionDetermined.fits",  # R150:0.650 HAM
    "process_arcs/GMOS/N20101212S0213_distortionDetermined.fits",  # R400:0.550 EEV
    "process_arcs/GMOS/N20100202S0214_distortionDetermined.fits",  # R400:0.700 EEV
    "process_arcs/GMOS/N20130106S0194_distortionDetermined.fits",  # R400:0.500 E2V
    "process_arcs/GMOS/N20130422S0217_distortionDetermined.fits",  # R400:0.700 E2V
    "process_arcs/GMOS/N20170108S0210_distortionDetermined.fits",  # R400:0.660 HAM
    "process_arcs/GMOS/N20171113S0135_distortionDetermined.fits",  # R400:0.750 HAM
    "process_arcs/GMOS/N20100427S1276_distortionDetermined.fits",  # R600:0.675 EEV
    "process_arcs/GMOS/N20180120S0417_distortionDetermined.fits",  # R600:0.860 HAM
    "process_arcs/GMOS/N20100212S0143_distortionDetermined.fits",  # R831:0.450 EEV
    "process_arcs/GMOS/N20100720S0247_distortionDetermined.fits",  # R831:0.850 EEV
    "process_arcs/GMOS/N20130808S0490_distortionDetermined.fits",  # R831:0.571 E2V
    "process_arcs/GMOS/N20130830S0291_distortionDetermined.fits",  # R831:0.845 E2V
    "process_arcs/GMOS/N20170910S0009_distortionDetermined.fits",  # R831:0.653 HAM
    "process_arcs/GMOS/N20170509S0682_distortionDetermined.fits",  # R831:0.750 HAM
    "process_arcs/GMOS/N20181114S0512_distortionDetermined.fits",  # R831:0.865 HAM
    "process_arcs/GMOS/N20170416S0058_distortionDetermined.fits",  # R831:0.865 HAM
    "process_arcs/GMOS/N20170416S0081_distortionDetermined.fits",  # R831:0.865 HAM
    "process_arcs/GMOS/N20180120S0315_distortionDetermined.fits",  # R831:0.865 HAM
    # Process Arcs: GMOS-S ---
    # "process_arcs/GMOS/S20130218S0126_distortionDetermined.fits",  # B600:0.500 EEV - todo: won't pass
    "process_arcs/GMOS/S20130111S0278_distortionDetermined.fits",  # B600:0.520 EEV
    "process_arcs/GMOS/S20130114S0120_distortionDetermined.fits",  # B600:0.500 EEV
    "process_arcs/GMOS/S20130216S0243_distortionDetermined.fits",  # B600:0.480 EEV
    "process_arcs/GMOS/S20130608S0182_distortionDetermined.fits",  # B600:0.500 EEV
    "process_arcs/GMOS/S20131105S0105_distortionDetermined.fits",  # B600:0.500 EEV
    "process_arcs/GMOS/S20140504S0008_distortionDetermined.fits",  # B600:0.500 EEV
    "process_arcs/GMOS/S20170103S0152_distortionDetermined.fits",  # B600:0.600 HAM
    "process_arcs/GMOS/S20170108S0085_distortionDetermined.fits",  # B600:0.500 HAM
    "process_arcs/GMOS/S20130510S0103_distortionDetermined.fits",  # B1200:0.450 EEV
    "process_arcs/GMOS/S20130629S0002_distortionDetermined.fits",  # B1200:0.525 EEV
    "process_arcs/GMOS/S20131123S0044_distortionDetermined.fits",  # B1200:0.595 EEV
    "process_arcs/GMOS/S20170116S0189_distortionDetermined.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/S20170103S0149_distortionDetermined.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/S20170730S0155_distortionDetermined.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/S20171219S0117_distortionDetermined.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/S20170908S0189_distortionDetermined.fits",  # B1200:0.550 HAM
    "process_arcs/GMOS/S20131230S0153_distortionDetermined.fits",  # R150:0.550 EEV
    "process_arcs/GMOS/S20130801S0140_distortionDetermined.fits",  # R150:0.700 EEV
    "process_arcs/GMOS/S20170430S0060_distortionDetermined.fits",  # R150:0.717 HAM
    # "process_arcs/GMOS/S20170430S0063_distortionDetermined.fits",  # R150:0.727 HAM - todo: won't pass
    "process_arcs/GMOS/S20171102S0051_distortionDetermined.fits",  # R150:0.950 HAM
    "process_arcs/GMOS/S20130114S0100_distortionDetermined.fits",  # R400:0.620 EEV
    "process_arcs/GMOS/S20130217S0073_distortionDetermined.fits",  # R400:0.800 EEV
    "process_arcs/GMOS/S20170108S0046_distortionDetermined.fits",  # R400:0.550 HAM
    "process_arcs/GMOS/S20170129S0125_distortionDetermined.fits",  # R400:0.685 HAM
    "process_arcs/GMOS/S20170703S0199_distortionDetermined.fits",  # R400:0.800 HAM
    "process_arcs/GMOS/S20170718S0420_distortionDetermined.fits",  # R400:0.910 HAM
    # "process_arcs/GMOS/S20100306S0460_distortionDetermined.fits",  # R600:0.675 EEV - todo: won't pass
    # "process_arcs/GMOS/S20101218S0139_distortionDetermined.fits",  # R600:0.675 EEV - todo: won't pass
    "process_arcs/GMOS/S20110306S0294_distortionDetermined.fits",  # R600:0.675 EEV
    "process_arcs/GMOS/S20110720S0236_distortionDetermined.fits",  # R600:0.675 EEV
    "process_arcs/GMOS/S20101221S0090_distortionDetermined.fits",  # R600:0.690 EEV
    "process_arcs/GMOS/S20120322S0122_distortionDetermined.fits",  # R600:0.900 EEV
    "process_arcs/GMOS/S20130803S0011_distortionDetermined.fits",  # R831:0.576 EEV
    "process_arcs/GMOS/S20130414S0040_distortionDetermined.fits",  # R831:0.845 EEV
    "process_arcs/GMOS/S20170214S0059_distortionDetermined.fits",  # R831:0.440 HAM
    "process_arcs/GMOS/S20170703S0204_distortionDetermined.fits",  # R831:0.600 HAM
    "process_arcs/GMOS/S20171018S0048_distortionDetermined.fits",  # R831:0.865 HAM
]

reference_files = [
    "_".join(f.split("_")[:-1]) + "_distortionCorrected.fits"
    for f in input_files
]


# Local Fixtures and Helper Functions ------------------------------------------
@pytest.fixture(scope="module")
def ad(request, ad_factory, path_to_outputs, path_to_refs):
    """
    Loads existing input FITS files as AstroData objects, runs the
    `distortionCorrect` primitive on it, and return the output object free of
    distortion.

    This makes tests more efficient because the primitive is run only once,
    instead of N x Numbes of tests.

    If the input file does not exist, this fixture raises a IOError.

    If the input file does not exist **and** PyTest is called with the
    `--force-preprocess-data`, this fixture looks for cached raw data and
    process it. If the raw data does not exist, it is then cached via download
    from the Gemini Archive.

    Parameters
    ----------
    request : fixture
        PyTest's built-in fixture with information about the test itself.
    ad_factory : fixture
        Custom fixture defined in the `conftest.py` file that loads cached data,
        or download and/or process it if needed.
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
    fname = request.param

    p = primitives_gmos_spect.GMOSSpect([])
    p.viewer = geminidr.dormantViewer(p, None)

    print('\n\n Running test inside folder:\n  {}'.format(path_to_outputs))

    _ad = ad_factory(fname, preprocess_recipe)
    ad_out = p.distortionCorrect([_ad], arc=_ad, order=3, subsample=1)[0]

    tests_failed_before_module = request.session.testsfailed
    yield ad_out

    _dir = os.path.join(path_to_outputs, os.path.dirname(request.param))
    _ref_dir = os.path.join(path_to_refs, os.path.dirname(request.param))
    os.makedirs(_dir, exist_ok=True)

    if request.config.getoption("--do-plots"):
        do_plots(ad_out, _dir, _ref_dir)

    if request.session.testsfailed > tests_failed_before_module:
        fname_out = os.path.join(_dir, ad_out.filename)
        ad_out.write(filename=fname_out, overwrite=True)
        print('\n Saved file to:\n  {}\n'.format(fname_out))

    del ad_out


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


def preprocess_recipe(ad, path):
    """
    Recipe used to generate input data for `distortionCorrect` tests.

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
    _p.makeIRAFCompatible()

    ad = _p.determineDistortion(
        spatial_order=3, spectral_order=4, id_only=False, min_snr=5.,
        fwidth=None, nsum=10, max_shift=0.05, max_missed=5)[0]

    _p.writeOutputs(outfilename=os.path.join(path, ad.filename))

    return ad


# Tests Definitions ------------------------------------------------------------
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad, ad_ref", zip(input_files, reference_files), indirect=True)
def test_distortion_correct_is_stable(ad, ad_ref):
    """
    Runs the `distortionCorrect` primitive on a preprocessed data and compare
    its model with the one in the reference file.
    """
    assert ad.filename == ad_ref.filename

    data = np.ma.masked_invalid(ad[0].data)
    ref_data = np.ma.masked_invalid(ad_ref[0].data)

    np.testing.assert_allclose(data, ref_data, atol=1)


@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.parametrize("fname", input_files)
def test_full_frame_distortion_works_on_smaller_region(fname, path_to_inputs):
    """
    Takes a full-frame arc and self-distortion-corrects it. It then fakes
    subregions of this and corrects those using the full-frame distortion to
    confirm that the result is the same as the appropriate region of the
    distortion-corrected full-frame image. There's no need to do this more
    than once for a given binning, so we loop within the function, keeping
    track of binnings we've already processed.
    """
    subpath, basename = os.path.split(fname)
    basename, extension = os.path.splitext(basename)
    basename = basename.split('_')[0] + extension

    raw_fname = testing.download_from_archive(basename, path=subpath)

    _ad = astrodata.open(os.path.join(path_to_inputs, subpath, raw_fname))

    NSUB = 4  # we're going to take combos of horizontal quadrants
    completed_binnings = []

    xbin, ybin = _ad.detector_x_bin(), _ad.detector_y_bin()

    if _ad.detector_roi_setting() != "Full Fame" or (xbin, ybin) in completed_binnings:
        return

    p = primitives_gmos_longslit.GMOSLongslit([_ad])
    p.viewer.viewer_name = None
    p.prepare()
    p.addDQ(static_bpm=None)
    p.overscanCorrect()
    p.ADUToElectrons()
    p.mosaicDetectors(outstream='mosaic')
    # Speed things up a bit with a larger step
    p.determineDistortion(stream='mosaic', step=48 // ybin)

    ad_out = p.distortionCorrect(
        [deepcopy(_ad)], arc=p.streams['mosaic'][0], order=1)[0]

    for start in range(NSUB):
        for end in range(start + 1, NSUB + 1):
            ad_copy = deepcopy(_ad)
            y1b = start * _ad[0].shape[0] // NSUB
            y2b = end * _ad[0].shape[0] // NSUB
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
            ad2 = p.distortionCorrect([ad_copy], arc=p.streams['mosaic'][0],
                                      order=1)[0]

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
