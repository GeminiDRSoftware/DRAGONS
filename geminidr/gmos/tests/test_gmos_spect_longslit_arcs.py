#!/usr/bin/env python
"""
Tests related to GMOS Long-slit Spectroscopy Arc primitives.
"""
import os

import numpy as np
from copy import deepcopy

# noinspection PyPackageRequirements
import pytest

# noinspection PyPackageRequirements
from astropy.modeling import models
from scipy import ndimage

import astrodata

# noinspection PyUnresolvedReferences
import gemini_instruments
import geminidr
from geminidr.gmos import primitives_gmos_spect
from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit
from gempy.library import astromodels
from gempy.utils import logutils

dataset_file_list = [
    # Process Arcs: GMOS-N ---
    "process_arcs/GMOS/N20100115S0346.fits",  # B600:0.500 EEV
    "process_arcs/GMOS/N20130112S0390.fits",  # B600:0.500 E2V
    "process_arcs/GMOS/N20170609S0173.fits",  # B600:0.500 HAM
    "process_arcs/GMOS/N20170403S0452.fits",  # B600:0.590 HAM Full Frame 1x1
    "process_arcs/GMOS/N20170415S0255.fits",  # B600:0.590 HAM Central Spectrum 1x1
    "process_arcs/GMOS/N20171016S0010.fits",  # B600:0.500 HAM, ROI="Central Spectrum", bin=1x2
    "process_arcs/GMOS/N20171016S0127.fits",  # B600:0.500 HAM, ROI="Full Frame", bin=1x2
    "process_arcs/GMOS/N20100307S0236.fits",  # B1200:0.445 EEV
    "process_arcs/GMOS/N20130628S0290.fits",  # B1200:0.420 E2V
    "process_arcs/GMOS/N20170904S0078.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/N20170627S0116.fits",  # B1200:0.520 HAM
    "process_arcs/GMOS/N20100830S0594.fits",  # R150:0.500 EEV
    "process_arcs/GMOS/N20100702S0321.fits",  # R150:0.700 EEV
    # "process_arcs/GMOS/N20130606S0291.fits",  # R150:0.550 E2V - todo: test_determine_distortion fails
    "process_arcs/GMOS/N20130112S0574.fits",  # R150:0.700 E2V
    # 'process_arcs/GMOS/N20130809S0337.fits',  # R150:0.700 E2V - todo: RMS > 0.5 (RMS = 0.59)
    # "process_arcs/GMOS/N20140408S0218.fits",  # R150:0.700 E2V - todo: RMS > 0.5 (RMS = 0.51)
    # 'process_arcs/GMOS/N20180119S0232.fits',  # R150:0.520 HAM - todo: RMS > 0.5 (RMS = 0.73)
    # 'process_arcs/GMOS/N20180516S0214.fits',  # R150:0.610 HAM ROI="Central Spectrum", bin=2x2 - todo: fails test_distortion_model_is_the_same
    # "process_arcs/GMOS/N20171007S0439.fits",  # R150:0.650 HAM - todo: breaks test_reduced_arcs_contains_stable_wavelength_solution
    "process_arcs/GMOS/N20171007S0441.fits",  # R150:0.650 HAM
    "process_arcs/GMOS/N20101212S0213.fits",  # R400:0.550 EEV
    "process_arcs/GMOS/N20100202S0214.fits",  # R400:0.700 EEV
    "process_arcs/GMOS/N20130106S0194.fits",  # R400:0.500 E2V
    "process_arcs/GMOS/N20130422S0217.fits",  # R400:0.700 E2V
    "process_arcs/GMOS/N20170108S0210.fits",  # R400:0.660 HAM
    "process_arcs/GMOS/N20171113S0135.fits",  # R400:0.750 HAM
    "process_arcs/GMOS/N20100427S1276.fits",  # R600:0.675 EEV
    "process_arcs/GMOS/N20180120S0417.fits",  # R600:0.860 HAM - todo: RMS > 0.5 (RMS = 0.58)
    "process_arcs/GMOS/N20100212S0143.fits",  # R831:0.450 EEV
    "process_arcs/GMOS/N20100720S0247.fits",  # R831:0.850 EEV
    "process_arcs/GMOS/N20130808S0490.fits",  # R831:0.571 E2V
    "process_arcs/GMOS/N20130830S0291.fits",  # R831:0.845 E2V
    "process_arcs/GMOS/N20170910S0009.fits",  # R831:0.653 HAM
    "process_arcs/GMOS/N20170509S0682.fits",  # R831:0.750 HAM
    # 'process_arcs/GMOS/N20181114S0512.fits',  # R831:0.865 HAM - todo: RMS > 0.5 (RMS = 0.52) | `gswavelength` cannot find solution either.
    "process_arcs/GMOS/N20170416S0058.fits",  # R831:0.865 HAM
    "process_arcs/GMOS/N20170416S0081.fits",  # R831:0.865 HAM
    "process_arcs/GMOS/N20180120S0315.fits",  # R831:0.865 HAM
    # # Process Arcs: GMOS-S ---
    # 'process_arcs/GMOS/S20130218S0126.fits',  # B600:0.500 EEV - todo: breaks p.determineWavelengthSolution() | `gswavelength` cannot find solution either.
    "process_arcs/GMOS/S20130111S0278.fits",  # B600:0.520 EEV
    "process_arcs/GMOS/S20130114S0120.fits",  # B600:0.500 EEV
    "process_arcs/GMOS/S20130216S0243.fits",  # B600:0.480 EEV
    "process_arcs/GMOS/S20130608S0182.fits",  # B600:0.500 EEV
    "process_arcs/GMOS/S20131105S0105.fits",  # B600:0.500 EEV
    "process_arcs/GMOS/S20140504S0008.fits",  # B600:0.500 EEV
    "process_arcs/GMOS/S20170103S0152.fits",  # B600:0.600 HAM
    "process_arcs/GMOS/S20170108S0085.fits",  # B600:0.500 HAM
    "process_arcs/GMOS/S20130510S0103.fits",  # B1200:0.450 EEV
    "process_arcs/GMOS/S20130629S0002.fits",  # B1200:0.525 EEV
    "process_arcs/GMOS/S20131123S0044.fits",  # B1200:0.595 EEV
    'process_arcs/GMOS/S20170116S0189.fits',  # B1200:0.440 HAM - todo: very weird non-linear plot | non-linear plot using `gswavelength` seems fine.
    "process_arcs/GMOS/S20170103S0149.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/S20170730S0155.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/S20171219S0117.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/S20170908S0189.fits",  # B1200:0.550 HAM
    "process_arcs/GMOS/S20131230S0153.fits",  # R150:0.550 EEV
    # "process_arcs/GMOS/S20130801S0140.fits",  # R150:0.700 EEV - todo: RMS > 0.5 (RMS = 0.69)
    # "process_arcs/GMOS/S20170430S0060.fits",  # R150:0.717 HAM - todo: RMS > 0.5 (RMS = 0.78)
    # "process_arcs/GMOS/S20170430S0063.fits",  # R150:0.727 HAM - todo: RMS > 0.5 (RMS = 1.26)
    "process_arcs/GMOS/S20171102S0051.fits",  # R150:0.950 HAM
    "process_arcs/GMOS/S20130114S0100.fits",  # R400:0.620 EEV
    "process_arcs/GMOS/S20130217S0073.fits",  # R400:0.800 EEV
    # "process_arcs/GMOS/S20170108S0046.fits",  # R400:0.550 HAM - todo: RMS > 0.5 (RMS = 0.60)
    "process_arcs/GMOS/S20170129S0125.fits",  # R400:0.685 HAM
    "process_arcs/GMOS/S20170703S0199.fits",  # R400:0.800 HAM
    "process_arcs/GMOS/S20170718S0420.fits",  # R400:0.910 HAM
    # 'process_arcs/GMOS/S20100306S0460.fits',  # R600:0.675 EEV - todo: breaks p.determineWavelengthSolution
    # 'process_arcs/GMOS/S20101218S0139.fits',  # R600:0.675 EEV - todo: breaks p.determineWavelengthSolution
    "process_arcs/GMOS/S20110306S0294.fits",  # R600:0.675 EEV
    # 'process_arcs/GMOS/S20110720S0236.fits',  # R600:0.675 EEV - todo: test_determine_distortion fails
    "process_arcs/GMOS/S20101221S0090.fits",  # R600:0.690 EEV
    "process_arcs/GMOS/S20120322S0122.fits",  # R600:0.900 EEV
    "process_arcs/GMOS/S20130803S0011.fits",  # R831:0.576 EEV
    "process_arcs/GMOS/S20130414S0040.fits",  # R831:0.845 EEV
    "process_arcs/GMOS/S20170214S0059.fits",  # R831:0.440 HAM
    "process_arcs/GMOS/S20170703S0204.fits",  # R831:0.600 HAM
    "process_arcs/GMOS/S20171018S0048.fits",  # R831:0.865 HAM
]


@pytest.fixture(scope="class", params=dataset_file_list)
def config(request, path_to_inputs, path_to_outputs, path_to_refs):
    """
    Super fixture that returns an object with the data required for the tests
    inside this file. This super fixture avoid confusions with Pytest, Fixtures
    and Parameters that could generate a very large matrix of configurations.

    The `path_to_*` fixtures are defined inside the `conftest.py` file.

    Parameters
    ----------
    request :
    path_to_inputs : pytest.fixture
        Fixture inherited from astrodata.testing with path to the input files.
    path_to_outputs : pytest.fixture
        Fixture inherited from astrodata.testing with path to the output files.
    path_to_refs : pytest.fixture
        Fixture inherited from astrodata.testing with path to the reference files.

    Returns
    -------
    namespace
        An object that contains `.ad`, `.output_dir`, `.ref_dir`, and
        `.filename` attributes.
    """
    # Setup log ---
    log_file = os.path.join(
        path_to_outputs,
        "{}.log".format(os.path.splitext(os.path.basename(__file__))[0]),
    )

    logutils.config(mode="quiet", file_name=log_file)

    c = ConfigTest(request.param, path_to_inputs, path_to_outputs, path_to_refs)
    yield c

    do_plots(c.ad, c.output_dir, c.ref_dir)
    del c


def do_plots(ad, output_dir, ref_dir):
    """
    Generate diagnostic plots.

    Parameters
    ----------
    ad : astrodata
    output_dir : str
    ref_dir : str
    """
    try:
        from .plots_gmos_spect_longslit_arcs import PlotGmosSpectLongslitArcs

        p = PlotGmosSpectLongslitArcs(ad, output_dir, ref_dir)
        p.wavelength_calibration_plots()
        p.distortion_diagnosis_plots()
        p.close_all()

    except ImportError:
        from warnings import warn

        warn("Could not generate plots")


class ConfigTest:
    """
    Config class created for each dataset file. It is created from within
    this a fixture so it can inherit the `path_to_*` fixtures as well.
    """

    def __init__(self, filename, input_dir, output_dir, ref_dir):
        input_file = os.path.join(input_dir, filename)
        dataset_sub_dir = os.path.dirname(filename)

        reference_folder = os.path.join(ref_dir, dataset_sub_dir)
        output_folder = os.path.join(output_dir, dataset_sub_dir)

        old_mask = os.umask(000)
        os.makedirs(output_folder, exist_ok=True, mode=0o775)
        os.umask(old_mask)

        output_file = os.path.join(output_dir, filename)
        output_file, _ = os.path.splitext(output_file)
        output_file = output_file + "_arc.fits"
        output_file = os.path.join(output_folder, output_file)

        r = self.reduce(input_file)

        ad = r.writeOutputs(outfilename=output_file, overwrite=True)[0]

        self.ad = ad
        self.filename = ad.filename
        self.output_file = output_file
        self.output_dir = output_folder
        self.ref_dir = reference_folder

    @staticmethod
    def reduce(filename):

        _p = primitives_gmos_spect.GMOSSpect([astrodata.open(filename)])
        _p.viewer = geminidr.dormantViewer(_p, None)

        _p.prepare()
        _p.addDQ(static_bpm=None)
        _p.addVAR(read_noise=True)
        _p.overscanCorrect()
        _p.ADUToElectrons()
        _p.addVAR(poisson_noise=True)
        _p.mosaicDetectors()
        _p.makeIRAFCompatible()
        _p.determineWavelengthSolution(suffix="_arc")

        return _p


@pytest.mark.gmosls
class TestGmosSpectLongslitArcs:
    """
    Collection of tests that will run on every `dataset` file.

    Here, it is important to keep the order of the tests since it reflects the
    order that they are executed.
    """

    @staticmethod
    def test_reduced_arcs_contain_wavelength_solution_model_with_expected_rms(config):
        """
        Make sure that the WAVECAL model was fitted with an RMS smaller
        than 0.5.
        """
        ad_out = config.ad

        for ext in ad_out:

            if not hasattr(ext, "WAVECAL"):
                continue

            table = ext.WAVECAL
            coefficients = table["coefficients"]
            rms = coefficients[table["name"] == "rms"]

            np.testing.assert_array_less(rms, 0.5)

        del ad_out

    @staticmethod
    def test_reduced_arcs_contains_stable_wavelength_solution(config):
        """
        Make sure that the wavelength solution gives same results on different
        runs.
        """
        output = os.path.join(config.output_dir, config.filename)
        reference = os.path.join(config.ref_dir, config.filename)

        if not os.path.exists(output):
            pytest.skip("Output file not found: {}".format(output))

        if not os.path.exists(reference):
            pytest.fail("Reference file not found: {}".format(reference))

        ad_out = config.ad
        ad_ref = astrodata.open(reference)

        for ext_out, ext_ref in zip(ad_out, ad_ref):
            model = astromodels.dict_to_chebyshev(
                dict(zip(ext_out.WAVECAL["name"], ext_out.WAVECAL["coefficients"]))
            )

            ref_model = astromodels.dict_to_chebyshev(
                dict(zip(ext_ref.WAVECAL["name"], ext_ref.WAVECAL["coefficients"]))
            )

            x = np.arange(ext_out.shape[1])
            y = model(x)
            ref_y = ref_model(x)

            np.testing.assert_allclose(y, ref_y, rtol=1)

        del ad_out, ad_ref

    @staticmethod
    def test_reduced_arcs_are_similar(config):
        """
        Compares the output and the reference arcs to make sure their data are
        similar before running further tests.
        """
        output = os.path.join(config.output_dir, config.filename)
        reference = os.path.join(config.ref_dir, config.filename)

        if not os.path.exists(output):
            pytest.skip("Output file not found: {}".format(output))

        if not os.path.exists(reference):
            pytest.fail("Reference file not found: {}".format(reference))

        ad_out = config.ad
        ad_ref = astrodata.open(reference)

        # Test reduced arcs are similar
        for ext_out, ext_ref in zip(ad_out, ad_ref):
            np.testing.assert_allclose(ext_out.data, ext_ref.data, rtol=1)

        del ad_out, ad_ref

    @staticmethod
    def test_determine_distortion(config):
        """
        Corrects distortion on both output and reference files using the
        distortion model stored in themselves. Previous tests assures that
        these data are similar and that distortion correct is applied the same
        way. Now, this one tests the model itself.
        """
        if not os.path.exists(config.output_file):
            pytest.skip("Processed arc file not found: {}".format(config.output_file))

        ad_out = config.ad

        p = primitives_gmos_spect.GMOSSpect([])

        # Using with id_only=True isolates this test from the wavelength
        # calibration tests
        ad_out = p.determineDistortion(
            adinputs=[ad_out], id_only=False, suffix="_distortionDetermined"
        )[0]
        ad_out.write(overwrite=True)

        os.rename(ad_out.filename, os.path.join(config.output_dir, ad_out.filename))

        # Reads the reference file ---
        reference = os.path.join(config.ref_dir, ad_out.filename)

        if not os.path.exists(reference):
            pytest.fail("Reference file not found: {}".format(reference))

        ad_ref = astrodata.open(reference)

        # Compare them ---
        for ext_out, ext_ref in zip(ad_out, ad_ref):

            coeffs_out = np.ma.masked_invalid(ext_out.FITCOORD["coefficients"])
            coeffs_ref = np.ma.masked_invalid(ext_ref.FITCOORD["coefficients"])

            np.testing.assert_allclose(coeffs_out, coeffs_ref, atol=0.1)

        del ad_out, ad_ref, p

    @staticmethod
    def test_distortion_correct(config):
        """
        Corrects distortion on both output and reference files using the
        distortion model stored in themselves. Previous tests assures that
        these data are similar and that distortion correct is applied the same
        way. Now, this one tests the model itself.
        """
        # Recover name of the distortion corrected arc files ---
        basename = os.path.basename(config.filename)
        filename, extension = os.path.splitext(basename)
        filename = filename.split("_")[0] + "_distortionDetermined" + extension

        output = os.path.join(config.output_dir, filename)
        reference = os.path.join(config.ref_dir, filename)

        if not os.path.exists(output):
            pytest.fail("Processed arc file not found: {}".format(output))

        if not os.path.exists(reference):
            pytest.fail("Processed reference file not found: {}".format(reference))

        p = primitives_gmos_spect.GMOSSpect([])

        ad_out = astrodata.open(output)
        ad_out_corrected_with_out = p.distortionCorrect([ad_out], arc=output)[0]
        ad_out_corrected_with_ref = p.distortionCorrect([ad_out], arc=reference)[0]

        ad_out_corrected_with_out.write(
            overwrite=True,
            filename=os.path.join(
                config.output_dir, ad_out_corrected_with_out.filename
            ),
        )

        for ext_out, ext_ref in zip(
            ad_out_corrected_with_out, ad_out_corrected_with_ref
        ):
            np.testing.assert_allclose(ext_out.data, ext_ref.data)

    @staticmethod
    @pytest.mark.skip(reason="not fully implemented yet")
    def test_distortionCorrect_works_when_using_FullROI_on_CentralROI(path_to_outputs):
        """
        Test if a model obtained in a Full Frame ROI can be applied to a Central
        Spectra ROI.
        """

        class Config:
            def __init__(self, name):
                sub_path, basename = os.path.split(name)

                basename, ext = os.path.splitext(basename)
                basename = basename + "_distortionDetermined.fits"

                self._filename = basename
                self.output_dir = os.path.join(path_to_outputs, sub_path)
                self.full_name = os.path.join(self.output_dir, self._filename)

            @property
            def filename(self):
                return self._filename

            @filename.setter
            def filename(self, name):
                self._filename = name
                self.full_name = os.path.join(self.output_dir, self._filename)

        # B600:0.500 HAM, ROI="Central Spectrum" ---
        cs = Config("process_arcs/GMOS/N20171016S0010.fits")
        cs.ad = astrodata.open(os.path.join(cs.output_dir, cs.filename))

        # B600:0.500 HAM, ROI="Full Frame" ---
        ff = Config("process_arcs/GMOS/N20171016S0127.fits")

        # Apply full frame roi to central-spect roi
        p = primitives_gmos_spect.GMOSSpect([])
        cs.ad = p.distortionCorrect(adinputs=[cs.ad], arc=ff.full_name)[0]
        cs.filename = cs.filename.replace(".fits", "_fromFullFrame.fits")
        cs.ad.write(filename=cs.full_name, overwrite=True)

    @staticmethod
    @pytest.mark.skip(reason="not fully implemented yet")
    def test_distortion_correction_is_applied_the_same_way(config):
        """
        Applies the same distortion correction model to both output and reference
        arcs and compares the results.
        """
        # Process the output file ---
        basename, ext = os.path.splitext(config.filename)
        basename, _ = basename.split("_")[0], basename.split("_")[1:]

        arc_basename = "{:s}_{:s}{:s}".format(basename, "distortionDetermined", ext)

        arc_name = os.path.join(config.output_dir, arc_basename)

        if not os.path.exists(arc_name):
            pytest.skip("Arc file not found: {}".format(arc_name))

        ad_out = config.ad

        p = primitives_gmos_spect.GMOSSpect([])
        ad_out = p.distortionCorrect(adinputs=[ad_out], arc=arc_name)[0]

        filename = ad_out.filename
        ad_out = p.determineDistortion(adinputs=[ad_out])[0]

        for ext in ad_out:
            assert hasattr(ext, "FITCOORD")

        ad_out.write(filename=filename, overwrite=True)

        os.rename(filename, os.path.join(config.output_dir, filename))

        # assert False

        # # Reads the reference file ---
        # reference = os.path.join(config.ref_dir, ad_out.filename)
        #
        # if not os.path.exists(reference):
        #     pytest.fail('Reference file not found: {}'.format(reference))
        #
        # ad_ref = astrodata.open(reference)

        # Evaluate them ---
        for ext in ad_out:
            coefficients = dict(zip(ext.FITCOORD["name"], ext.FITCOORD["coefficients"]))

            # print(coefficients)

        # for ext_out, ext_ref in zip(ad_out, ad_ref):
        #
        #     model_out = astromodels.dict_to_chebyshev(
        #         dict(zip(
        #             ext_out.FITCOORD['name'], ext_out.FITCOORD['coefficients'])))
        #
        #     # model_ref = astromodels.dict_to_chebyshev(
        #     #     dict(zip(
        #     #         ext_ref.FITCOORD['name'], ext_ref.FITCOORD['coefficients'])))
        #
        # del ad_out, ad_ref, p


def test_full_frame_distortion_works_on_smaller_region(path_to_inputs):
    """
    Takes a full-frame arc and self-distortion-corrects it. It then fakes
    subregions of this and corrects those using the full-frame distortion to
    confirm that the result is the same as the appropriate region of the
    distortion-corrected full-frame image. There's no need to do this more
    than once for a given binning, so we loop within the function, keeping
    track of binnings we've already processed.
    """
    NSUB = 4  # we're going to take combos of horizontal quadrants
    completed_binnings = []

    for filename in dataset_file_list:
        ad = astrodata.open(os.path.join(path_to_inputs, filename))
        xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
        if (ad.detector_roi_setting() != "Full Fame" or
                (xbin, ybin) in completed_binnings):
            continue
        p = GMOSLongslit([ad])
        p.viewer.viewer_name = None
        p.prepare()
        p.addDQ(static_bpm=None)
        p.overscanCorrect()
        p.ADUToElectrons()
        p.mosaicDetectors(outstream='mosaic')
        # Speed things up a bit with a larger step
        p.determineDistortion(stream='mosaic', step=48 // ybin)
        ad_out = p.distortionCorrect([deepcopy(ad)], arc=p.streams['mosaic'][0],
                                     order=1)[0]

        for start in range(NSUB):
            for end in range(start+1, NSUB+1):
                ad_copy = deepcopy(ad)
                y1b = start * ad[0].shape[0] // NSUB
                y2b = end * ad[0].shape[0] // NSUB
                y1, y2 = y1b * ybin, y2b * ybin  # unbinned pixels

                # Fake the section header keywords and set the SCI and DQ
                # to the appropriate sub-region
                for ext in ad_copy:
                    arrsec = ext.array_section()
                    detsec = ext.detector_section()
                    ext.hdr['CCDSEC'] = '[{}:{},{}:{}]'.format(arrsec.x1+1,
                                                               arrsec.x2, y1+1, y2)
                    ext.hdr['DETSEC'] = '[{}:{},{}:{}]'.format(detsec.x1+1,
                                                               detsec.x2, y1+1, y2)
                    ext.data = ext.data[y1b:y2b]
                    ext.mask = ext.mask[y1b:y2b]
                    ext.hdr['DATASEC'] = '[1:{},1:{}]'.format(ext.shape[1], y2b-y1b)
                ad2 = p.distortionCorrect([ad_copy], arc=p.streams['mosaic'][0],
                                          order=1)[0]

                # It's GMOS LS so the offset between this AD and the full-frame
                # will be the same as the DETSEC offset, but the width may be
                # smaller so we need to shuffle the smaller image within the
                # larger one to look for a match
                ny, nx = ad2[0].shape
                xsizediff = ad_out[0].shape[1] - nx
                ok = False
                for xoffset in range(xsizediff+1):
                    # Confirm that all unmasked pixels are similar
                    diff = (np.ma.masked_array(ad2[0].data, mask=ad2[0].mask) -
                            ad_out[0].data[y1b:y1b+ny, xoffset:xoffset+nx])
                    if np.logical_and(abs(diff) > 0.01, ~diff.mask).sum() == 0:
                        ok = True
                        break
                assert ok, "Problem with {} {}:{}".format(ad.filename, start, end)

        completed_binnings.append((xbin, ybin))