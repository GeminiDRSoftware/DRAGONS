#!/usr/bin/env python
"""
Tests related to GMOS Long-slit Spectroscopy Arc primitives.
"""
import glob
import os
import tarfile
import warnings

import numpy as np

# noinspection PyPackageRequirements
import pytest

# noinspection PyPackageRequirements
from astropy.modeling import models
from matplotlib import colors
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage

import astrodata

# noinspection PyUnresolvedReferences
import gemini_instruments
import geminidr
from geminidr.gmos import primitives_gmos_spect
from gempy.library import astromodels, transform
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
    "process_arcs/GMOS/N20130606S0291.fits",  # R150:0.550 E2V
    "process_arcs/GMOS/N20130112S0574.fits",  # R150:0.700 E2V
    # 'process_arcs/GMOS/N20130809S0337.fits',  # R150:0.700 E2V - todo: RMS > 0.5 (RMS = 0.61) | `gswavelength` cannot find solution either.
    "process_arcs/GMOS/N20140408S0218.fits",  # R150:0.700 E2V
    # 'process_arcs/GMOS/N20180119S0232.fits',  # R150:0.520 HAM - todo: RMS > 0.5 (RMS = 0.85) | Unstable Wav. Sol. | `gswavelength` cannot find solution either.
    # 'process_arcs/GMOS/N20180516S0214.fits',  # R150:0.610 HAM ROI="Central Spectrum", bin=2x2 - todo: fails test_distortion_model_is_the_same
    "process_arcs/GMOS/N20171007S0439.fits",  # R150:0.650 HAM
    "process_arcs/GMOS/N20171007S0441.fits",  # R150:0.650 HAM
    "process_arcs/GMOS/N20101212S0213.fits",  # R400:0.550 EEV
    "process_arcs/GMOS/N20100202S0214.fits",  # R400:0.700 EEV
    "process_arcs/GMOS/N20130106S0194.fits",  # R400:0.500 E2V
    "process_arcs/GMOS/N20130422S0217.fits",  # R400:0.700 E2V
    "process_arcs/GMOS/N20170108S0210.fits",  # R400:0.660 HAM
    "process_arcs/GMOS/N20171113S0135.fits",  # R400:0.750 HAM
    "process_arcs/GMOS/N20100427S1276.fits",  # R600:0.675 EEV
    "process_arcs/GMOS/N20180120S0417.fits",  # R600:0.860 HAM
    "process_arcs/GMOS/N20100212S0143.fits",  # R831:0.450 EEV
    "process_arcs/GMOS/N20100720S0247.fits",  # R831:0.850 EEV
    "process_arcs/GMOS/N20130808S0490.fits",  # R831:0.571 E2V
    "process_arcs/GMOS/N20130830S0291.fits",  # R831:0.845 E2V
    "process_arcs/GMOS/N20170910S0009.fits",  # R831:0.653 HAM
    "process_arcs/GMOS/N20170509S0682.fits",  # R831:0.750 HAM
    # 'process_arcs/GMOS/N20181114S0512.fits',  # R831:0.865 HAM - todo: RMS > 0.5 (RMS = 0.646) | `gswavelength` cannot find solution either.
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
    # 'process_arcs/GMOS/S20170116S0189.fits',  # B1200:0.440 HAM - todo: very weird non-linear plot | non-linear plot using `gswavelength` seems fine.
    "process_arcs/GMOS/S20170103S0149.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/S20170730S0155.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/S20171219S0117.fits",  # B1200:0.440 HAM
    "process_arcs/GMOS/S20170908S0189.fits",  # B1200:0.550 HAM
    "process_arcs/GMOS/S20131230S0153.fits",  # R150:0.550 EEV
    "process_arcs/GMOS/S20130801S0140.fits",  # R150:0.700 EEV
    "process_arcs/GMOS/S20170430S0060.fits",  # R150:0.717 HAM
    # "process_arcs/GMOS/S20170430S0063.fits",  # R150:0.727 HAM - todo: TypeError: Lengths of the first three arguments (x,y,w) must be equal
    "process_arcs/GMOS/S20171102S0051.fits",  # R150:0.950 HAM
    "process_arcs/GMOS/S20130114S0100.fits",  # R400:0.620 EEV
    "process_arcs/GMOS/S20130217S0073.fits",  # R400:0.800 EEV
    "process_arcs/GMOS/S20170108S0046.fits",  # R400:0.550 HAM
    "process_arcs/GMOS/S20170129S0125.fits",  # R400:0.685 HAM
    "process_arcs/GMOS/S20170703S0199.fits",  # R400:0.800 HAM
    "process_arcs/GMOS/S20170718S0420.fits",  # R400:0.910 HAM
    # 'process_arcs/GMOS/S20100306S0460.fits',  # R600:0.675 EEV - todo: breaks p.determineWavelengthSolution
    # 'process_arcs/GMOS/S20101218S0139.fits',  # R600:0.675 EEV - todo: breaks p.determineWavelengthSolution
    "process_arcs/GMOS/S20110306S0294.fits",  # R600:0.675 EEV
    'process_arcs/GMOS/S20110720S0236.fits',  # R600:0.675 EEV - todo: RMS > 0.5 (RMS = 0.508)
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

    try:
        old_mask = os.umask(000)
        os.chmod(log_file, 0o775)
        os.umask(old_mask)
    except PermissionError:
        pass

    # Setup configuration for tests ---
    class ConfigTest:
        """
        Config class created for each dataset file. It is created from within
        this a fixture so it can inherit the `path_to_*` fixtures as well.
        """

        def __init__(self, filename):
            input_file = os.path.join(path_to_inputs, filename)
            dataset_sub_dir = os.path.dirname(filename)

            reference_folder = os.path.join(path_to_refs, dataset_sub_dir)
            output_folder = os.path.join(path_to_outputs, dataset_sub_dir)

            old_mask = os.umask(000)
            os.makedirs(output_folder, exist_ok=True, mode=0o775)
            os.umask(old_mask)

            output_file = os.path.join(path_to_outputs, filename)
            output_file, _ = os.path.splitext(output_file)
            output_file = output_file + "_arc.fits"
            output_file = os.path.join(output_folder, output_file)

            r = self.reduce(input_file)

            ad = r.writeOutputs(outfilename=output_file, overwrite=True)[0]
            os.chmod(output_file, mode=0o775)

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

        @staticmethod
        def generate_fake_data(ad):

            nrows, ncols = ad.shape
            dispaxis = ad.dispersion_axis() - 1

            n_lines = 100
            np.random.seed(0)

            data = np.zeros((nrows, ncols))
            line_positions = np.random.random_integers(0, ncols, size=n_lines)
            line_intensities = 100 * np.random.random_sample(n_lines)

            if dispaxis == 0:
                data[:, line_positions] = line_intensities
                data = ndimage.gaussian_filter(data, [5, 1])
            else:
                data[line_positions, :] = line_intensities
                data = ndimage.gaussian_filter(data, [1, 5])

            data = data + (np.random.random_sample(data.shape) - 0.5) * 10

            return data

    c = ConfigTest(request.param)

    yield c

    p = PlotGmosSpectLongslitArcs(c.ad, c.output_dir)
    p.wavelength_calibration_plots()
    # p.distortion_diagnosis_plots()

    del c, p


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
    def test_distortion_model_is_the_same(config):
        """
        Corrects distortion on both output and reference files using the
        distortion model stored in themselves. Previous tests assures that
        these data are similar and that distortion correct is applied the same
        way. Now, this one tests the model itself.
        """
        if not os.path.exists(config.output_file):
            pytest.skip("Output file not found: {}".format(config.output_file))

        ad_out = config.ad

        p = primitives_gmos_spect.GMOSSpect([])

        # Using with id_only=True isolates this test from the wavelength
        # calibration tests
        ad_out = p.determineDistortion(
            adinputs=[ad_out], id_only=False, suffix="_distortionDetermined"
        )[0]
        ad_out.write(overwrite=True)

        os.rename(ad_out.filename, os.path.join(config.output_dir, ad_out.filename))

        old_mask = os.umask(000)
        os.chmod(os.path.join(config.output_dir, ad_out.filename), mode=0o775)
        os.umask(old_mask)

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

    # @staticmethod
    # def test_distortion_model_is_the_same(config):
    #     """
    #     Corrects distortion on both output and reference files using the
    #     distortion model stored in themselves. Previous tests assures that
    #     these data are similar and that distortion correct is applied the same
    #     way. Now, this one tests the model itself.
    #     """
    #     # Process the output file ---
    #     output = os.path.join(config.output_dir, config.filename)
    #
    #     if not os.path.exists(output):
    #         pytest.skip('Output file not found: {}'.format(output))
    #
    #     ad_out = config.ad
    #
    #     p = primitives_gmos_spect.GMOSSpect([])
    #
    #     # Using with id_only=True isolates this test from the wavelength
    #     # calibration tests
    #     ad_out = p.determineDistortion(adinputs=[ad_out], id_only=True)[0]
    #
    #     ad_out.write(overwrite=True)
    #     os.rename(ad_out.filename, os.path.join(config.output_dir, ad_out.filename))
    #
    #     old_mask = os.umask(000)
    #     os.chmod(os.path.join(config.output_dir, ad_out.filename), mode=0o775)
    #     os.umask(old_mask)
    #
    #     # Reads the reference file ---
    #     reference = os.path.join(config.ref_dir, ad_out.filename)
    #
    #     if not os.path.exists(reference):
    #         pytest.fail('Reference file not found: {}'.format(reference))
    #
    #     ad_ref = astrodata.open(reference)
    #
    #     # Helper function to build model ---
    #     def build_model(ext):
    #
    #         dispaxis = ext.dispersion_axis() - 1
    #
    #         m = models.Identity(2)
    #         m_inv = astromodels.dict_to_chebyshev(
    #             dict(zip(
    #                 ext.FITCOORD["name"], ext.FITCOORD["coefficients"])))
    #
    #         # See https://docs.astropy.org/en/stable/modeling/compound-models.html#advanced-mappings
    #         if dispaxis == 0:
    #             m.inverse = models.Mapping((0, 1, 1)) | (m_inv & models.Identity(1))
    #         else:
    #             m.inverse = models.Mapping((0, 0, 1)) | (models.Identity(1) & m_inv)
    #
    #         return m
    #
    #     # Compare them ---
    #     for ext_out, ext_ref in zip(ad_out, ad_ref):
    #         data = config.generate_fake_data(ext_out)
    #
    #         model_out = build_model(ext_out)
    #         model_ext = build_model(ext_ref)
    #
    #         transform_out = transform.Transform(model_out)
    #         transform_ref = transform.Transform(model_ext)
    #
    #         data_out = transform_out.apply(data, output_shape=ext_out.shape)
    #         data_ref = transform_ref.apply(data, output_shape=ext_ref.shape)
    #
    #         data_out = np.ma.masked_invalid(data_out)
    #         data_ref = np.ma.masked_invalid(data_ref)
    #
    #         fig, ax = plt.subplots(num="Distortion Comparison: {}".format(config.filename))
    #
    #         im = ax.imshow(data_ref - data_out)
    #
    #         ax.set_xlabel('X [px]')
    #         ax.set_ylabel('Y [px]')
    #         ax.set_title('Difference between output and reference \n {}'.format(
    #             os.path.basename(config.filename)))
    #
    #         divider = make_axes_locatable(ax)
    #         cax = divider.append_axes('right', size='5%', pad=0.05)
    #
    #         cbar = fig.colorbar(im, extend='max', cax=cax, orientation='vertical')
    #         cbar.set_label('Distortion [px]')
    #
    #         fig.savefig(output.replace('.fits', '_distortionDifference.png'))
    #
    #         np.testing.assert_allclose(data_out, data_ref)
    #         np.testing.assert_allclose(ext_out.FITCOORD['coefficients'], ext_ref.FITCOORD['coefficients'])
    #
    #     del ad_out, ad_ref, p

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

        old_mask = os.umask(000)
        os.chmod(os.path.join(cs.full_name), mode=0o775)
        os.umask(old_mask)

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

        old_mask = os.umask(000)
        os.chmod(os.path.join(config.output_dir, filename), mode=0o775)
        os.umask(old_mask)

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


class PlotGmosSpectLongslitArcs:
    """
    Plot solutions for extensions inside `ad` that have a `WAVECAL` and save the
    results inside the `output_folder`

    Parameters
    ----------
    ad : AstroData
        Reduced arc with a wavelength solution.
    output_folder : str
        Path to where the plots will be saved.
    """

    def __init__(self, ad, output_folder):

        filename = ad.filename
        self.ad = ad
        self.name, _ = os.path.splitext(filename)
        self.grating = ad.disperser(pretty=True)
        self.bin_x = ad.detector_x_bin()
        self.bin_y = ad.detector_y_bin()
        self.central_wavelength = ad.central_wavelength() * 1e9  # in nanometers
        self.output_folder = output_folder

        self.package_dir = os.path.dirname(primitives_gmos_spect.__file__)
        self.arc_table = os.path.join(self.package_dir, "lookups", "CuAr_GMOS.dat")
        self.arc_lines = np.loadtxt(self.arc_table, usecols=[0]) / 10.0

    def create_artifact_from_plots(self):
        """
        Created a .tar.gz file using the plots generated here so Jenkins can deliver
        it as an artifact.
        """
        # Runs only from inside Jenkins
        if "BUILD_ID" in os.environ:

            branch_name = os.environ["BRANCH_NAME"].replace("/", "_")
            build_number = int(os.environ["BUILD_NUMBER"])

            tar_name = os.path.join(
                self.output_folder,
                "plots_{:s}_b{:03d}.tar.gz".format(branch_name, build_number),
            )

            with tarfile.open(tar_name, "w:gz") as tar:
                for _file in glob.glob(os.path.join(self.output_folder, "*.png")):
                    tar.add(name=_file, arcname=os.path.basename(_file))

            target_dir = "./plots/"
            target_file = os.path.join(target_dir, os.path.basename(tar_name))

            os.makedirs(target_dir, exist_ok=True)
            os.rename(tar_name, target_file)

            try:
                os.chmod(target_file, 0o775)
            except PermissionError:
                warnings.warn(
                    "Failed to update permissions for file: {}".format(target_file)
                )

    def distortion_diagnosis_plots(self):
        """
        Makes the Diagnosis Plots for `determineDistortion` and
        `distortionCorrect` for each extension inside the reduced arc.
        """
        full_name = os.path.join(self.output_folder, self.name + ".fits")

        file_list = [
            full_name,
            full_name.replace("distortionDetermined", "distortionCorrected"),
        ]

        for filename in file_list:

            ad = astrodata.open(filename)

            for e_num, e in enumerate(ad):

                if not hasattr(e, "FITCOORD"):
                    continue

                distortion_model = astromodels.dict_to_chebyshev(
                    dict(zip(e.FITCOORD["name"], e.FITCOORD["coefficients"]))
                )

                model_dict = dict(zip(e.FITCOORD["name"], e.FITCOORD["coefficients"]))

                self.plot_distortion_map(ad.filename, e_num, e.shape, distortion_model)

                if "distortionCorrected" in filename:
                    self.plot_distortion_residuals(
                        ad.filename, e_num, e.shape, distortion_model
                    )

    def plot_distortion_map(self, fname, ext_num, shape, model):
        """
        Plots the distortion map determined for a given file.

        Parameters
        ----------
        fname : str
            File name
        ext_num : int
            Extension number.
        shape : tuple
            Data shape.
        model : Chebyshev1D
            Model that represents the wavelength solution.
        """
        n_hlines = 50
        n_vlines = 50
        n_rows, n_cols = shape

        x = np.linspace(0, n_cols, n_vlines, dtype=int)
        y = np.linspace(0, n_rows, n_hlines, dtype=int)

        X, Y = np.meshgrid(x, y)

        U = X - model(X, Y)
        V = np.zeros_like(U)

        fig, ax = plt.subplots(num="Distortion Map {}".format(fname))

        vmin = U.min() if U.min() < 0 else -0.1 * U.ptp()
        vmax = U.max() if U.max() > 0 else +0.1 * U.ptp()
        vcen = 0

        Q = ax.quiver(
            X,
            Y,
            U,
            V,
            U,
            cmap="coolwarm",
            norm=colors.DivergingNorm(vcenter=vcen, vmin=vmin, vmax=vmax),
        )

        ax.set_xlabel("X [px]")
        ax.set_ylabel("Y [px]")
        ax.set_title(
            "Distortion Map\n{} - Bin {:d}x{:d}".format(fname, self.bin_x, self.bin_y)
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = fig.colorbar(Q, extend="max", cax=cax, orientation="vertical")
        cbar.set_label("Distortion [px]")

        fig.tight_layout()

        fig_name = os.path.join(
            self.output_folder,
            "{:s}_{:d}_{:s}_{:.0f}_dmap.png".format(
                fname, ext_num, self.grating, self.central_wavelength
            ),
        )

        fig.savefig(fig_name)

        try:
            old_mask = os.umask(000)
            os.chmod(fig_name, 0o775)
            os.umask(old_mask)
        except PermissionError:
            pass

    def plot_distortion_residuals(self, fname, ext_num, shape, model):
        """
        Plots the distortion residuals calculated on an arc dataset that passed
        through `distortionCorrect`. The residuals are calculated based on an
        artificial mesh and using a model obtained from `determinedDistortion`
        applied to the distortion corrected file.

        Parameters
        ----------
        fname : str
            File name
        ext_num : int
            Number of the extension.
        shape : tuple
            Data shape
        model : distortion model calculated on a distortion corrected file.
        """
        n_hlines = 25
        n_vlines = 25
        n_rows, n_cols = shape

        x = np.linspace(0, n_cols, n_vlines, dtype=int)
        y = np.linspace(0, n_rows, n_hlines, dtype=int)

        X, Y = np.meshgrid(x, y)

        U = X - model(X, Y)

        width = 0.75 * np.diff(x).mean()
        _min, _med, _max = np.percentile(U, [0, 50, 100], axis=0)

        fig, ax = plt.subplots(
            num="Corrected Distortion Residual Stats {:s}".format(fname)
        )

        ax.scatter(x, _min, marker="^", s=4, c="C0")
        ax.scatter(x, _max, marker="v", s=4, c="C0")

        parts = ax.violinplot(
            U, positions=x, showmeans=True, showextrema=True, widths=width
        )

        parts["cmins"].set_linewidth(0)
        parts["cmaxes"].set_linewidth(0)
        parts["cbars"].set_linewidth(0.75)
        parts["cmeans"].set_linewidth(0.75)

        ax.grid("k-", alpha=0.25)
        ax.set_xlabel("X [px]")
        ax.set_ylabel("Position Residual [px]")
        ax.set_title("Corrected Distortion Residual Stats\n{}".format(fname))

        fig.tight_layout()

        fig_name = os.path.join(
            self.output_folder,
            "{:s}_{:d}_{:s}_{:.0f}_dres.png".format(
                fname, ext_num, self.grating, self.central_wavelength
            ),
        )

        fig.savefig(fig_name)

        try:
            old_mask = os.umask(000)
            os.chmod(fig_name, 0o775)
            os.umask(old_mask)
        except PermissionError:
            pass

    def plot_lines(self, ext_num, data, peaks, model):
        """
        Plots and saves the wavelength calibration model residuals for diagnosis.

        Parameters
        ----------
        ext_num : int
            Extension number.
        data : ndarray
            1D numpy masked array that represents the data.
        peaks : ndarray
            1D array with 1-indexed peaks positon.
        model : Chebyshev1D
            Model that represents the wavelength solution.
        """
        fig, ax = plt.subplots(
            dpi=300,
            num="{:s}_{:d}_{:s}_{:.0f}".format(
                self.name, ext_num, self.grating, self.central_wavelength
            ),
        )

        w = model(np.arange(data.size))

        arcs = [ax.vlines(line, 0, 1, color="k", alpha=0.25) for line in self.arc_lines]
        wavs = [
            ax.vlines(peak, 0, 1, color="r", ls="--", alpha=0.25)
            for peak in model(peaks)
        ]
        plot, = ax.plot(w, data, "k-", lw=0.75)

        ax.legend(
            (plot, arcs[0], wavs[0]),
            ("Normalized Data", "Reference Lines", "Matched Lines"),
        )

        x0, x1 = model([0, data.size])

        ax.grid(alpha=0.1)
        ax.set_xlim(x0, x1)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Normalized intensity")
        ax.set_title(
            "Wavelength Calibrated Spectrum for\n"
            "{:s} obtained with {:s} at {:.0f}".format(
                self.name, self.grating, self.central_wavelength
            )
        )

        if x0 > x1:
            ax.invert_xaxis()

        fig_name = os.path.join(
            self.output_folder,
            "{:s}_{:d}_{:s}_{:.0f}.png".format(
                self.name, ext_num, self.grating, self.central_wavelength
            ),
        )

        fig.savefig(fig_name)

        try:
            os.chmod(fig_name, 0o775)
        except PermissionError:
            warnings.warn("Failed to update permissions for file: {}".format(fig_name))

        del fig, ax

    def plot_non_linear_components(self, ext_num, peaks, wavelengths, model):
        """
        Plots the non-linear residuals.

        Parameters
        ----------
        ext_num : int
            Extension number.
        peaks : ndarray
            1D array with 1-indexed peaks positon.
        wavelengths : ndarray
            1D array with wavelengths matching peaks.
        model : Chebyshev1D
            Model that represents the wavelength solution.
        """
        fig, ax = plt.subplots(
            dpi=300,
            num="{:s}_{:d}_{:s}_{:.0f}_non_linear_comps".format(
                self.name, ext_num, self.grating, self.central_wavelength
            ),
        )

        non_linear_model = model.copy()
        _ = [setattr(non_linear_model, "c{}".format(k), 0) for k in [0, 1]]
        residuals = wavelengths - model(peaks)

        p = np.linspace(min(peaks), max(peaks), 1000)
        ax.plot(model(p), non_linear_model(p), "C0-", label="Generic Representation")
        ax.plot(
            model(peaks),
            non_linear_model(peaks) + residuals,
            "ko",
            label="Non linear components and residuals",
        )
        ax.legend()

        ax.grid(alpha=0.25)
        ax.set_xlabel("Wavelength [nm]")

        ax.set_title(
            "Non-linear components for\n"
            "{:s} obtained with {:s} at {:.0f}".format(
                self.name, self.grating, self.central_wavelength
            )
        )

        fig_name = os.path.join(
            self.output_folder,
            "{:s}_{:d}_{:s}_{:.0f}_non_linear_comps.png".format(
                self.name, ext_num, self.grating, self.central_wavelength
            ),
        )

        fig.savefig(fig_name)

        try:
            os.chmod(fig_name, 0o775)
        except PermissionError:
            warnings.warn("Failed to update permissions for file: {}".format(fig_name))

        del fig, ax

    def plot_residuals(self, ext_num, peaks, wavelengths, model):
        """
        Plots the matched wavelengths versus the residuum  between them and their
        correspondent peaks applied to the fitted model.

        Parameters
        ----------
        ext_num : int
            Extension number.
        peaks : ndarray
            1D array with 1-indexed peaks positon.
        wavelengths : ndarray
            1D array with wavelengths matching peaks.
        model : Chebyshev1D
            Model that represents the wavelength solution.
        """
        fig, ax = plt.subplots(
            dpi=300,
            num="{:s}_{:d}_{:s}_{:.0f}_residuals".format(
                self.name, ext_num, self.grating, self.central_wavelength
            ),
        )

        ax.plot(wavelengths, wavelengths - model(peaks), "ko")

        ax.grid(alpha=0.25)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Residuum [nm]")
        ax.set_title(
            "Wavelength Calibrated Residuum for\n"
            "{:s} obtained with {:s} at {:.0f}".format(
                self.name, self.grating, self.central_wavelength
            )
        )

        fig_name = os.path.join(
            self.output_folder,
            "{:s}_{:d}_{:s}_{:.0f}_residuals.png".format(
                self.name, ext_num, self.grating, self.central_wavelength
            ),
        )

        fig.savefig(fig_name)

        try:
            os.chmod(fig_name, 0o775)
        except PermissionError:
            warnings.warn("Failed to update permissions for file: {}".format(fig_name))

        del fig, ax

    def wavelength_calibration_plots(self):
        """
        Makes the Wavelength Calibration Diagnosis Plots for each extension
        inside the reduced arc.
        """

        for ext_num, ext in enumerate(self.ad):

            if not hasattr(ext, "WAVECAL"):
                continue

            peaks = ext.WAVECAL["peaks"] - 1  # ToDo: Refactor peaks to be 0-indexed
            wavelengths = ext.WAVECAL["wavelengths"]

            wavecal_model = astromodels.dict_to_chebyshev(
                dict(zip(ext.WAVECAL["name"], ext.WAVECAL["coefficients"]))
            )

            mask = np.round(np.average(ext.mask, axis=0)).astype(int)
            data = np.ma.masked_where(mask > 0, np.average(ext.data, axis=0))
            data = (data - data.min()) / data.ptp()

            self.plot_lines(ext_num, data, peaks, wavecal_model)
            self.plot_non_linear_components(ext_num, peaks, wavelengths, wavecal_model)
            self.plot_residuals(ext_num, peaks, wavelengths, wavecal_model)
            self.create_artifact_from_plots()
