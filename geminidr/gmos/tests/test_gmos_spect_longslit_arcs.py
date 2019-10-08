#!/usr/bin/env python
"""
Tests related to GMOS Long-slit Spectroscopy arc processing.
"""
import glob
import os
import tarfile
import warnings

import numpy as np
# noinspection PyPackageRequirements
import pytest
# noinspection PyPackageRequirements
from matplotlib import pyplot as plt

import astrodata
# noinspection PyUnresolvedReferences
import gemini_instruments
import geminidr
from geminidr.gmos import primitives_gmos_spect
from gempy.library import astromodels
from gempy.utils import logutils

dataset_file_list = [
    # Process Arcs: GMOS-N ---
    'process_arcs/GMOS/N20100115S0346.fits',  # B600:0.500 EEV
    'process_arcs/GMOS/N20130112S0390.fits',  # B600:0.500 E2V
    'process_arcs/GMOS/N20170609S0173.fits',  # B600:0.500 HAM
    'process_arcs/GMOS/N20100307S0236.fits',  # B1200:0.445 EEV
    'process_arcs/GMOS/N20130628S0290.fits',  # B1200:0.420 E2V
    'process_arcs/GMOS/N20170904S0078.fits',  # B1200:0.440 HAM
    'process_arcs/GMOS/N20170627S0116.fits',  # B1200:0.520 HAM
    'process_arcs/GMOS/N20100830S0594.fits',  # R150:0.500 EEV
    'process_arcs/GMOS/N20100702S0321.fits',  # R150:0.700 EEV
    'process_arcs/GMOS/N20130606S0291.fits',  # R150:0.550 E2V
    # # 'process_arcs/GMOS/N20130809S0337.fits',  # R150:0.700 E2V - todo: RMS > 0.5 (RMS = 0.61)
    # # 'process_arcs/GMOS/N20180119S0232.fits',  # R150:0.520 HAM - todo: RMS > 0.5 (RMS = 0.85) | Unstable Wav. Sol.
    # 'process_arcs/GMOS/N20180516S0214.fits',  # R150:0.610 HAM
    # 'process_arcs/GMOS/N20101212S0213.fits',  # R400:0.550 EEV
    # 'process_arcs/GMOS/N20100202S0214.fits',  # R400:0.700 EEV
    # 'process_arcs/GMOS/N20130106S0194.fits',  # R400:0.500 E2V
    # 'process_arcs/GMOS/N20130422S0217.fits',  # R400:0.700 E2V
    # 'process_arcs/GMOS/N20170108S0210.fits',  # R400:0.660 HAM
    # 'process_arcs/GMOS/N20171113S0135.fits',  # R400:0.750 HAM
    # 'process_arcs/GMOS/N20100427S1276.fits',  # R600:0.675 EEV
    # 'process_arcs/GMOS/N20180120S0417.fits',  # R600:0.860 HAM
    # 'process_arcs/GMOS/N20100212S0143.fits',  # R831:0.450 EEV
    # 'process_arcs/GMOS/N20100720S0247.fits',  # R831:0.850 EEV
    # 'process_arcs/GMOS/N20130808S0490.fits',  # R831:0.571 E2V
    # 'process_arcs/GMOS/N20130830S0291.fits',  # R831:0.845 E2V
    # 'process_arcs/GMOS/N20170910S0009.fits',  # R831:0.653 HAM
    # 'process_arcs/GMOS/N20170509S0682.fits',  # R831:0.750 HAM
    # # 'process_arcs/GMOS/N20181114S0512.fits',  # R831:0.865 HAM - todo: RMS > 0.5 (RMS = 0.646)
    #
    # # Process Arcs: GMOS-S ---
    # # 'process_arcs/GMOS/S20130218S0126.fits',  # B600:0.500 EEV - todo: breaks p.determineWavelengthSolution()
    # 'process_arcs/GMOS/S20140504S0008.fits',  # B600:0.500 EEV
    # 'process_arcs/GMOS/S20170103S0152.fits',  # B600:0.600 HAM
    # 'process_arcs/GMOS/S20170108S0085.fits',  # B600:0.500 HAM
    # 'process_arcs/GMOS/S20130510S0103.fits',  # B1200:0.450 EEV
    # 'process_arcs/GMOS/S20130629S0002.fits',  # B1200:0.525 EEV
    # 'process_arcs/GMOS/S20131123S0044.fits',  # B1200:0.595 EEV
    # 'process_arcs/GMOS/S20170116S0189.fits',  # B1200:0.440 HAM - todo: very weird non-linear plot
    # 'process_arcs/GMOS/S20170908S0189.fits',  # B1200:0.550 HAM
    # 'process_arcs/GMOS/S20131230S0153.fits',  # R150:0.550 EEV
    # 'process_arcs/GMOS/S20130801S0140.fits',  # R150:0.700 EEV
    # 'process_arcs/GMOS/S20170430S0060.fits',  # R150:0.717 HAM
    # 'process_arcs/GMOS/S20170430S0063.fits',  # R150:0.727 HAM
    # 'process_arcs/GMOS/S20171102S0051.fits',  # R150:0.950 HAM
    # 'process_arcs/GMOS/S20130114S0100.fits',  # R400:0.620 EEV
    # 'process_arcs/GMOS/S20130217S0073.fits',  # R400:0.800 EEV
    # 'process_arcs/GMOS/S20170108S0046.fits',  # R400:0.550 HAM
    # 'process_arcs/GMOS/S20170129S0125.fits',  # R400:0.685 HAM
    # 'process_arcs/GMOS/S20170703S0199.fits',  # R400:0.800 HAM
    # 'process_arcs/GMOS/S20170718S0420.fits',  # R400:0.910 HAM
    # # 'process_arcs/GMOS/S20110720S0236.fits',  # R600:0.675 EEV - todo: RMS > 0.5 (RMS = 0.508)
    # 'process_arcs/GMOS/S20120322S0122.fits',  # R600:0.900 EEV
    # 'process_arcs/GMOS/S20130803S0011.fits',  # R831:0.576 EEV
    # 'process_arcs/GMOS/S20130414S0040.fits',  # R831:0.845 EEV
    # 'process_arcs/GMOS/S20170214S0059.fits',  # R831:0.440 HAM
    # 'process_arcs/GMOS/S20170703S0204.fits',  # R831:0.600 HAM
    # 'process_arcs/GMOS/S20171018S0048.fits',  # R831:0.865 HAM
    #
    # 'process_arcs/GMOS/N20100115S0346.fits',  # B600:0.500 EEV
    # 'process_arcs/GMOS/N20130112S0390.fits',  # B600:0.500 E2V
    # 'process_arcs/GMOS/N20170609S0173.fits',  # B600:0.500 HAM
    # 'process_arcs/GMOS/N20100307S0236.fits',  # B1200:0.445 EEV
    # 'process_arcs/GMOS/N20130628S0290.fits',  # B1200:0.420 E2V
    # 'process_arcs/GMOS/N20170904S0078.fits',  # B1200:0.440 HAM
]


@pytest.fixture(scope='class', params=dataset_file_list)
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

    logutils.config(mode='quiet', file_name='foo.log')

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

            oldmask = os.umask(000)
            os.makedirs(output_folder, exist_ok=True, mode=0o775)
            os.umask(oldmask)

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

            p = PlotGmosSpectLongslitArcs(ad, output_folder)
            p.plot_all()

        @staticmethod
        def reduce(filename):
            p = primitives_gmos_spect.GMOSSpect([astrodata.open(filename)])
            p.viewer = geminidr.dormantViewer(p, None)

            p.prepare()
            p.addDQ(static_bpm=None)
            p.addVAR(read_noise=True)
            p.overscanCorrect()
            p.ADUToElectrons()
            p.addVAR(poisson_noise=True)
            p.mosaicDetectors()
            p.makeIRAFCompatible()
            # p.determineWavelengthSolution(plot=True)
            p.determineWavelengthSolution()
            p.determineDistortion(suffix="_arc")

            return p

    return ConfigTest(request.param)


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
        self.central_wavelength = ad.central_wavelength() * 1e9  # in nanometers
        self.output_folder = output_folder

        self.package_dir = os.path.dirname(primitives_gmos_spect.__file__)
        self.arc_table = os.path.join(self.package_dir, "lookups", "CuAr_GMOS.dat")
        self.arc_lines = np.loadtxt(self.arc_table, usecols=[0]) / 10.

    def create_artifact_from_plots(self):
        """
        Created a .tar.gz file using the plots generated here so Jenkins can deliver
        it as an artifact.
        """
        # Runs only from inside Jenkins
        if 'BUILD_ID' in os.environ:

            branch_name = os.environ['BRANCH_NAME'].replace('/', '.')
            build_number = int(os.environ['BUILD_NUMBER'])

            tar_name = os.path.join(
                self.output_folder, "plots_{:s}_b{:03d}.tar.gz".format(
                    branch_name, build_number))

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
                    "Failed to update permissions for file: {}".format(target_file))

    def plot_all(self):

        for ext_num, ext in enumerate(self.ad):

            if not hasattr(ext, 'WAVECAL'):
                continue

            peaks = ext.WAVECAL['peaks'] - 1  # ToDo: Refactor peaks to be 0-indexed
            wavelengths = ext.WAVECAL['wavelengths']

            model = astromodels.dict_to_chebyshev(
                dict(
                    zip(
                        ext.WAVECAL["name"], ext.WAVECAL["coefficients"]
                    )
                )
            )

            mask = np.round(np.average(ext.mask, axis=0)).astype(int)
            data = np.ma.masked_where(mask > 0, np.average(ext.data, axis=0))
            data = (data - data.min()) / data.ptp()

            self.plot_lines(ext_num, data, peaks, model)
            self.plot_non_linear_components(ext_num, peaks, wavelengths, model)
            self.plot_residuals(ext_num, peaks, wavelengths, model)
            self.create_artifact_from_plots()

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
            dpi=300, num="{:s}_{:d}_{:s}_{:.0f}".format(
                self.name, ext_num, self.grating, self.central_wavelength))

        w = model(np.arange(data.size))

        arcs = [ax.vlines(line, 0, 1, color="k", alpha=0.25)
                for line in self.arc_lines]
        wavs = [ax.vlines(peak, 0, 1, color="r", ls="--", alpha=0.25)
                for peak in model(peaks)]
        plot, = ax.plot(w, data, 'k-', lw=0.75)

        ax.legend((plot, arcs[0], wavs[0]),
                  ('Normalized Data', 'Reference Lines', 'Matched Lines'))

        x0, x1 = model([0, data.size])

        ax.grid(alpha=0.1)
        ax.set_xlim(x0, x1)
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('Normalized intensity')
        ax.set_title(
            "Wavelength Calibrated Spectrum for\n"
            "{:s} obtained with {:s} at {:.0f}".format(
                self.name, self.grating, self.central_wavelength))

        if x0 > x1:
            ax.invert_xaxis()

        fig_name = os.path.join(
            self.output_folder, "{:s}_{:d}_{:s}_{:.0f}.png".format(
                self.name, ext_num, self.grating, self.central_wavelength))

        fig.savefig(fig_name)

        try:
            os.chmod(fig_name, 0o775)
        except PermissionError:
            warnings.warn(
                "Failed to update permissions for file: {}".format(fig_name))

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
            dpi=300, num="{:s}_{:d}_{:s}_{:.0f}_non_linear_comps".format(
                self.name, ext_num, self.grating, self.central_wavelength))

        non_linear_model = model.copy()
        _ = [setattr(non_linear_model, 'c{}'.format(k), 0) for k in [0, 1]]
        residuals = wavelengths - model(peaks)

        p = np.linspace(min(peaks), max(peaks), 1000)
        ax.plot(model(p), non_linear_model(p), 'C0-', label="Generic Representation")
        ax.plot(model(peaks), non_linear_model(peaks) + residuals, 'ko',
                label="Non linear components and residuals")
        ax.legend()

        ax.grid(alpha=0.25)
        ax.set_xlabel("Wavelength [nm]")

        ax.set_title(
            "Non-linear components for\n"
            "{:s} obtained with {:s} at {:.0f}".format(
                self.name, self.grating, self.central_wavelength))

        fig_name = os.path.join(
            self.output_folder, "{:s}_{:d}_{:s}_{:.0f}_non_linear_comps.png".format(
                self.name, ext_num, self.grating, self.central_wavelength))

        fig.savefig(fig_name)

        try:
            os.chmod(fig_name, 0o775)
        except PermissionError:
            warnings.warn(
                "Failed to update permissions for file: {}".format(fig_name))

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
            dpi=300, num="{:s}_{:d}_{:s}_{:.0f}_residuals".format(
                self.name, ext_num, self.grating, self.central_wavelength))

        ax.plot(wavelengths, wavelengths - model(peaks), 'ko')

        ax.grid(alpha=0.25)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Residuum [nm]")
        ax.set_title(
            "Wavelength Calibrated Residuum for\n"
            "{:s} obtained with {:s} at {:.0f}".format(
                self.name, self.grating, self.central_wavelength))

        fig_name = os.path.join(
            self.output_folder, "{:s}_{:d}_{:s}_{:.0f}_residuals.png".format(
                self.name, ext_num, self.grating, self.central_wavelength))

        fig.savefig(fig_name)

        try:
            os.chmod(fig_name, 0o775)
        except PermissionError:
            warnings.warn(
                "Failed to update permissions for file: {}".format(fig_name))

        del fig, ax


@pytest.mark.gmosls
class TestGmosSpectLongslitArcs:
    """
    Collection of tests that will run on every `dataset` file.
    """

    @staticmethod
    def test_reduced_arcs_contain_wavelength_solution_model_with_expected_rms(config):
        """
        Make sure that the WAVECAL model was fitted with an RMS smaller
        than 0.5.
        """
        for ext in config.ad:

            if not hasattr(ext, 'WAVECAL'):
                continue

            table = ext.WAVECAL
            coefficients = table['coefficients']
            rms = coefficients[table['name'] == 'rms']

            np.testing.assert_array_less(rms, 0.5)

    @staticmethod
    def test_reduced_arcs_contains_stable_wavelength_solution(config):
        """
        Make sure that the wavelength solution gives same results on different
        runs.
        """
        output = os.path.join(config.output_dir, config.filename)
        reference = os.path.join(config.ref_dir, config.filename)

        if not os.path.exists(output):
            pytest.skip('Output file not found: {}'.format(output))

        if not os.path.exists(reference):
            pytest.fail('Reference file not found: {}'.format(reference))

        ad_ref = astrodata.open(reference)

        for ext, ext_ref in zip(config.ad, ad_ref):
            model = astromodels.dict_to_chebyshev(
                dict(zip(ext.WAVECAL["name"], ext.WAVECAL["coefficients"]))
            )

            ref_model = astromodels.dict_to_chebyshev(
                dict(zip(ext_ref.WAVECAL["name"], ext_ref.WAVECAL["coefficients"]))
            )

            x = np.arange(ext.shape[1])
            y = model(x)
            ref_y = ref_model(x)

            np.testing.assert_allclose(y, ref_y, rtol=1)

    @staticmethod
    def test_reduced_arcs_are_similar(config):
        """
        Compares the output and the reference arcs to make sure their data are
        similar before running further tests.
        """
        output = os.path.join(config.output_dir, config.filename)
        reference = os.path.join(config.ref_dir, config.filename)

        if not os.path.exists(output):
            pytest.skip('Output file not found: {}'.format(output))

        if not os.path.exists(reference):
            pytest.fail('Reference file not found: {}'.format(reference))

        ad_ref = astrodata.open(reference)

        # Test reduced arcs are similar
        for ext, ext_ref in zip(config.ad, ad_ref):
            np.testing.assert_allclose(ext.data, ext_ref.data, rtol=1)

    @staticmethod
    def test_distortion_correction_is_applied_the_same_way(config):
        """
        Applies the same distortion correction model to both output and reference
        arcs and compares the results.
        """
        output = os.path.join(config.output_dir, config.filename)
        reference = os.path.join(config.ref_dir, config.filename)

        if not os.path.exists(output):
            pytest.skip('Output file not found: {}'.format(output))

        if not os.path.exists(reference):
            pytest.fail('Reference file not found: {}'.format(reference))

        ad_ref = astrodata.open(reference)

        p = primitives_gmos_spect.GMOSSpect([])

        distortion_corrected_ad = p.distortionCorrect(
            adinputs=[config.ad], arc=config.ad)

        distortion_corrected_ref = p.distortionCorrect(
            adinputs=[ad_ref], arc=config.ad)

        for ext, ext_ref in zip(distortion_corrected_ad, distortion_corrected_ref):
            np.testing.assert_allclose(ext.data, ext_ref.data, rtol=1)

    @staticmethod
    def test_distortion_model_is_the_same(config):
        """
        Correscts distortion on both output and reference files using the
        distortion model stored in themselves. Previous tests assures that
        these data are similar and that distortion correct is applied the same
        way. Now, this one tests the model itself.
        """
        output = os.path.join(config.output_dir, config.filename)
        reference = os.path.join(config.ref_dir, config.filename)

        if not os.path.exists(output):
            pytest.skip('Output file not found: {}'.format(output))

        if not os.path.exists(reference):
            pytest.fail('Reference file not found: {}'.format(reference))

        ad_ref = astrodata.open(reference)

        p = primitives_gmos_spect.GMOSSpect([])

        distortion_corrected_ad = p.distortionCorrect(
            adinputs=[config.ad], arc=config.ad)

        distortion_corrected_ref = p.distortionCorrect(
            adinputs=[ad_ref], arc=ad_ref)

        for ext, ext_ref in zip(distortion_corrected_ad, distortion_corrected_ref):
            np.testing.assert_allclose(ext.data, ext_ref.data, rtol=1)