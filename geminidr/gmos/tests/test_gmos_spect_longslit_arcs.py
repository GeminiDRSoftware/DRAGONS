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
    'process_arcs/GMOS/N20100115S0346.fits',  # B600:0.500 EEV
    # 'process_arcs/GMOS/N20130112S0390.fits',  # B600:0.500 E2V
    # 'process_arcs/GMOS/N20170530S0006.fits',  # B600:0.520 HAM
    # 'process_arcs/GMOS/N20170609S0173.fits',  # B600:0.500 HAM
    # # 'process_arcs/GMOS/N20180119S0232.fits',  # R150:0.520 HAM - todo: RMS > 0.5 and solution mismatch
    # # 'process_arcs/GMOS/N20181114S0512.fits',  # R831:0.865 HAM - todo: RMS > 0.5 (RMS = 0.646)
    # 'process_arcs/GMOS/N20180120S0417.fits',  # R600:0.860 HAM
    # 'process_arcs/GMOS/N20180516S0214.fits',  # R150:0.610 HAM
    # # 'process_arcs/GMOS/S20130218S0126.fits',  # B600:0.500 EEV - todo: breaks p.determineWavelengthSolution()
    # 'process_arcs/GMOS/S20140504S0008.fits',  # B600:0.500 EEV
    # 'process_arcs/GMOS/S20170103S0152.fits',  # B600:0.600 HAM
    # 'process_arcs/GMOS/S20170108S0085.fits',  # B600:0.500 HAM
    # 'process_arcs/GMOS/S20170116S0189.fits',  # B1200:0.440 HAM
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

            p = self.reduce(input_file)

            ad = p.writeOutputs(outfilename=output_file, overwrite=True)[0]
            os.chmod(output_file, mode=0o775)

            self.ad = ad
            self.filename = ad.filename
            self.output_file = output_file
            self.output_dir = output_folder
            self.ref_dir = reference_folder

            for _file in glob.glob(os.path.join(output_folder, "*.png")):
                os.remove(_file)

            plot_lines(ad, output_folder)
            plot_residuals(ad, output_folder)
            plot_non_linear_components(ad, output_folder)

            create_artifact_from_plots(output_folder)

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


def create_artifact_from_plots(output_folder):
    """
    Created a .tar.gz file using the plots generated here so Jenkins can deliver
    it as an artifact.

    Parameters
    ----------
    output_folder : str
        Path to where the PNG files are generated
    """
    # Runs only from inside Jenkins
    if 'BUILD_ID' in os.environ:

        tar_name = os.path.join(
            output_folder, "test_gmos_lsspec_arcs.tar.gz".format())

        with tarfile.open(tar_name, "w:gz") as tar:
            for _file in glob.glob(os.path.join(output_folder, "*.png")):
                tar.add(_file)

        target_dir = "./plots/"
        target_file = os.path.join(target_dir, os.path.basename(tar_name))

        os.makedirs(target_dir, exist_ok=True)
        os.rename(tar_name, target_file)

        try:
            os.chmod(target_file, 0o775)
        except PermissionError:
            warnings.warn(
                "Failed to update permissions for file: {}".format(target_file))


def plot_lines(ad, output_folder):
    """
    Plots and saves the wavelength calibration model residuals for diagnosis.

    Parameters
    ----------
    ad : AstroData
        Arc Lamp with wavelength calibration table `.WAVECAL` in any of its
        extensions.
    output_folder : str
        Path to where the plots will be saved
    """
    filename = ad.filename
    name, _ = os.path.splitext(filename)
    grating = ad.disperser(pretty=True)
    central_wavelength = ad.central_wavelength() * 1e9  # in nanometers

    package_dir = os.path.dirname(primitives_gmos_spect.__file__)
    arc_table = os.path.join(package_dir, "lookups", "CuAr_GMOS.dat")
    arc_lines = np.loadtxt(arc_table, usecols=[0]) / 10.

    for ext_num, ext in enumerate(ad):

        if not hasattr(ext, 'WAVECAL'):
            continue

        peaks = ext.WAVECAL['peaks'] - 1  # ToDo: Refactor peaks to be 0-indexed

        model = astromodels.dict_to_chebyshev(
            dict(
                zip(
                    ad[0].WAVECAL["name"], ad[0].WAVECAL["coefficients"]
                )
            )
        )

        mask = np.round(np.average(ext.mask, axis=0)).astype(int)
        data = np.ma.masked_where(mask > 0, np.average(ext.data, axis=0))
        data = (data - data.min()) / data.ptp()

        fig, ax = plt.subplots(num="{:s}_{:d}_{:s}_{:.0f}".format(
            name, ext_num, grating, central_wavelength), dpi=300)

        w = model(np.arange(data.size))

        arcs = [ax.vlines(line, 0, 1, color="k", alpha=0.25) for line in arc_lines]
        wavs = [ax.vlines(peak, 0, 1, color="r", ls="--", alpha=0.25) for peak in model(peaks)]
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
                name, grating, central_wavelength))

        if x0 > x1:
            ax.invert_xaxis()

        fig_name = os.path.join(output_folder, "{:s}_{:d}_{:s}_{:.0f}.png".format(
            name, ext_num, grating, central_wavelength))

        fig.savefig(fig_name)

        try:
            os.chmod(fig_name, 0o775)
        except PermissionError:
            warnings.warn(
                "Failed to update permissions for file: {}".format(fig_name))

        del fig, ax


def plot_residuals(ad, output_folder):
    """
    Plots the matched wavelengths versus the residuum  between them and their
    correspondent peaks applied to the fitted model

    Parameters
    ----------
    ad : AstroData
        Arc Lamp with wavelength calibration table `.WAVECAL` in any of its
        extensions.
    output_folder : str
        Path to where the plots will be saved
    """
    filename = ad.filename
    name, _ = os.path.splitext(filename)
    grating = ad.disperser(pretty=True)
    central_wavelength = ad.central_wavelength() * 1e9  # in nanometers

    for i, ext in enumerate(ad):

        if not hasattr(ext, 'WAVECAL'):
            continue

        fig, ax = plt.subplots(num="{:s}_{:d}_{:s}_{:.0f}_residuals".format(
            filename, i, grating, central_wavelength), dpi=300)

        peaks = ext.WAVECAL['peaks'] - 1  # ToDo: Refactor peaks to be 0-indexed
        wavelengths = ext.WAVECAL['wavelengths']

        model = astromodels.dict_to_chebyshev(
            dict(
                zip(
                    ad[0].WAVECAL["name"], ad[0].WAVECAL["coefficients"]
                )
            )
        )

        ax.plot(wavelengths, wavelengths - model(peaks - 1), 'ko')

        ax.grid(alpha=0.25)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Residuum [nm]")
        ax.set_title(
            "Wavelength Calibrated Residuum for\n"
            "{:s} obtained with {:s} at {:.0f}".format(
                name, grating, central_wavelength))

        fig_name = os.path.join(output_folder,
                                "{:s}_{:d}_{:s}_{:.0f}_residuals.png".format(
                                    name, i, grating, central_wavelength))

        fig.savefig(fig_name)

        try:
            os.chmod(fig_name, 0o775)
        except PermissionError:
            warnings.warn(
                "Failed to update permissions for file: {}".format(fig_name))

        del fig, ax


def plot_non_linear_components(ad, output_folder):
    """
    Plots the non-linear residuals.

    Parameters
    ----------
    ad : AstroData
        Arc Lamp with wavelength calibration table `.WAVECAL` in any of its
        extensions.
    output_folder : str
        Path to where the plots will be saved
    """
    filename = ad.filename
    name, _ = os.path.splitext(filename)
    grating = ad.disperser(pretty=True)
    central_wavelength = ad.central_wavelength() * 1e9  # in nanometers

    for ext_num, ext in enumerate(ad):

        if not hasattr(ext, 'WAVECAL'):
            continue

        fig, ax = plt.subplots(num="{:s}_{:d}_{:s}_{:.0f}_non_linear_comps".format(
            filename, ext_num, grating, central_wavelength), dpi=300)

        peaks = ext.WAVECAL['peaks'] - 1  # ToDo: Refactor peaks to be 0-indexed
        wavelengths = ext.WAVECAL['wavelengths']

        model = astromodels.dict_to_chebyshev(
            dict(
                zip(
                    ad[0].WAVECAL["name"], ad[0].WAVECAL["coefficients"]
                )
            )
        )

        non_linear_model = model.copy()
        _ = [setattr(non_linear_model, 'c{}'.format(k), 0) for k in [0, 1]]
        residuals = wavelengths - model(peaks)

        p = np.linspace(min(peaks), max(peaks), 1000)
        ax.plot(model(p), non_linear_model(p), 'C0-', label="Generic Representation")
        ax.plot(model(peaks), non_linear_model(peaks) + residuals, 'ko', label="Non linear components and residuals")
        ax.legend()

        ax.grid(alpha=0.25)
        ax.set_xlabel("Wavelength [nm]")

        ax.set_title(
            "Non-linear components for\n"
            "{:s} obtained with {:s} at {:.0f}".format(
                name, grating, central_wavelength))

        fig_name = os.path.join(
            output_folder, "{:s}_{:d}_{:s}_{:.0f}_non_linear_comps.png".format(
                name, ext_num, grating, central_wavelength))

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
