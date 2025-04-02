#!/usr/bin/env python
"""
Plots related to GMOS Long-slit Spectroscopy Arc primitives.

todo @bquint: Needs clean up. Since I started to refactor the tests for GMOS LS
 arcs, this could be much simpler.
"""

import glob
import os
import tarfile
import warnings

import numpy as np
from astropy.modeling import models

# noinspection PyPackageRequirements
from matplotlib import colors
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage

import astrodata

# noinspection PyUnresolvedReferences
import gemini_instruments
from geminidr.gmos import primitives_gmos_spect
from gempy.library import astromodels as am
from gempy.library import transform

plt.ioff()


def generate_fake_data(shape, dispersion_axis, n_lines=100):
    """
    Helper function that generates fake arc data.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the output data with (nrows, ncols)
    dispersion_axis : {0, 1}
        Dispersion axis along rows (0) or along columns (1)
    n_lines : int
        Number of random lines to be added (default: 100)

    Returns
    -------
    :class:`~astropy.modeling.models.Model`
        2D Model that can be applied to an array.
    """
    np.random.seed(0)
    nrows, ncols = shape

    data = np.zeros((nrows, ncols))
    line_positions = np.random.random_integers(0, ncols, size=n_lines)
    line_intensities = 100 * np.random.random_sample(n_lines)

    if dispersion_axis == 0:
        data[:, line_positions] = line_intensities
        data = ndimage.gaussian_filter(data, [5, 1])
    else:
        data[line_positions, :] = line_intensities
        data = ndimage.gaussian_filter(data, [1, 5])

    data = data + (np.random.random_sample(data.shape) - 0.5) * 10

    return data


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

    def __init__(self, ad, output_folder, ref_folder=None):

        filename = ad.filename
        self.ad = ad
        self.name, _ = os.path.splitext(filename)
        self.grating = ad.disperser(pretty=True)
        self.bin_x = ad.detector_x_bin()
        self.bin_y = ad.detector_y_bin()
        self.central_wavelength = ad.central_wavelength() * 1e9  # in nanometers
        self.output_folder = output_folder
        self.ref_folder = ref_folder

        self.package_dir = os.path.dirname(primitives_gmos_spect.__file__)
        self.arc_table = os.path.join(self.package_dir, "lookups", "CuAr_GMOS.dat")
        self.arc_lines = np.loadtxt(self.arc_table, usecols=[0]) / 10.0

    def close_all(self):
        """
        Makes sure we close every figure in memory.
        """
        plt.close('all')

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

    def distortion_diagnosis_plots(self):
        """
        Makes the Diagnosis Plots for `determineDistortion` and
        `distortionCorrect` for each extension inside the reduced arc.
        """
        output_file = os.path.join(self.output_folder, self.name + ".fits")
        reference_file = os.path.join(self.ref_folder, self.name + ".fits")

        ad = astrodata.from_file(output_file)
        ad_ref = astrodata.from_file(reference_file)

        self.show_distortion_map(ad)
        self.show_distortion_model_difference(ad, ad_ref)

    def plot_distortion_residuals(self, ad):
        """
        Plots the distortion residuals calculated on an arc dataset that passed
        through `distortionCorrect`. The residuals are calculated based on an
        artificial mesh and using a model obtained from `determinedDistortion`
        applied to the distortion corrected file.

        Parameters
        ----------
        ad : AstroData
            Distortion Determined arc.
        """
        for num, ext in enumerate(ad):

            fname, _ = os.path.splitext(os.path.basename(ext.filename))

            n_hlines = 25
            n_vlines = 25
            n_rows, n_cols = ext.shape

            x = np.linspace(0, n_cols, n_vlines, dtype=int)
            y = np.linspace(0, n_rows, n_hlines, dtype=int)

            X, Y = np.meshgrid(x, y)

            model = ext.wcs.get_transform("pixels", "distortion_corrected")[1]
            U = X - model(X, Y)

            width = 0.75 * np.diff(x).mean()
            _min, _med, _max = np.percentile(U, [0, 50, 100], axis=0)

            fig, ax = plt.subplots(
                num="Corrected Distortion Residual Stats {:s} #{:d}".format(fname, num)
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
                    fname, num, self.grating, self.central_wavelength
                ),
            )

            fig.savefig(fig_name)

    def plot_lines(self, ext_num, data, peaks, model):
        """
        Plots the spectrum with the normalized data, the reference lines and
        the matched lines. The reference lines are obtained from the default
        look-up table. The matched lines are calculated using the `peaks`
        applied to the `model`.

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
            dpi=150,
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
            "{:s}\n obtained with {:s} at {:.0f} nm".format(
                self.name, self.grating, self.central_wavelength
            )
        )

        if x0 > x1:
            ax.invert_xaxis()

        fig_name = os.path.join(
            self.output_folder,
            "{:s}_{:d}_{:s}_{:.0f}.svg".format(
                self.name, ext_num, self.grating, self.central_wavelength
            ),
        )

        fig.savefig(fig_name)
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
            dpi=150,
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
            "{:s}_{:d}_{:s}_{:.0f}_non_linear_comps.svg".format(
                self.name, ext_num, self.grating, self.central_wavelength
            ),
        )

        fig.savefig(fig_name)
        del fig, ax

    def plot_wavelength_solution_residuals(self, ext_num, peaks, wavelengths, model):
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
            dpi=150,
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
            "{:s}_{:d}_{:s}_{:.0f}_residuals.svg".format(
                self.name, ext_num, self.grating, self.central_wavelength
            ),
        )

        fig.savefig(fig_name)

        del fig, ax

    def show_distortion_model_difference(self, ad, ad_ref):
        """
        Shows the difference between the distortion corrected output file and
        the corresponding reference file.

        Parameters
        ----------
        ad : AstroData
            Distortion Determined AstroData object
        ad_ref : AstroData
            Distortion Determined AstroData reference object
        """
        for num, (ext, ext_ref) in enumerate(zip(ad, ad_ref)):

            name, _ = os.path.splitext(ext.filename)
            shape = ext.shape
            data = generate_fake_data(shape, ext.dispersion_axis() - 1)

            model_out = ext.wcs.get_transform("pixels", "distortion_corrected")
            model_ref = ext_ref.wcs.get_transform("pixels",
                                                  "distortion_corrected")

            transform_out = transform.Transform(model_out)
            transform_ref = transform.Transform(model_ref)

            data_out = transform_out.apply(data, output_shape=ext.shape)
            data_ref = transform_ref.apply(data, output_shape=ext.shape)

            data_out = np.ma.masked_invalid(data_out)
            data_ref = np.ma.masked_invalid(data_ref)

            fig, ax = plt.subplots(
                dpi=150, num="Distortion Comparison: {:s} #{:d}".format(name, num)
            )

            im = ax.imshow(data_ref - data_out)

            ax.set_xlabel("X [px]")
            ax.set_ylabel("Y [px]")
            ax.set_title(
                "Difference between output and reference: \n {:s} #{:d} ".format(
                    name, num
                )
            )

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            cbar = fig.colorbar(im, extend="max", cax=cax, orientation="vertical")
            cbar.set_label("Distortion [px]")

            fig_name = os.path.join(
                self.output_folder,
                "{:s}_{:d}_{:s}_{:.0f}_distDiff.svg".format(
                    name, num, self.grating, self.central_wavelength
                ),
            )

            fig.savefig(fig_name)

    def show_distortion_map(self, ad):
        """
        Plots the distortion map determined for a given file.

        Parameters
        ----------
        ad : AstroData
            Distortion determined data.
        """
        n_hlines = 50
        n_vlines = 50

        for num, ext in enumerate(ad):

            fname, _ = os.path.splitext(os.path.basename(ext.filename))

            n_rows, n_cols = ext.shape

            x = np.linspace(0, n_cols, n_vlines, dtype=int)
            y = np.linspace(0, n_rows, n_hlines, dtype=int)

            X, Y = np.meshgrid(x, y)

            model = ext.wcs.get_transform("pixels", "distortion_corrected")[1]
            U = X - model(X, Y)
            V = np.zeros_like(U)

            fig, ax = plt.subplots(num="Distortion Map {:s} #{:d}".format(fname, num))

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
                "Distortion Map\n{:s} #{:d}- Bin {:d}x{:d}".format(
                    fname, num, self.bin_x, self.bin_y
                )
            )

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            cbar = fig.colorbar(Q, extend="max", cax=cax, orientation="vertical")
            cbar.set_label("Distortion [px]")

            fig.tight_layout()

            fig_name = os.path.join(
                self.output_folder,
                "{:s}_{:d}_{:s}_{:.0f}_distMap.svg".format(
                    fname, num, self.grating, self.central_wavelength
                ),
            )

            fig.savefig(fig_name)
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

            wavecal_model = am.get_named_submodel(ext.wcs.forward_transform, "WAVE")

            middle = ext.data.shape[0] // 2
            sum_size = 10
            r1 = middle - sum_size // 2
            r2 = middle + sum_size // 2

            mask = np.round(np.average(ext.mask[r1:r2], axis=0)).astype(int)
            data = np.ma.masked_where(mask > 0, np.sum(ext.data[r1:r2], axis=0))
            data = (data - data.min()) / data.ptp()

            self.plot_lines(ext_num, data, peaks, wavecal_model)
            self.plot_non_linear_components(ext_num, peaks, wavelengths, wavecal_model)
            self.plot_wavelength_solution_residuals(
                ext_num, peaks, wavelengths, wavecal_model
            )
            self.create_artifact_from_plots()
