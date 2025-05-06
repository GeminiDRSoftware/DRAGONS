#
#                                                                  gemini_python
#
#                                                     primtives_gmos_longslit.py
# ------------------------------------------------------------------------------

from copy import copy, deepcopy
from importlib import import_module
from itertools import groupby

from gempy.library.config import RangeField

import astrodata
import geminidr
import numpy as np
from scipy.signal import correlate

from astrodata.provenance import add_provenance
from astropy import visualization as vis
from astropy.modeling import models, fitting

from geminidr.gemini.lookups import DQ_definitions as DQ

from gempy.gemini import gemini_tools as gt
from gempy.library.fitting import fit_1D
from gempy.library import astromodels, peak_finding, transform
from gempy.library import astrotools as at

from gwcs import coordinate_frames
from gwcs.wcs import WCS as gWCS

from matplotlib import gridspec
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from recipe_system.utils.decorators import parameter_override, capture_provenance
from recipe_system.utils.md5 import md5sum

from .primitives_gmos_spect import GMOSSpect
from .primitives_gmos_nodandshuffle import GMOSNodAndShuffle
from ..core.primitives_longslit import Longslit
from . import parameters_gmos_longslit


# ------------------------------------------------------------------------------
from ..interactive.fit import fit1d
from ..interactive.fit.help import NORMALIZE_FLAT_HELP_TEXT
from ..interactive.interactive import UIParameters


@parameter_override
@capture_provenance
class GMOSLongslit():
    """
    "Magic" class to provide the correct class for N&S and classic data
    """
    tagset = {"GEMINI", "GMOS", "SPECT", "LS"}

    def __new__(cls, adinputs, **kwargs):
        if adinputs:
            _class = GMOSNSLongslit if "NODANDSHUFFLE" in adinputs[0].tags else GMOSClassicLongslit
            return _class(adinputs, **kwargs)
        raise ValueError("GMOSLongslit objects cannot be instantiated without"
                         " specifying 'adinputs'. Please instantiate either"
                         "'GMOSClassicLongslit' or 'GMOSNSLongslit' instead.")


@parameter_override
@capture_provenance
class GMOSClassicLongslit(GMOSSpect, Longslit):
    """
    This is the class containing all of the preprocessing primitives
    for the GMOSLongslit level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_gmos_longslit)

    def addIllumMaskToDQ(self, adinputs=None, suffix=None, illum_mask=None,
                         shift=None, max_shift=20, debug_plot=False):
        """
        Adds an illumination mask to each AD object. This is only done for
        full-frame (not Central Spectrum) GMOS spectra, and is calculated by
        making a model illumination patter from the attached MDF and cross-
        correlating it with the spatial profile of the data.

        Parameters
        ----------
        suffix : str
            suffix to be added to output files
        illum_mask : str/None
            name of illumination mask mask (None -> use default)
        shift : int/None
            user-defined shift to apply to illumination mask
        max_shift : int
            maximum shift (in unbinned pixels) allowable for the cross-
            correlation
        """
        offset_dict = {("GMOS-N", "Hamamatsu-N"): 1.5,
                       ("GMOS-N", "e2vDD"): -0.2,
                       ("GMOS-N", "EEV"): 0.7,
                       ("GMOS-S", "Hamamatsu-S"): 5.5,
                       ("GMOS-S", "EEV"): 3.8}
        edges = 60  # try to eliminate issues at the very edges

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        # Do this now for memory management reasons. We'll be creating large
        # arrays temporarily and don't want the permanent mask arrays to
        # fragment the free memory.
        for ad in adinputs:
            for ext in ad:
                if ext.mask is None:
                    ext.mask = np.zeros_like(ext.data).astype(DQ.datatype)

        for ad, illum in zip(*gt.make_lists(adinputs, illum_mask, force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning(f'No changes will be made to {ad.filename}, since'
                    ' it has already been processed by addIllumMaskToDQ')
                continue

            xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
            mshift = max_shift // ybin + 10
            ad_detsec = ad.detector_section()
            no_bridges = all(detsec.y1 > 1600 and detsec.y2 < 2900
                             for detsec in ad_detsec)
            has_48rows = (all(detsec.y2 == 4224 for detsec in ad_detsec) and
                          'Hamamatsu' in ad.detector_name(pretty=True))

            if has_48rows:
                actual_rows = 48 // ybin
                for ext in ad:
                    ext.mask[:actual_rows] |= DQ.unilluminated

            if illum:
                log.fullinfo("Using {} as illumination mask".format(illum.filename))
                final_illum = gt.clip_auxiliary_data(ad, aux=illum, aux_type='bpm',
                                          return_dtype=DQ.datatype)

                for ext, illum_ext in zip(ad, final_illum):
                    if illum_ext is not None:
                        # Ensure we're only adding the unilluminated bit
                        iext = np.where(illum_ext.data > 0, DQ.unilluminated,
                                        0).astype(DQ.datatype)
                        ext.mask |= iext
            elif not no_bridges:   # i.e. there are bridges.
                try:
                    mdf = ad.MDF
                except AttributeError:
                    log.warning(f"MDF not present in {ad.filename} - cannot "
                                "add illumination mask.")
                    continue

                # Default operation for GMOS full-frame LS
                # Sadly, we cannot do this reliably without concatenating the
                # arrays and using a big chunk of memory. Try at least to
                # control memory usage
                #####row_medians = np.zeros((ad[0].shape[0] + 2 * mshift,))
                max_xsize = max([ext.detector_section().x2 for ext in ad]) // xbin
                all_data = np.zeros((ad[0].shape[0], max_xsize))
                for ext in ad:
                    _slice = (slice(None), slice(ext.detector_section().x1 // xbin,
                                                 ext.detector_section().x2 // xbin))
                    all_data[_slice] = ext.data[ext.data_section().asslice()]
                    if ext.mask is not None:
                        all_data[_slice][ext.mask[ext.data_section().asslice()]
                                         > 0] = np.nan
                # To supress a numpy RuntimeWarning if a row is all NaNs, we
                # set such rows to zero. Hopefully these will get boxcar'd out
                all_data[np.isnan(all_data).min(axis=1)] = 0
                row_medians = np.nanpercentile(all_data, 95, axis=1)
                del all_data

                # Construct a model of the slit illumination from the MDF
                # coefficients are from G-IRAF except c0, approx. from data
                # Pad model to ensure cross-correlation doesn't go off the edge
                model = np.zeros((row_medians.size + 2 * mshift,), dtype=int)
                slit_location_msg = ""
                for ypos, ysize in mdf['slitpos_my', 'slitsize_my']:
                    y = ypos + np.array([-0.5, 0.5]) * ysize
                    c0 = offset_dict[ad.instrument(), ad.detector_name(pretty=True)]
                    if ad.instrument() == "GMOS-S":
                        c1, c2, c3 = (0.99911, -1.7465e-5, 3.0494e-7)
                    else:
                        c1, c2, c3 = (0.99591859227, 5.3042211333437e-8,
                                      1.7447902551997e-7)
                    yccd = ((c0 + y * (c1 + y * (c2 + y * c3))) *
                            1.611444 / ad.pixel_scale() + 0.5 * row_medians.size).astype(int)
                    model[yccd[0]+mshift:yccd[1]+mshift+1] = 1
                    slit_location_msg += ("Expected slit location from pixels "
                                          f"{yccd[0]+1} to {yccd[1]+1}\n")
                log.stdinfo(slit_location_msg)

                # For N&S data, repeat the slit below where the MDF locates it
                if 'NODANDSHUFFLE' in ad.tags:
                    shuffle_pixels = ad.shuffle_pixels() // ybin
                    model[:-shuffle_pixels] += model[shuffle_pixels:]

                # Find largest number of pixels between slits, which will
                # define a smoothing box scale. This is necessary to take out
                # the slit function, if most of the detector is illuminated
                if model.mean() > 0.75:
                    longest_gap = max([len(list(group)) for item, group in
                                       groupby(model) if item == 0])
                    row_medians -= at.boxcar(row_medians, size=longest_gap // 2)
                # Remove single bad rows
                row_medians = at.boxcar(row_medians, size=2)
                #print(row_medians.min(), row_medians.max())

                if debug_plot:  # pragma: no cover
                    plt.ioff()
                    fig, ax = plt.subplots()
                    ax.plot(row_medians / row_medians.max(), 'b-')
                    ax.plot(model[mshift:-mshift], 'k-')
                    print(f"Row medians scaled by {row_medians.max()}")

                if shift is None:
                    rmed = row_medians[edges:-edges]
                    xcorr = correlate(model, rmed - rmed.min(), mode='valid')
                    cntr = xcorr.size // 2
                    xcorr = xcorr[cntr-mshift:cntr+mshift+1]
                    # This line avoids numerical errors
                    xcorr -= np.median(xcorr)

                    # Only keep maxima if the fitted peak value is close to
                    # the actual peak (should remove single-pixel peaks)
                    extrema = peak_finding.get_extrema(xcorr, remove_edge_maxima=False)
                    if debug_plot:  # pragma: no cover
                        print(extrema)
                    maxima = [int(x[0] + 0.5) for x in extrema if x[2]]

                    if debug_plot:  # pragma: no cover
                        xpixels = row_medians.size // 2 - mshift + np.arange(xcorr.size)
                        ax.plot(xpixels, xcorr / xcorr.max(), 'r-')
                        ax.plot([row_medians.size // 2] * 2, [0, 1], 'r:')
                        ax.set_ylim(-0.01, 1.01)
                        plt.show()
                        plt.ion()

                    yshift = mshift - maxima[0]
                    if len(maxima) > 1 or abs(yshift) > mshift:
                        log.warning(f"{ad.filename}: cross-correlation peak is"
                                    " untrustworthy so not adding illumination "
                                    "mask. Please re-run with a specified shift.")
                        yshift = None
                        log.stdinfo(slit_location_msg)
                else:
                    yshift = shift

                if yshift is not None:
                    log.debug(slit_location_msg)
                    log.stdinfo(f"{ad.filename}: Shifting mask by {yshift} pixels")
                    # Trim excess to return model to same size as slit profile
                    model = model[mshift:-mshift]
                    row_mask = np.ones_like(model, dtype=int)
                    if yshift < 0:
                        row_mask[:yshift] = 1 - model[-yshift:]
                    elif yshift > 0:
                        row_mask[yshift:] = 1 - model[:-yshift]
                    else:
                        row_mask[:] = 1 - model
                    row_mask = at.boxcar(row_mask, operation=np.bitwise_or, size=2)
                    for ext in ad:
                        ext.mask |= (row_mask * DQ.unilluminated).astype(
                            DQ.datatype)[:, np.newaxis]
                    slices = np.ma.clump_masked(
                        np.ma.masked_array(np.zeros_like(row_mask), row_mask))
                    for _slice in slices:
                        log.debug(f"Masking rows {_slice.start+1} to {_slice.stop}")

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)

        return adinputs

    def makeSlitIllum(self, adinputs=None, **params):
        """
        Makes the processed Slit Illumination Function by binning a 2D
        spectrum along the dispersion direction, fitting a smooth function
        for each bin, fitting a smooth 2D model, and reconstructing the 2D
        array using this last model.

        Its implementation based on the IRAF's `noao.twodspec.longslit.illumination`
        task following the algorithm described in [Valdes, 1968].

        It expects an input calibration image to be an a dispersed image of the
        slit without illumination problems (e.g, twilight flat). The spectra is
        not required to be smooth in wavelength and may contain strong emission
        and absorption lines. The image should contain a `.mask` attribute in
        each extension, and it is expected to be overscan and bias corrected.

        Parameters
        ----------
        adinputs : list
            List of AstroData objects containing the dispersed image of the
            slit of a source free of illumination problems. The data needs to
            have been overscan and bias corrected and is expected to have a
            Data Quality mask.
        bins : {None, int}, optional
            Total number of bins across the dispersion axis. If None,
            the number of bins will match the number of extensions on each
            input AstroData object. It it is an int, it will create N bins
            with the same size.
        border : int, optional
            Border size that is added on every edge of the slit illumination
            image before cutting it down to the input AstroData frame.
        smooth_order : int, optional
            Order of the spline that is used in each bin fitting to smooth
            the data (Default: 3)
        x_order : int, optional
            Order of the x-component in the Chebyshev2D model used to
            reconstruct the 2D data from the binned data.
        y_order : int, optional
            Order of the y-component in the Chebyshev2D model used to
            reconstruct the 2D data from the binned data.

        Return
        ------
        List of AstroData : containing an AstroData with the Slit Illumination
            Response Function for each of the input object.

        References
        ----------
        .. [Valdes, 1968] Francisco Valdes "Reduction Of Long Slit Spectra With
           IRAF", Proc. SPIE 0627, Instrumentation in Astronomy VI,
           (13 October 1986); https://doi.org/10.1117/12.968155
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        suffix = params["suffix"]
        bins = params["bins"]
        border = params["border"]
        debug_plot = params["debug_plot"]
        smooth_order = params["smooth_order"]
        cheb2d_x_order = params["x_order"]
        cheb2d_y_order = params["y_order"]

        ad_outputs = []
        for ad in adinputs:

            if len(ad) > 1 and "mosaic" not in ad[0].wcs.available_frames:

                log.info('Add "mosaic" gWCS frame to input data')
                geotable = import_module('.geometry_conf', self.inst_lookups)

                # deepcopy prevents modifying input `ad` inplace
                ad = transform.add_mosaic_wcs(deepcopy(ad), geotable)

                log.info("Temporarily mosaicking multi-extension file")
                mosaicked_ad = transform.resample_from_wcs(
                    ad, "mosaic", attributes=None, interpolant="linear",
                    process_objcat=False)

            else:

                log.info('Input data already has one extension and has a '
                         '"mosaic" frame.')

                # deepcopy prevents modifying input `ad` inplace
                mosaicked_ad = deepcopy(ad)

            log.info("Transposing data if needed")
            dispaxis = 2 - mosaicked_ad[0].dispersion_axis()  # python sense
            should_transpose = dispaxis == 1

            data, mask, variance = at.transpose_if_needed(
                mosaicked_ad[0].data, mosaicked_ad[0].mask,
                mosaicked_ad[0].variance, transpose=should_transpose)

            log.info("Masking data")
            data = np.ma.masked_array(data, mask=mask)
            variance = np.ma.masked_array(variance, mask=mask)
            std = np.sqrt(variance)  # Easier to work with

            log.info("Creating bins for data and variance")
            height = data.shape[0]
            width = data.shape[1]

            if bins is None:
                nbins = max(len(ad), 12)
                bin_limits = np.linspace(0, height, nbins + 1, dtype=int)
            elif isinstance(bins, int):
                nbins = bins
                bin_limits = np.linspace(0, height, nbins + 1, dtype=int)
            else:
                # ToDo: Handle input bins as array
                raise TypeError("Expected None or Int for `bins`. "
                                "Found: {}".format(type(bins)))

            bin_top = bin_limits[1:]
            bin_bot = bin_limits[:-1]
            binned_data = np.zeros_like(data)
            binned_std = np.zeros_like(std)

            log.info("Smooth binned data and variance, and normalize them by "
                     "smoothed central value")
            for bin_idx, (b0, b1) in enumerate(zip(bin_bot, bin_top)):

                rows = np.arange(width)

                avg_data = np.ma.mean(data[b0:b1], axis=0)
                model_1d_data = astromodels.UnivariateSplineWithOutlierRemoval(
                    rows, avg_data, order=smooth_order)

                avg_std = np.ma.mean(std[b0:b1], axis=0)
                model_1d_std = astromodels.UnivariateSplineWithOutlierRemoval(
                    rows, avg_std, order=smooth_order)

                slit_central_value = model_1d_data(rows)[width // 2]
                binned_data[b0:b1] = model_1d_data(rows) / slit_central_value
                binned_std[b0:b1] = model_1d_std(rows) / slit_central_value

            log.info("Reconstruct 2D mosaicked data")
            bin_center = np.array(0.5 * (bin_bot + bin_top), dtype=int)
            cols_fit, rows_fit = np.meshgrid(np.arange(width), bin_center)

            fitter = fitting.LinearLSQFitter()
            model_2d_init = models.Chebyshev2D(
                x_degree=cheb2d_x_order, x_domain=(0, width),
                y_degree=cheb2d_y_order, y_domain=(0, height))

            model_2d_data = fitter(model_2d_init, cols_fit, rows_fit,
                                   binned_data[rows_fit, cols_fit])

            model_2d_std = fitter(model_2d_init, cols_fit, rows_fit,
                                  binned_std[rows_fit, cols_fit])

            rows_val, cols_val = \
                np.mgrid[-border:height+border, -border:width+border]

            slit_response_data = model_2d_data(cols_val, rows_val)
            slit_response_mask = np.pad(mask, border, mode='edge')  # ToDo: any update to the mask?
            slit_response_std = model_2d_std(cols_val, rows_val)
            slit_response_var = slit_response_std ** 2

            del cols_fit, cols_val, rows_fit, rows_val

            _data, _mask, _variance = at.transpose_if_needed(
                slit_response_data, slit_response_mask, slit_response_var,
                transpose=dispaxis == 1)

            log.info("Update slit response data and data_section")
            slit_response_ad = deepcopy(mosaicked_ad)
            slit_response_ad[0].data = _data
            slit_response_ad[0].mask = _mask
            slit_response_ad[0].variance = _variance

            if "mosaic" in ad[0].wcs.available_frames:

                log.info("Map coordinates between slit function and mosaicked data")  # ToDo: Improve message?
                slit_response_ad = _split_mosaic_into_extensions(
                    ad, slit_response_ad, border_size=border)

            elif len(ad) == 1:

                log.info("Trim out borders")

                slit_response_ad[0].data = \
                    slit_response_ad[0].data[border:-border, border:-border]
                slit_response_ad[0].mask = \
                    slit_response_ad[0].mask[border:-border, border:-border]
                slit_response_ad[0].variance = \
                    slit_response_ad[0].variance[border:-border, border:-border]

            log.info("Update metadata and filename")
            gt.mark_history(
                slit_response_ad, primname=self.myself(), keyword=timestamp_key)

            slit_response_ad.update_filename(suffix=suffix, strip=True)
            ad_outputs.append(slit_response_ad)

            # Plotting ------
            if debug_plot:  # pragma: no cover

                log.info("Creating plots")
                palette = copy(plt.cm.cividis)
                palette.set_bad('r', 0.75)

                norm = vis.ImageNormalize(data[~data.mask],
                                          stretch=vis.LinearStretch(),
                                          interval=vis.PercentileInterval(97))

                fig = plt.figure(
                    num="Slit Response from MEF - {}".format(ad.filename),
                    figsize=(12, 9), dpi=110)

                gs = gridspec.GridSpec(nrows=2, ncols=3, figure=fig)

                # Display raw mosaicked data and its bins ---
                ax1 = fig.add_subplot(gs[0, 0])
                im1 = ax1.imshow(data, cmap=palette, origin='lower',
                                 vmin=norm.vmin, vmax=norm.vmax)

                ax1.set_title("Mosaicked Data\n and Spectral Bins", fontsize=10)
                ax1.set_xlim(-1, data.shape[1])
                ax1.set_xticks([])
                ax1.set_ylim(-1, data.shape[0])
                ax1.set_yticks(bin_center)
                ax1.tick_params(axis=u'both', which=u'both', length=0)

                ax1.set_yticklabels(
                    ["Bin {}".format(i) for i in range(len(bin_center))],
                    fontsize=6)

                _ = [ax1.spines[s].set_visible(False) for s in ax1.spines]
                _ = [ax1.axhline(b, c='w', lw=0.5) for b in bin_limits]

                divider = make_axes_locatable(ax1)
                cax1 = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im1, cax=cax1)

                # Display non-smoothed bins ---
                ax2 = fig.add_subplot(gs[0, 1])
                im2 = ax2.imshow(binned_data, cmap=palette, origin='lower')

                ax2.set_title("Binned, smoothed\n and normalized data ", fontsize=10)
                ax2.set_xlim(0, data.shape[1])
                ax2.set_xticks([])
                ax2.set_ylim(0, data.shape[0])
                ax2.set_yticks(bin_center)
                ax2.tick_params(axis=u'both', which=u'both', length=0)

                ax2.set_yticklabels(
                    ["Bin {}".format(i) for i in range(len(bin_center))],
                    fontsize=6)

                _ = [ax2.spines[s].set_visible(False) for s in ax2.spines]
                _ = [ax2.axhline(b, c='w', lw=0.5) for b in bin_limits]

                divider = make_axes_locatable(ax2)
                cax2 = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im2, cax=cax2)

                # Display reconstructed slit response ---
                vmin = slit_response_data.min()
                vmax = slit_response_data.max()

                ax3 = fig.add_subplot(gs[1, 0])
                im3 = ax3.imshow(slit_response_data, cmap=palette,
                                 origin='lower', vmin=vmin, vmax=vmax)

                ax3.set_title("Reconstructed\n Slit response", fontsize=10)
                ax3.set_xlim(0, data.shape[1])
                ax3.set_xticks([])
                ax3.set_ylim(0, data.shape[0])
                ax3.set_yticks([])
                ax3.tick_params(axis=u'both', which=u'both', length=0)
                _ = [ax3.spines[s].set_visible(False) for s in ax3.spines]

                divider = make_axes_locatable(ax3)
                cax3 = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im3, cax=cax3)

                # Display extensions ---
                ax4 = fig.add_subplot(gs[1, 1])
                ax4.set_xticks([])
                ax4.set_yticks([])
                _ = [ax4.spines[s].set_visible(False) for s in ax4.spines]

                sub_gs4 = gridspec.GridSpecFromSubplotSpec(
                    nrows=len(ad), ncols=1, subplot_spec=gs[1, 1], hspace=0.03)

                # The [::-1] is needed to put the fist extension in the bottom
                for i, ext in enumerate(slit_response_ad[::-1]):

                    ext_data, ext_mask, ext_variance = at.transpose_if_needed(
                        ext.data, ext.mask, ext.variance, transpose=dispaxis == 1)

                    ext_data = np.ma.masked_array(ext_data, mask=ext_mask)

                    sub_ax = fig.add_subplot(sub_gs4[i])

                    im4 = sub_ax.imshow(ext_data, origin="lower", vmin=vmin,
                                        vmax=vmax, cmap=palette)

                    sub_ax.set_xlim(0, ext_data.shape[1])
                    sub_ax.set_xticks([])
                    sub_ax.set_ylim(0, ext_data.shape[0])
                    sub_ax.set_yticks([ext_data.shape[0] // 2])

                    sub_ax.set_yticklabels(
                        ["Ext {}".format(len(slit_response_ad) - i - 1)],
                        fontsize=6)

                    _ = [sub_ax.spines[s].set_visible(False) for s in sub_ax.spines]

                    if i == 0:
                        sub_ax.set_title("Multi-extension\n Slit Response Function")

                divider = make_axes_locatable(ax4)
                cax4 = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im4, cax=cax4)

                # Display Signal-To-Noise Ratio ---
                snr = data / np.sqrt(variance)

                norm = vis.ImageNormalize(snr[~snr.mask],
                                          stretch=vis.LinearStretch(),
                                          interval=vis.PercentileInterval(97))

                ax5 = fig.add_subplot(gs[0, 2])

                im5 = ax5.imshow(snr, cmap=palette, origin='lower',
                                 vmin=norm.vmin, vmax=norm.vmax)

                ax5.set_title("Mosaicked Data SNR", fontsize=10)
                ax5.set_xlim(-1, data.shape[1])
                ax5.set_xticks([])
                ax5.set_ylim(-1, data.shape[0])
                ax5.set_yticks(bin_center)
                ax5.tick_params(axis=u'both', which=u'both', length=0)

                ax5.set_yticklabels(
                    ["Bin {}".format(i) for i in range(len(bin_center))],
                    fontsize=6)

                _ = [ax5.spines[s].set_visible(False) for s in ax5.spines]
                _ = [ax5.axhline(b, c='w', lw=0.5) for b in bin_limits]

                divider = make_axes_locatable(ax5)
                cax5 = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im5, cax=cax5)

                # Display Signal-To-Noise Ratio of Slit Illumination ---
                slit_response_snr = np.ma.masked_array(
                    slit_response_data / np.sqrt(slit_response_var),
                    mask=slit_response_mask)

                ax6 = fig.add_subplot(gs[1, 2])

                im6 = ax6.imshow(slit_response_snr, origin="lower",
                                 vmin=norm.vmin, vmax=norm.vmax, cmap=palette)

                ax6.set_xlim(0, slit_response_snr.shape[1])
                ax6.set_xticks([])
                ax6.set_ylim(0, slit_response_snr.shape[0])
                ax6.set_yticks([])
                ax6.set_title("Reconstructed\n Slit Response SNR")

                _ = [ax6.spines[s] .set_visible(False) for s in ax6.spines]

                divider = make_axes_locatable(ax6)
                cax6 = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im6, cax=cax6)

                # Save plots ---
                fig.tight_layout(rect=[0, 0, 0.95, 1], pad=0.5)
                fname = slit_response_ad.filename.replace(".fits", ".png")
                log.info("Saving plots to {}".format(fname))
                plt.savefig(fname)

        return ad_outputs

    def normalizeFlat(self, adinputs=None, **params):
        """
        This primitive normalizes a GMOS Longslit spectroscopic flatfield
        in a manner similar to that performed by gsflat in Gemini-IRAF.
        A cubic spline is fitted along the dispersion direction of each
        row, separately for each CCD.

        As this primitive is GMOS-specific, we know the dispersion direction
        will be along the rows, and there will be 3 CCDs.

        For Hamamatsu CCDs, the 21 unbinned columns at each CCD edge are
        masked out, following the procedure in gsflat.
        TODO: Should we add these in the BPM?

        Parameters
        ----------
        suffix : str/None
            suffix to be added to output files
        center : int/None
            central row/column for 1D extraction (None => use middle)
        nsum : int
            number of rows/columns around center to combine
        function : str
            type of function to fit (splineN or polynomial types)
        order : int/str
            Order of the spline fit to be performed
            (can be 3 ints, separated by commas)
        lsigma : float/None
            lower rejection limit in standard deviations
        hsigma : float/None
            upper rejection limit in standard deviations
        niter : int
            maximum number of rejection iterations
        grow : float/False
            growth radius for rejected pixels
        threshold : float
            threshold (relative to peak) for flagging unilluminated pixels
        interactive : bool
            set to activate an interactive preview to fine tune the input parameters
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        # For flexibility, the code is going to pass whatever validated
        # parameters it gets (apart from suffix and spectral_order) to
        # the spline fitter
        suffix = params["suffix"]
        threshold = params["threshold"]
        spectral_order = params["order"]
        all_fp_init = [fit_1D.translate_params(params)] * 3
        interactive_reduce = params["interactive"]

        # Parameter validation should ensure we get an int or a list of 3 ints
        try:
            orders = [int(x) for x in spectral_order]
        except TypeError:
            orders = [spectral_order] * 3
        # capture the per extension order into the fit parameters
        for order, fp_init in zip(orders, all_fp_init):
            fp_init["order"] = order

        for ad in adinputs:
            xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
            array_info = gt.array_information(ad)
            is_hamamatsu = 'Hamamatsu' in ad.detector_name(pretty=True)
            ad_tiled = self.tileArrays([ad], tile_all=False)[0]
            ad_fitted = astrodata.create(ad.phu)
            all_fp_init = []

            # If the entire row is unilluminated, we want to fit
            # the pixels but still keep the edges masked
            for ext in ad_tiled:
                try:
                    ext.mask ^= (np.bitwise_and.reduce(ext.mask, axis=1) & DQ.unilluminated)[:, None]
                except TypeError:  # ext.mask is None
                    pass
                else:
                    if is_hamamatsu:
                        ext.mask[:, :21 // xbin] = 1
                        ext.mask[:, -21 // xbin:] = 1

                all_fp_init.append(fit_1D.translate_params(params))

            # Parameter validation should ensure we get an int or a list of 3 ints
            try:
                orders = [int(x) for x in spectral_order]
            except TypeError:
                orders = [spectral_order] * 3
            # capture the per extension order into the fit parameters
            for order, fp_init in zip(orders, all_fp_init):
                fp_init["order"] = order

            # Interactive or not
            if interactive_reduce:
                # all_X arrays are used to track appropriate inputs for each of the N extensions
                all_pixels = []
                all_domains = []
                nrows = ad_tiled[0].shape[0]
                for ext, order, indices in zip(ad_tiled, orders, array_info.extensions):
                    pixels = np.arange(ext.shape[1])

                    all_pixels.append(pixels)
                    dispaxis = 2 - ext.dispersion_axis()
                    all_domains.append([0, ext.shape[dispaxis] - 1])

                config = self.params[self.myself()]
                config.update(**params)

                # This function is used by the interactive fitter to generate the x,y,weights to use
                # for each fit.  We only want to fit a single row of data interactively, so that we can
                # be responsive in the UI.  The 'row' extra parameter defined above will create a
                # slider for the user and we will have access to the selected value in the 'extras'
                # dictionary passed in here.
                def reconstruct_points(ui_params=None):
                    r = max(0, ui_params.values['row'] - 1)
                    data = {"x": [], "y": [], "weights": []}
                    for rppixels, rpext in zip(all_pixels, ad_tiled):
                        masked_data = np.ma.masked_array(rpext.data[r],
                                                         mask=None if rpext.mask is None else rpext.mask[r])
                        data["x"].append(rppixels)
                        data["y"].append(masked_data)
                        data["weights"].append(None if rpext.variance is None
                                               else np.sqrt(at.divide0(1., rpext.variance[r])))
                    return data

                if ad.filename:
                    filename_info = ad.filename
                else:
                    filename_info = ''

                # Create a 'row' parameter to add to the UI so the user can select the row they
                # want to fit.
                reinit_params = ["row", ]
                extras = {"row": RangeField("Row of data to operate on", int, int(nrows/2), min=1, max=nrows)}
                uiparams = UIParameters(config, reinit_params=reinit_params, extras=extras)
                visualizer = fit1d.Fit1DVisualizer(reconstruct_points, all_fp_init,
                                                   tab_name_fmt=lambda i: f"CCD {i+1}",
                                                   xlabel='x (pixels)', ylabel='counts',
                                                   domains=all_domains,
                                                   title="Normalize Flat",
                                                   primitive_name="normalizeFlat",
                                                   filename_info=filename_info,
                                                   enable_user_masking=False,
                                                   enable_regions=True,
                                                   help_text=NORMALIZE_FLAT_HELP_TEXT,
                                                   recalc_inputs_above=True,
                                                   modal_message="Recalculating",
                                                   ui_params=uiparams)
                geminidr.interactive.server.interactive_fitter(visualizer)
                log.stdinfo('Interactive Parameters retrieved, performing flat normalization...')

                # The fit models were done on a single row, so we need to
                # get the parameters that were used in the final fit for
                # each one, and then rerun it on the full data for that
                # extension.
                all_m_final = visualizer.results()
                for m_final, ext in zip(all_m_final, ad_tiled):
                    masked_data = np.ma.masked_array(ext.data, mask=ext.mask)
                    weights = np.sqrt(at.divide0(1., ext.variance))

                    fit1d_params = m_final.extract_params()
                    fitted_data = fit_1D(masked_data, weights=weights, **fit1d_params,
                                         axis=1).evaluate()

                    # Copy header so we have the _section() descriptors
                    ad_fitted.append(fitted_data, header=ext.hdr)
            else:
                for ext, indices, fit1d_params in zip(ad_tiled, array_info.extensions, all_fp_init):
                    masked_data = np.ma.masked_array(ext.data, mask=ext.mask)
                    weights = np.sqrt(at.divide0(1., ext.variance))

                    fitted_data = fit_1D(masked_data, weights=weights, **fit1d_params,
                                         axis=1).evaluate()

                    # Copy header so we have the _section() descriptors
                    ad_fitted.append(fitted_data, header=ext.hdr)

            # Find the largest spline value for each row across all extensions
            # and mask pixels below the requested fraction of the peak
            row_max = np.array([ext_fitted.data.max(axis=1)
                                for ext_fitted in ad_fitted]).max(axis=0)

            # Prevent runtime error in division
            row_max[row_max == 0] = np.inf

            for ext_fitted in ad_fitted:
                ext_fitted.mask = np.where(
                    (ext_fitted.data.T / row_max).T < threshold,
                    DQ.unilluminated, DQ.good).astype(DQ.datatype)

            for ext_fitted, indices in zip(ad_fitted, array_info.extensions):
                tiled_arrsec = ext_fitted.array_section()
                for i in indices:
                    ext = ad[i]
                    arrsec = ext.array_section()
                    slice_ = (slice((arrsec.y1 - tiled_arrsec.y1) // ybin, (arrsec.y2 - tiled_arrsec.y1) // ybin),
                              slice((arrsec.x1 - tiled_arrsec.x1) // xbin, (arrsec.x2 - tiled_arrsec.x1) // xbin))
                    # Suppress warnings to do with fitted_data==0
                    # (which create NaNs in variance)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        ext.divide(ext_fitted.nddata[slice_])
                    np.nan_to_num(ext.data, copy=False, posinf=0, neginf=0)
                    np.nan_to_num(ext.variance, copy=False, posinf=0, neginf=0)

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)

        return adinputs

    def slitIllumCorrect(self, adinputs=None, slit_illum=None,
                               do_cal=None, suffix="_illumCorrected"):
        """
        This primitive will divide each SCI extension of the inputs by those
        of the corresponding slit illumination image. If the inputs contain
        VAR or DQ frames, those will also be updated accordingly due to the
        division on the data.

        Parameters
        ----------
        adinputs : list of AstroData
            Data to be corrected.
        slit_illum : str or AstroData
            Slit illumination path or AstroData object.
        do_cal: str
            Perform slit illumination correction? (Default: 'procmode')
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        qecorr_key = self.timestamp_keys['QECorrect']

        if do_cal == 'skip':
            log.warning("Slit Illumination correction has been turned off.")
            return adinputs

        if slit_illum is None:
            raise NotImplementedError
        else:
            slit_illum_list = slit_illum

        # Provide a Slit Illum Ad object for every science frame
        ad_outputs = []
        for ad, slit_illum_ad in zip(*gt.make_lists(adinputs, slit_illum_list, force_ad=True)):

            if ad.phu.get(timestamp_key):
                log.warning(
                    "No changes will be made to {}, since it has "
                    "already been processed by flatCorrect".format(ad.filename))
                continue

            if slit_illum_ad is None:
                if self.mode in ['sq'] or do_cal == 'force':
                    raise OSError(
                        "No processed slit illumination listed for {}".format(
                            ad.filename))
                else:
                    log.warning(
                        "No changes will be made to {}, since no slit "
                        "illumination has been specified".format(ad.filename))
                    continue

            gt.check_inputs_match(ad, slit_illum_ad, check_shape=False)

            if not all([e1.shape == e2.shape for (e1, e2) in zip(ad, slit_illum_ad)]):
                slit_illum_ad = gt.clip_auxiliary_data(
                    adinput=[ad], aux=[slit_illum_ad])[0]

            log.info("Dividing the input AstroData object {} by this \n"
                     "slit illumination file:  \n{}".format(ad.filename, slit_illum_ad.filename))

            ad_out = deepcopy(ad)
            ad_out.divide(slit_illum_ad)

            # Update the header and filename, copying QECORR keyword from flat
            ad_out.phu.set("SLTILLIM", slit_illum_ad.filename,
                           self.keyword_comments["SLTILLIM"])

            try:
                qecorr_value = slit_illum_ad.phu[qecorr_key]
            except KeyError:
                pass
            else:
                log.fullinfo("Copying {} keyword from slit illumination".format(qecorr_key))
                ad_out.phu.set(qecorr_key, qecorr_value,
                               slit_illum_ad.phu.comments[qecorr_key])

            gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
            ad_out.update_filename(suffix=suffix, strip=True)

            if slit_illum_ad.path:
                add_provenance(ad_out, slit_illum_ad.filename,
                               md5sum(slit_illum_ad.path) or "", self.myself())

            ad_outputs.append(ad_out)

        return ad_outputs


def _split_mosaic_into_extensions(ref_ad, mos_ad, border_size=0):
    """
    Split the `mos_ad` mosaicked data into multiple extensions using
    coordinate frames and transformations stored in the `ref_ad` object.

    Right now, the pixels at the border of each extensions might not
    match the expected values. The mosaicking and de-mosaicking is an
    interpolation, because there's a small rotation. This will only interpolate,
    not extrapolate beyond the boundaries of the input data, so you lose some
    information at the edges when you perform both operations and consequently
    the edges of the input frame get lost.

    Parameters
    ----------
    ref_ad : AstroData
        Reference multi-extension-file object containing a gWCS.
    mos_ad : AstroData
        Mosaicked data that will be split containing a single extension.
    border_size : int
        Number of pixels to be trimmed out from each border.

    Returns
    -------
    AstroData : Split multi-extension-file object.

    See Also
    --------
    - :func:`gempy.library.transform.add_mosaic_wcs`
    - :func:`gempy.library.transform.resample_from_wcs`
    """
    # Check input data
    if len(mos_ad) > 1:
        raise ValueError("Expected number of extensions of `mos_ad` to be 1. "
                         "Found {:d}".format(len(mos_ad)))

    if len(mos_ad[0].shape) != 2:
        raise ValueError("Expected ndim for `mos_ad` to be 2. "
                         "Found {:d}".format(len(mos_ad[0].shape)))

    # Get original relative shift
    origin_shift_y, origin_shift_x = mos_ad[0].nddata.meta['transform']['origin']

    # Create shift transformation
    shift_x = models.Shift(origin_shift_x - border_size)
    shift_y = models.Shift(origin_shift_y - border_size)

    # Create empty AD
    ad_out = astrodata.create(ref_ad.phu)

    # Update data_section to be able to resample WCS frames
    datasec_kw = mos_ad._keyword_for('data_section')
    mos_ad[0].hdr[datasec_kw] = '[1:{},1:{}]'.format(*mos_ad[0].shape[::-1])

    # Loop across all extensions
    for i, ref_ext in enumerate(ref_ad):

        # Create new transformation pipeline
        in_frame = ref_ext.wcs.input_frame
        mos_frame = coordinate_frames.Frame2D(name="mosaic")

        mosaic_to_pixel = ref_ext.wcs.get_transform(mos_frame, in_frame)

        pipeline = [(mos_frame, mosaic_to_pixel),
                    (in_frame, None)]

        mos_ad[0].wcs = gWCS(pipeline)

        # Shift mosaic in order to set reference (0, 0) on Detector 2
        mos_ad[0].wcs.insert_transform(mos_frame, shift_x & shift_y, after=True)

        # Apply transformation
        temp_ad = transform.resample_from_wcs(
            mos_ad, in_frame.name, origin=(0, 0), output_shape=ref_ext.shape)

        # Update data_section
        datasec_kw = ref_ad._keyword_for('data_section')
        temp_ad[0].hdr[datasec_kw] = \
            '[1:{:d},1:{:d}]'.format(*temp_ad[0].shape[::-1])

        # If detector_section returned something, set an appropriate value
        det_sec_kw = ref_ext._keyword_for('detector_section')
        det_sec = ref_ext.detector_section()

        if det_sec:
            temp_ad[0].hdr[det_sec_kw] = \
                '[{}:{},{}:{}]'.format(
                    det_sec.x1 + 1, det_sec.x2, det_sec.y1 + 1, det_sec.y2)
        else:
            del temp_ad[0].hdr[det_sec_kw]

        # If array_section returned something, set an appropriate value
        arr_sec_kw = ref_ext._keyword_for('array_section')
        arr_sec = ref_ext.array_section()

        if arr_sec:
            temp_ad[0].hdr[arr_sec_kw] = \
                '[{}:{},{}:{}]'.format(
                    arr_sec.x1 + 1, arr_sec.x2, arr_sec.y1 + 1, arr_sec.y2)
        else:
            del temp_ad[0].hdr[arr_sec_kw]

        ad_out.append(temp_ad[0])

    return ad_out


@parameter_override
@capture_provenance
class GMOSNSLongslit(GMOSClassicLongslit, GMOSNodAndShuffle, Longslit):
    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_gmos_longslit)
