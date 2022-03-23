#
#                                                                  gemini_python
#
#                                                     primtives_gmos_longslit.py
# ------------------------------------------------------------------------------

from copy import deepcopy
from importlib import import_module
from itertools import groupby

import re

from gempy.library.config import RangeField

import astrodata
import geminidr
import numpy as np
from scipy.signal import correlate
from scipy.interpolate.fitpack2 import InterpolatedUnivariateSpline

from astrodata.provenance import add_provenance
from astropy.modeling import models

from geminidr.gemini.lookups import DQ_definitions as DQ

from gempy.gemini import gemini_tools as gt
from gempy.library.fitting import fit_1D
from gempy.library import transform
from gempy.library import astrotools as at

from gwcs import coordinate_frames
from gwcs.wcs import WCS as gWCS

from recipe_system.utils.decorators import parameter_override, capture_provenance
from recipe_system.utils.md5 import md5sum

from .primitives_gmos_spect import GMOSSpect
from .primitives_gmos_nodandshuffle import GMOSNodAndShuffle
from . import parameters_gmos_longslit


# ------------------------------------------------------------------------------
from ..interactive.fit import fit1d
from ..interactive.fit import bineditor
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
class GMOSClassicLongslit(GMOSSpect):
    """
    This is the class containing all of the preprocessing primitives
    for the GMOSLongslit level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_gmos_longslit)

    def addIllumMaskToDQ(self, adinputs=None, suffix=None, illum_mask=None,
                         shift=None, max_shift=20):
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
        edges = 50  # try to eliminate issues at the very edges

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
                log.warning('No changes will be made to {}, since it has '
                    'already been processed by addIllumMaskToDQ'.
                            format(ad.filename))
                continue

            ybin = ad.detector_y_bin()
            ad_detsec = ad.detector_section()
            no_bridges = all(detsec.y1 > 1600 and detsec.y2 < 2900
                             for detsec in ad_detsec)
            has_48rows = (all(detsec.y2 == 4224 for detsec in ad_detsec) and
                          'Hamamatsu' in ad.detector_name(pretty=True))

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
                    log.warning(f"MDF not found for {ad.filename} - cannot "
                                "add illumination mask.")
                    continue

                # Default operation for GMOS full-frame LS
                # Sadly, we cannot do this reliably without concatenating the
                # arrays and using a big chunk of memory.
                row_medians = np.percentile(np.concatenate(
                    [ext.data for ext in ad], axis=1),
                    95, axis=1)

                # Construct a model of the slit illumination from the MDF
                # coefficients are from G-IRAF except c0, approx. from data
                model = np.zeros_like(row_medians, dtype=int)
                for ypos, ysize in mdf['slitpos_my', 'slitsize_my']:
                    y = ypos + np.array([-0.5, 0.5]) * ysize
                    c0 = offset_dict[ad.instrument(), ad.detector_name(pretty=True)]
                    if ad.instrument() == "GMOS-S":
                        c1, c2, c3 = (0.99911, -1.7465e-5, 3.0494e-7)
                    else:
                        c1, c2, c3 = (0.99591859227, 5.3042211333437e-8,
                                      1.7447902551997e-7)
                    yccd = ((c0 + y * (c1 + y * (c2 + y * c3))) *
                            1.611444 / ad.pixel_scale() + 0.5 * model.size).astype(int)
                    model[yccd[0]:yccd[1]+1] = 1
                    log.stdinfo("Expected slit location from pixels "
                                 f"{yccd[0]+1} to {yccd[1]+1}")

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

                if shift is None:
                    mshift = max_shift // ybin + 2
                    mshift2 = mshift + edges
                    # model[] indexing avoids reduction in signal as slit
                    # is shifted off the top of the image
                    cntr = model.size - edges - mshift2 - 1
                    xcorr = correlate(row_medians[edges:-edges], model[mshift2:-mshift2],
                                      mode='full')[cntr - mshift:cntr + mshift]
                    # This line avoids numerical errors in the spline fit
                    xcorr -= np.median(xcorr)

                    # This calculates the offsets of each point from the
                    # straight line between its neighbours
                    std = (xcorr[1:-1] - 0.5 *
                           (xcorr + np.roll(xcorr, 2))[2:]).std()
                    xspline = fit_1D(xcorr, function="spline3", order=None,
                                     weights=np.full(len(xcorr), 1. / std)).evaluate()
                    yshift = xspline.argmax() - mshift
                    maxima = xspline[1:-1][np.logical_and(np.diff(xspline[:-1]) > 0,
                                                          np.diff(xspline[1:]) < 0)]
                    significant_maxima = (maxima > xspline.max() - 3 * std).sum()
                    if significant_maxima > 1 or abs(yshift // ybin) > max_shift:
                        log.warning(f"{ad.filename}: cross-correlation peak is"
                                    " untrustworthy so not adding illumination "
                                    "mask. Please re-run with a specified shift.")
                        yshift = None
                else:
                    yshift = shift

                if yshift is not None:
                    log.stdinfo(f"{ad.filename}: Shifting mask by {yshift} pixels")
                    row_mask = np.ones_like(model, dtype=int)
                    if yshift < 0:
                        row_mask[:yshift] = 1 - model[-yshift:]
                    elif yshift > 0:
                        row_mask[yshift:] = 1 - model[:-yshift]
                    else:
                        row_mask[:] = 1 - model
                    for ext in ad:
                        ext.mask |= (row_mask * DQ.unilluminated).astype(
                            DQ.datatype)[:, np.newaxis]

            if has_48rows:
                actual_rows = 48 // ybin
                for ext in ad:
                    ext.mask[:actual_rows] |= DQ.unilluminated

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)

        return adinputs

    def makeSlitIllum(self, adinputs=None, **params):
        """
        Makes the processed Slit Illumination Function by binning a 2D
        spectrum in wavelength, averaging data points withing each bin along dispersion
        axis, then fitting a smooth function to each bin in spatial direction.
        Each fitted function is then normalized to slit center, and a 2D slit illumination
        function is created by interpolating between dispersion points for each row.

        The implementation is based on the IRAF's `noao.twodspec.longslit.illumination`
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
            have been overscan and bias corrected and it is expected to have a
            Data Quality mask.
        bins : int/str/None
            Either an integer number of equally-spaced bins covering the whole dispersion
            range, or a comma-separated list of pixel coordinate pairs defining the
            dispersion bins to be used for the slit profile fitting. If None, the number
            of bins is max of 12 or the number of extensions.
        regions : str/None
            Sample region(s) selected along the spatial axis to use for fitting each dispersion
            bin, as a comma-separated list of pixel ranges. Any pixels outside these
            ranges will be ignored when fitting each dispersion bin.
        function : str/None
            Type of function to fit along the the spatial axis (for dispersion bin fitting).
            None => "spline3".
        order : int/None
            Order of the bin fitting function (None => 20)
        lsigma : float/None
            Lower rejection limit in standard deviations of the bin fit (None => 3).
        hsigma : float/None
            Upper rejection limit in standard deviations of the bin fit (None => 3).
        niter : int/None
            Maximum number of rejection iterations of the bin fit (None => 3).
        grow : float/None
            Growth radius for rejected pixels of the bin fit (None => 0).
        interp_order: int/None
            Order of the spline interpolator (1 <= interp_order <= 5). None => 3.
        debug_boundary_ext: bool/None
            Controls the extrapolation mode for the elements outside the interpolation interval
            (before the first bin center and after the last bin center). If set to "False" (default
            value), the row ends are set to constant values equal to the interpolation boundary values.
            If set to "True", the row ends are extrapolated.
        interactive : bool/None
            Activate the interactive preview of bin selection and bin fitting steps to fine tune the
            input parameters (None => "False")
        border : int/None
            Border size that is added on every edge of the slit illumination
            image before cutting it down to the input AstroData frame (None => 2)

        Return
        ------
        List of AstroData :
            Contains an AstroData with the Slit Illumination Response Function for each
            of the input object.

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
        interactive_reduce = params["interactive"]
        spat_params = fit_1D.translate_params(params)
        interp_order = params["interp_order"]
        boundary_ext = params["debug_boundary_ext"]
        ad_outputs = []

        for ad in adinputs:
            if len(ad) > 1 and "mosaic" not in ad[0].wcs.available_frames:

                log.info('Add "mosaic" gWCS frame to input data')
                geotable = import_module('.geometry_conf', self.inst_lookups)

                # deepcopy prevents modifying input `ad` inplace
                ad = transform.add_mosaic_wcs(deepcopy(ad), geotable)

                log.info("Temporarily mosaicking multi-extension file")
                mosaicked_ad = transform.resample_from_wcs(
                    ad, "mosaic", attributes=None, order=1, process_objcat=False)
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
                bin_list = list(zip(bin_limits[:-1], bin_limits[1:]))
            elif isinstance(bins, int):
                nbins = bins
                bin_limits = np.linspace(0, height, nbins + 1, dtype=int)
                bin_list = list(zip(bin_limits[:-1], bin_limits[1:]))
            else:
                bin_list = _parse_user_bins(bins, height)
                nbins = len(bin_list)

            if ad.filename:
                filename_info = ad.filename
            else:
                filename_info = ''

            bin_parameters = {
                'nbins': nbins,
                'bin_list': bin_list,
                'height': height
            }
            # Interactive interface for bin inspection and editing
            if interactive_reduce:
                model = {
                    'x': np.arange(height),
                    'y': np.ma.mean(data, axis=1),
                    'regions': [(left, right) for (left, right) in
                           bin_list]
                }
                visualizer = bineditor.BinVisualizer(model, domain=[0, height],
                                                     title='Set Dispersion Bins',
                                                     primitive_name='makeSlitIllum',
                                                     central_plot=False,
                                                     bin_parameters=bin_parameters,
                                                     filename_info=filename_info)
                geminidr.interactive.server.interactive_fitter(visualizer)
                bin_list = _parse_user_bins(''.join(visualizer.results().split()))
                nbins = len(bin_list)

            cols_val = np.arange(-border, height+border)
            rows_val = np.arange(-border, width+border)
            binned_shape = (nbins, len(rows_val))
            bin_data_avg = np.ma.empty((nbins, width))
            bin_std_avg = np.ma.empty((nbins, width))
            bin_data_fits = np.ma.zeros(binned_shape)
            bin_std_fits = np.ma.zeros(binned_shape)
            for i, bin in enumerate(bin_list):
                bin_data_avg[i] = np.ma.mean(data[bin[0]:bin[1]], axis=0)
                bin_std_avg[i] = np.ma.mean(std[bin[0]:bin[1]], axis=0)
            spat_fitting_pars = []
            for _ in range(nbins):
                spat_fitting_pars.append(spat_params)
            spat_fit_points = np.arange(width)

            log.info("Smooth binned data and variance, and normalize them by "
                     "smoothed central value")

            config = self.params[self.myself()]

            # Interactive interface for fitting dispersion bins
            if interactive_reduce:
                all_pixels = []
                all_domains = []
                for _ in range(nbins):
                    all_pixels.append(spat_fit_points)
                    all_domains.append([-border, width + border])

                data_with_weights = {"x": [], "y": [], "weights": []}
                for rppixels, avg_data, avg_std in zip(all_pixels, bin_data_avg, bin_std_avg):
                    data_with_weights["x"].append(rppixels)
                    data_with_weights["y"].append(avg_data)
                    data_with_weights["weights"].append(at.divide0(1., avg_std))

                config.update(**params)
                uiparams = UIParameters(config)

                x_label = "Rows" if dispaxis == 1 else "Columns"
                first_label = 'cols' if dispaxis == 1 else 'rows'
                tab_names = [f'[{start+1}:{end}]' for (start, end) in bin_list]
                tab_names[0] = f"Mean of {first_label} {tab_names[0]}"

                visualizer = fit1d.Fit1DVisualizer(data_with_weights, spat_fitting_pars,
                                                   tab_names=tab_names,
                                                   xlabel=x_label, ylabel='Counts',
                                                   domains=all_domains,
                                                   title="Make Slit Illumination Function",
                                                   primitive_name="makeSlitIllum",
                                                   filename_info=filename_info,
                                                   enable_user_masking=True,
                                                   enable_regions=True,
                                                  # help_text=NORMALIZE_FLAT_HELP_TEXT,
                                                   recalc_inputs_above=True,
                                                   modal_message="Recalculating",
                                                   pad_buttons=True,
                                                   ui_params=uiparams)
                geminidr.interactive.server.interactive_fitter(visualizer)
                fits = visualizer.results()
                spat_fitting_pars = [fit.extract_params() for fit in fits]
                for bin_idx, bin_fit in enumerate(fits):
                    bin_data_fit = bin_fit.evaluate(rows_val)
                    bin_data_fits[bin_idx,:] = bin_data_fit

            else:
                for bin_idx, (fit_params, avg_data, avg_std) in \
                        enumerate(zip(spat_fitting_pars, bin_data_avg, bin_std_avg)):
                    bin_data_fit = fit_1D(avg_data,
                                          points=spat_fit_points,
                                          weights=at.divide0(1., avg_std),
                                          domain=(-border, width+border),
                                          **fit_params, axis=0).evaluate(rows_val)
                    bin_data_fits[bin_idx,:] = bin_data_fit

            for bin_idx, (fit_params, avg_std, bin_data_fit) in \
                    enumerate(zip(spat_fitting_pars, bin_std_avg, bin_data_fits)):
                bin_std_fit = fit_1D(avg_std,
                                     points=spat_fit_points,
                                     **fit_params,
                                     axis=0).evaluate(rows_val)
                slit_central_value = bin_data_fit[(width+2*border) // 2]
                bin_data_fits[bin_idx,:] = bin_data_fit / slit_central_value
                bin_std_fits[bin_idx,:] = bin_std_fit / slit_central_value

            log.info("Reconstruct 2D mosaicked data")

            bin_center = np.array([0.5 * (bin_start + bin_end) for (bin_start, bin_end) in bin_list],
                                  dtype=int)
            slit_response_data = np.zeros((len(cols_val), len(rows_val)))
            slit_response_std = np.zeros((len(cols_val), len(rows_val)))

            # Interpolation between dispersion points along the rows
            if nbins > 1:
                for k, (data_row, std_row) in enumerate(zip(bin_data_fits.T, bin_std_fits.T)):
                    # Set extrapolated row ends to interpolation boundary value, or extrapolate
                    ext_val = 0 if boundary_ext else 3
                    f1 = InterpolatedUnivariateSpline(bin_center, data_row, k=interp_order, ext=ext_val)
                    f2 = InterpolatedUnivariateSpline(bin_center, std_row, k=interp_order, ext=ext_val)
                    slit_response_data[:,k] = f1(cols_val)
                    slit_response_std[:,k] = f2(cols_val)

            # If there is only one bin, copy the slit profile to each column
            elif nbins == 1:
                slit_response_data[:] = bin_data_fits
                slit_response_std[:] = bin_std_fits

            slit_response_var = slit_response_std ** 2
            slit_response_mask = np.pad(mask, border, mode='edge')

            _data, _mask, _variance = at.transpose_if_needed(
                slit_response_data, slit_response_mask, slit_response_var,
                transpose=dispaxis == 1)

            log.info("Update slit response data and data_section")
            slit_response_ad = deepcopy(mosaicked_ad)
            slit_response_ad[0].data = _data
            slit_response_ad[0].mask = _mask
            slit_response_ad[0].variance = _variance

            if "mosaic" in ad[0].wcs.available_frames:

                log.info("Map coordinates between slit function and mosaicked data") # ToDo: Improve message?
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
                                                   tab_name_fmt="CCD {}",
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
                    np.nan_to_num(ext.variance, copy=False)

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


            log.stdinfo(f"{ad.filename}: dividing by the slit illumination function "
                         f"{slit_illum_ad.filename}")

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

def _parse_user_bins(bins, frame_size:int=None):
    """
    Parse a string of bin ranges containing a comma-separated list of colon- or hyphen-separated
    pixel sections into a list of Python tuples. Arrange the bins in ascending order, remove the ones outside
    the frame pixel range, merge the overlapping bins.

    Parameters
    ----------
    bins: str
        Comma-separated list of colon- or hyphen-separated pixel ranges
    frame_size: int
        The size of the frame dimension along which the bins are selected.

    Returns
    -------
    A sorted list of 2-value tuples lying within the specified frame range, with no overlapping.
    """
    bin_list = []
    for bin in re.split(",|;| ", bins.strip("[]()'")):
        bin = bin.strip("()[]' ")
        bin_limits = re.split(":|-", bin)
        if not len(bin_limits) == 2:
            raise TypeError("Bin limits must be specified as comma-separated list "
                            "of colon- or hyphen-separated pixel sections, e.g. 1:300,301:500")
        int_bin_limits = []
        for bin_limit in bin_limits:
            try:
                int_bin_limit = int(bin_limit)
            except ValueError:
                raise TypeError("Bin ranges must be integer")
            if frame_size is not None:
                int_bin_limits.append(min(int_bin_limit, frame_size))
            else:
                int_bin_limits.append(int_bin_limit)
        int_bin_limits.sort()

        # trim off the bins that are outside the frame pixel range
        if frame_size is not None and int_bin_limits[0] == frame_size:
            break

        bin_list.append(tuple(int_bin_limits))
    bin_list = sorted(bin_list, key=lambda tup: tup[0])

    # merge overlapping bins
    merged_list = [bin_list[0]]
    for bin in bin_list[1:]:
        last = merged_list[-1]
        if last[1] > bin[0]:
            print("Merging the overlapping bins")
            merged_list[-1] = (last[0], max(last[1], bin[1]))
        elif last[1] == bin[0]:
            merged_list.append((bin[0]+1, bin[1]))
        else:
            merged_list.append(bin)

    adjusted_bin_list = []
    for i, bin in enumerate(merged_list):
        if i == 0 and bin[0] == 0:
            adjusted_bin_list.append(bin)
        else:
            adjusted_bin_list.append((bin[0]-1, bin[1]))
    return adjusted_bin_list


@parameter_override
@capture_provenance
class GMOSNSLongslit(GMOSClassicLongslit, GMOSNodAndShuffle):
    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_gmos_longslit)
