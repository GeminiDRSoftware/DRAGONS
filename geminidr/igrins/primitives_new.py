import os
from importlib import import_module
from bisect import bisect_left
import copy
import warnings

from matplotlib import pyplot as plt, colors as mcolors

from scipy.interpolate import make_interp_spline, PPoly
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate, savgol_filter

import numpy as np
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.table import Table
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning

from gwcs.wcs import WCS as gWCS
from gwcs import coordinate_frames as cf

import astrodata
from astrodata import wcs as adwcs
from gempy.gemini import gemini_tools as gt
from gempy.library import astromodels as am, astrotools as at
from gempy.library.fitting import fit_1D
from gempy.library import peak_finding, tracing, transform, wavecal

from geminidr.gemini.lookups import DQ_definitions as DQ
from gempy.adlibrary.manipulate_ad import reassemble_ad

from .primitives_igrins import IGRINS
from ..core.primitives_crossdispersed import CrossDispersed
from ..core.primitives_telluric import Telluric
from .cheb3d import Chebyshev3D, LSQFitterWithOutlierRemoval3D

from recipe_system.utils.decorators import parameter_override
from . import parameters_new

from .procedures.apertures import Apertures
from .procedures.correct_distortion import get_rectified_2dspec
from .procedures.flexure_correction import estimate_flexure
from .procedures.iraf_helper import get_wat_header
from .procedures.readout_pattern.readout_pattern_helper import remove_pattern


# Function to scale from SLITPOS to y-pixel location
slitpos_to_pix = models.Shift(0.5) | models.Scale(50)


@parameter_override
class IGRINSNew(IGRINS, Telluric, CrossDispersed):
    tagset = {}

    def _initialize(self, adinputs=None, **kwargs):
        self.inst_lookups = 'geminidr.igrins.lookups'
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_new)

    def _get_ad_flat(self, ad):
        return NotImplementedError("_get_ad_flat not implemented")

    def _get_ad_sky(self, ad):
        return NotImplementedError("_get_ad_sky not implemented")

    def maskReferencePixels(self, adinputs=None, **params):
        """Mask reference pixels as unilluminated."""
        for ad in adinputs:
            ad[0].mask[:4] = DQ.unilluminated
            ad[0].mask[-4:] = DQ.unilluminated
            ad[0].mask[:, :4] = DQ.unilluminated
            ad[0].mask[:, -4:] = DQ.unilluminated
        return adinputs

    def addMDF(self, adinputs=None, suffix=None, mdf=None):
        """
        This IGRINS2-specific implementation of addMDF() adds a "virtual MDF"
        (as in, created from data in this module rather than pulled from another
        file) to each IGRINS-2 frame.

        Parameters
        ----------
        suffix : str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        # Number of orders, value of lowest order, x pixel location, then
        # Polynomial1D coefficients for the slit midpoints for each order
        # and Polynomial1D coefficients for the slit lengths.
        order_location_dict = {'H': [26, 98, 800, (58.78, 110.29, -1.63, 0.016), (60.855, -0.716, 0.013)],
                               'K': [24, 70, 700, (8.14, 125.88, -2.30, 0.027), (60.076, -0.627, 0.013)]}

        for ad in adinputs:
            norders, low_order, x_ccd, coeffs1, coeffs2 = order_location_dict[ad.band()]
            order_poly = models.Polynomial1D(degree=len(coeffs1)-1,
                                             **{f"c{i}": c for i, c in enumerate(coeffs1)})
            y_ccd = order_poly(np.arange(norders))
            order_poly = models.Polynomial1D(degree=len(coeffs2)-1,
                                             **{f"c{i}": c for i, c in enumerate(coeffs2)})
            slitlen_pix = order_poly(np.arange(norders))

            # This coding means we don't need to turn the x_ccd column into a
            # list/array before adding it
            mdf_table = Table([range(1, len(y_ccd) + 1)], names=['slit_id'])
            mdf_table['x_ccd'] = x_ccd
            mdf_table['y_ccd'] = y_ccd
            mdf_table['specorder'] = mdf_table['slit_id'] + low_order - 1
            mdf_table['slitlength_pixels'] = slitlen_pix
            mdf_table['slitlength_asec'] = 5.0
            ad.MDF = mdf_table
            log.stdinfo(f"Adding MDF table for {ad.filename}")

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def applySlitModel(self, adinputs=None, **params):
        """
        This performs the standard applySlitModel operation, and then modifies the
        'rectified' transform to have the spatial axis be in "slitpos units"
        (i.e., extend from -0.5 to +0.5) instead of pixel units.

        We also have to modify the SKY model to account for this new origin,
        to ensure that the overall WCS is unchanged.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        adinputs = super().applySlitModel(adinputs, **params)
        #adinputs[0].write(adinputs[0].filename.replace("_slitModelApplied", "_intermediate"), overwrite=True)
        x = np.arange(2048)
        for ad in adinputs:
            slitlengths_asec = dict(zip(ad.MDF['specorder'],
                                        ad.MDF['slitlength_asec']))
            for ext, spec_order in zip(ad, ad.hdr['SPECORDR']):
                edge_models = [am.table_to_model(row) for row in ext.SLITEDGE]
                t = ext.wcs.get_transform('pixels', 'rectified')
                y1 = edge_models[0](x)
                y2 = edge_models[1](x)
                center = 0.5 * np.mean(t(x, y2)[1] + t(x, y1)[1])
                width = np.mean(t(x, y2)[1] - t(x, y1)[1])  # of rectified slit
                m = models.Identity(1) & (models.Shift(-center) | models.Scale(1. / width))
                slitpos_frame = cf.Frame2D(name="slitpos", axes_names=("x", "slitpos"),
                                           unit=(u.pix, u.dimensionless_unscaled))
                ext.wcs.insert_frame("rectified", m, slitpos_frame)
                wave_model = am.get_named_submodel(ext.wcs.forward_transform, "WAVE")
                sky_model = am.get_named_submodel(ext.wcs.forward_transform, "SKY")
                current_pixscale = 3600 * np.sqrt(np.linalg.det(sky_model[-3].matrix))
                sky_model[-3].matrix *= slitlengths_asec[spec_order] / current_pixscale  # slit is 5 arcsec
                # We then want to remove all shifts (there'll be the CRPIX2
                # shift, and a second shift from the origin reset after cutting).
                # We do it like this because the CD matrix may have been rotated
                # by 90 degrees if saved as FITS and read back in.
                for m in sky_model:
                    if m.__class__ == models.Shift:
                        m.offset = 0
                ext.wcs.set_transform("slitpos", "world", wave_model & sky_model)

                # super().applySlitModel() will have done the housekeeping

        return adinputs

    def attachWavelengthSolution(self, adinputs=None, **params):
        """
        Attach the distortion map (a Chebyshev2D model) and the mapping from
        distortion-corrected pixels to wavelengths (a Chebyshev1D model, when
        available after successful line matching) from a processed arc, or
        similar wavelength reference, to the WCS of the input data.

        This IGRINS-2 version of the primitive has to do some clean-up of the
        gWCS object because the slit-cutting inserts a shift in the spatial
        (vertical) direction to keep the astrometry of the orders consistent.
        However, this shift is inserted in units of raw pixels but the
        distortion model includes a conversion to uniformly-sized pixels and
        so it's incorrect. Given the way the gWCS has been constructed, we
        can just simplify the sky model.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            2D spectral images.
        suffix : str
            Suffix to be added to output files.
        arc : :class:`~astrodata.AstroData` or str or None
            Arc(s) containing distortion map & wavelength calibration.
        use_same_arc : bool
            Require the use of the same arc for all frames? If True and
            more than one arc is returned from the CalDB, then the most
            commonly-chosen arc is used for all frames.
        """
        log = self.log
        super().attachWavelengthSolution(adinputs, **params)
        for ad in adinputs:
            flex_shift = ad.phu.get('FLEXSHFT', 0)
            for ext in ad:
                wave_model = am.get_named_submodel(ext.wcs.forward_transform, "WAVE")
                sky_model = am.get_named_submodel(ext.wcs.forward_transform, "SKY")
                # This is the AffineTransform2D, Pix2Sky, RotateNative2Celestial
                new_sky_model = (models.Mapping((0, 0)) |
                                 (models.Const1D(0) & models.Identity(1)) |
                                 sky_model[-3:])
                new_sky_model.name = "SKY"

                # Apply the flexure shift if there is one
                if flex_shift:
                    log.stdinfo(f"Applying flexure shift of {flex_shift:.3f} "
                                f"pixels to {ad.filename}")
                    cheb = np.polynomial.chebyshev.Chebyshev(wave_model.parameters, np.asarray(wave_model.domain) + flex_shift)
                    coef = {f'c{i}': v for i, v in enumerate(cheb.convert(domain=wave_model.domain).coef)}
                    new_wave_model = wave_model.__class__(degree=wave_model.degree, **coef, domain=wave_model.domain,
                                                          name="WAVE")
                else:
                    new_wave_model = wave_model

                ext.wcs.set_transform("distortion_corrected", "world", new_wave_model & new_sky_model)

        # Timestamping/housekeeping was handled by the super() call
        return adinputs

    def cleanReadout(self, adinputs=None, **params):
        """
        Runs the IGRINSDR code.

        Parameters
        ----------
        suffix: str
            Suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        flat = params["flat"]
        remove_level = params["remove_level"]
        remove_amp_wise_var = params["remove_amp_wise_var"]

        if flat is None:
            flat_list = self.caldb.get_processed_flat(adinputs)
        else:
            flat_list = (flat, None)

        flat_masks = {}
        for ad, flat, origin in zip(*gt.make_lists(adinputs, *flat_list,
                                    force_ad=(1,))):
            try:
                mask = flat_masks[flat.filename]
            except KeyError:
                mask = reassemble_ad(flat, shape=ad[0].shape)[0].mask
                flat_masks[flat.filename] = mask

            # Do the division
            origin_str = f" (obtained from {origin})" if origin else ""
            log.stdinfo(f"{ad.filename}: using the mask from the flat "
                         f"{flat.filename}{origin_str}")
            # "mask" is the good pixels
            ad[0].data = remove_pattern(ad[0].data, mask=(mask & DQ.unilluminated == 0),
                                        remove_level=remove_level,
                                        remove_amp_wise_var=remove_amp_wise_var)
            # Why is the mask not passed here?
            ad[0].variance = remove_pattern(ad[0].variance, remove_level=1,
                                            remove_amp_wise_var=False)

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def correctFlexure(self, adinputs=None, **params):
        """
        Correct the flexure. This can be skipped using the
        "skip_primitive" recipe-level parameter.

        Weirdly though, the main code calls estimate_flexure if
        correct_flexure=False!?

        Parameters
        ----------
        suffix: str
            Suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        #timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]

        ad_sky = self._get_ad_sky(adinputs[0])
        adinputs = estimate_flexure(adinputs, ad_sky, adinputs[0].exposure_time())
        for ad in adinputs:
            # Timestamp and update the filename
            #gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def determineDistortion(self, adinputs=None, **params):
        """
        Maps the distortion on a detector by tracing lines perpendicular to the
        dispersion direction. Then it fits a 2D Chebyshev polynomial to the
        fitted coordinates in the dispersion direction. The distortion map does
        not change the coordinates in the spatial direction.

        The Chebyshev2D model is stored as part of a gWCS object in each
        `nddata.wcs` attribute, which gets mapped to a FITS table extension
        named `WCS` on disk.


        Parameters
        ----------
        suffix :  str
            Suffix to be added to output files.
        spatial_order : int
            Order of fit in spatial direction.
        spectral_order : int
            Order of fit in spectral direction.
        id_only : bool
            Trace using only those lines identified for wavelength calibration?
        min_snr : float
            Minimum signal-to-noise ratio for identifying lines (if
            id_only=False).
        nsum : int
            Number of rows/columns to sum at each step.
        step : int
            Size of step in pixels when tracing.
        max_shift : float
            Maximum orthogonal shift (per pixel) for line-tracing (unbinned).
        max_missed : int
            Maximum number of steps to miss before a line is lost.
        min_line_length: float
            Minimum length of traced feature (as a fraction of the tracing dimension
            length) to be considered as a useful line.
        debug_reject_bad: bool
            Reject lines with suspiciously high SNR (e.g. bad columns)?
        debug: bool
            plot arc line traces on image display window?
        debug_min_points_per_trace: int
            minimum number of points required for a trace to be considered
            valid
        debug_min_relative_peak_height: float
            minimum height of a peak relative to the its initial value during
            the tracing
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        disp_order = params["dispersion_order"]
        spat_order = params["spatial_order"]
        xd_order = params["xdorder_order"]
        id_only = params["id_only"]
        fwidth = params["fwidth"]
        min_snr = params["min_snr"]
        nsum = params["nsum"]
        step = params["step"]
        max_shift = params["max_shift"]
        max_missed = params["max_missed"]
        min_line_length = params["min_line_length"]
        debug_reject_bad = params["debug_reject_bad"]
        debug = params["debug"]
        min_points = params.get("debug_min_points_per_trace", 0)
        min_relative_height = params.get("debug_min_relative_peak_height", 0.)
        xpix = np.arange(2048)

        for ad in adinputs:
            slitlengths_asec = dict(zip(ad.MDF['specorder'],
                                        ad.MDF['slitlength_asec']))

            data_to_fit = []
            xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
            nlines = 0
            for ext, spec_order in zip(ad, spec_orders := ad.hdr['SPECORDR']):
                dispaxis = 2 - ext.dispersion_axis()  # python sense
                direction = "row" if dispaxis == 1 else "column"
                peak_to_centroid_func = self._convert_peak_to_centroid(ext)
                # The peak-to-centroid function always takes the dispersion
                # coordinate as its first argument, but in_coords and
                # ref_coords are always (x, y), so provide a single interface
                # to handle both orientations
                if dispaxis == 0:
                    convert_to_centroid = lambda x, y: (x, peak_to_centroid_func(y, x))
                else:
                    convert_to_centroid = lambda x, y: (peak_to_centroid_func(x, y), y)

                # Here's a lot of input-checking
                start = ext.shape[1 - dispaxis] // 2

                # This is identical to the code in determineWavelengthSolution()
                if fwidth is None:
                    data, mask, variance, extract_info = peak_finding.average_along_slit(
                        ext, center=start, nsum=nsum)
                    fwidth = peak_finding.estimate_peak_width(data, boxcar_size=30)
                    log.stdinfo(f"Estimated feature width: {fwidth:.2f} pixels")

                data, mask, variance, extract_info = peak_finding.average_along_slit(
                    ext, center=start, nsum=nsum)
                coeffs = [f"{key}: {value:.2f}" for key, value in
                          zip(extract_info.param_names,
                              extract_info.parameters)]
                log.stdinfo(f"Extracting 1D spectrum for order {spec_order}")
                log.stdinfo(f"  ±{nsum / 2:.1f} {direction}s "
                            "around polynomial with " + ", ".join(coeffs))

                initial_peaks, peak_values, _ = peak_finding.find_wavelet_peaks(
                    data, fwidth=fwidth, mask=mask & DQ.not_signal,
                    variance=variance, min_snr=min_snr, reject_bad=debug_reject_bad,
                    pinpoint_index=-1)

                if len(initial_peaks):
                    rwidth = peak_finding.ricker_widths(fwidth)[-1]
                    slit_edges = [am.table_to_model(row) for row in ext.SLITEDGE]
                    slit_length = np.mean(slit_edges[1](xpix) - slit_edges[0](xpix))
                    slit_length_frac = slit_length / ext.shape[1 - dispaxis]

                    traces = []
                    for peak, peak_value in zip(initial_peaks, peak_values):
                        # Need to start midway along the slit, which varies
                        # along the dispersion axis. `extract_info` here is the
                        # polynomial describing that midway line.
                        start = extract_info(peak)
                        try:
                            traces.extend(tracing.trace_lines(
                                ext, axis=1 - dispaxis,
                                start=start, initial=[peak],
                                rwidth=rwidth, halfwidth=max(fwidth // 2, 2), step=step,
                                nsum=nsum, max_missed=max_missed,
                                initial_tolerance=None,
                                max_shift=max_shift * ybin / xbin,
                                viewer=self.viewer if debug else None,
                                min_line_length=min_line_length * slit_length_frac,
                                min_peak_value=min_relative_height * peak_value))
                        except ValueError:  # too close to edge
                            pass

                    # Remove traces with too few points
                    traces = [trace for trace in traces if len(trace) >= min_points]
                    log.stdinfo(f"Traced {len(traces)} lines from "
                                f"{len(initial_peaks)} peaks")

                    # List of traced peak positions
                    in_coords = np.array([coord for trace in traces for
                                          coord in trace.input_coordinates()]).T
                    if in_coords.size == 0:
                        # Don't need a warning that there are no traces
                        continue

                else:
                    # Don't need a warning that there were no peaks
                    continue

                # List of "reference" positions (i.e., the coordinate
                # perpendicular to the line remains constant at its initial value
                ref_coords = np.array([coord for trace in traces for
                                       coord in trace.reference_coordinates()]).T

                # Convert all coordinates from peaks to centroids
                in_coords = np.asarray(convert_to_centroid(*in_coords))
                ref_coords = np.asarray(convert_to_centroid(*ref_coords))

                shift = ref_coords[1-dispaxis] - in_coords[1-dispaxis]
                slitpos = ext.wcs.get_transform('pixels', 'slitpos')(*in_coords)[dispaxis]
                data_to_fit.extend(list(zip(*[ref_coords[1-dispaxis], slitpos,
                                              [spec_order] * len(shift), shift])))
                nlines += len(traces)

            xref, slitpos, order, shift = np.asarray(data_to_fit).T
            linefit = Table([xref, slitpos, order, -shift],
                            names=["initial_mean_pixel", "slit_center",
                                   "order", "offset"],
                            dtype=[np.float32, np.float32, int, np.float32])
            ad.LINEFIT = linefit
            log.stdinfo(f"Fitting {nlines} lines across all orders")
            good = np.logical_and(abs(slitpos) < 0.5, abs(slitpos) > 0.001)
            xref = xref[good]
            slitpos = slitpos[good]
            order = order[good]
            shift = shift[good]

            # As with determineDistortion(), we fit to the shift
            fit_it = LSQFitterWithOutlierRemoval3D(outlier_func=sigma_clip,
                                                   niter=3)
            m_init = Chebyshev3D(x_degree=disp_order, y_degree=spat_order,
                                 z_degree=xd_order,
                                 x_domain=(0, 2047), y_domain=(-0.5, 0.5),
                                 z_domain=(min(spec_orders), max(spec_orders)))
            # Set all coefficients that do not have a slitpos term to be zero
            # as we want the model to produce no shift at the slit centre
            for p in m_init.param_names:
                if p[3] == "0":  # Chebyshev order of spatial component
                    getattr(m_init, p).fixed = True

            # Compute 3D polynomial from x->x' and its inverse
            # Map from the actual pixel location to the reference (vertical)
            m_final, fwd_mask = fit_it(m_init, xref-shift, slitpos, order, shift)
            # Put in the linear term so this is a proper coordinate transformation
            m_final.c0_0_0 = m_final.c1_0_0 = 1023.5
            fwd_rms = np.std((m_final(xref-shift, slitpos, order) - xref)[~fwd_mask])
            niter = fit_it.fit_info['niter']
            log.stdinfo(f"Forward rms={fwd_rms:.3f} pixels with "
                        f"{fwd_mask.sum()}/{xref.size} outliers in {niter} iterations")
            # And the reverse
            m_inverse, inv_mask = fit_it(m_init, xref, slitpos, order, -shift)
            m_inverse.c0_0_0 = m_inverse.c1_0_0 = 1023.5
            inv_rms = np.std((m_inverse(xref, slitpos, order) - (xref-shift))[~inv_mask])
            niter = fit_it.fit_info['niter']
            log.stdinfo(f"Inverse rms={inv_rms:.3f} pixels with "
                        f"{inv_mask.sum()}/{xref.size} outliers in {niter} iterations")

            volfit = Table([m_final.param_names, m_final.parameters,
                            m_inverse.parameters],
                           names=["parameter", "forward", "inverse"])
            ad.VOLFIT = volfit

            # Now add a 2D model to each extension's gWCS, which also
            # converts the y-coordinate to uniformly-sized pixels.
            for ext, order in zip(ad, spec_orders):
                slitlen_pix = slitlengths_asec[order] / ext.pixel_scale()
                pixscale = ext.pixel_scale()
                model = models.Mapping((0, 1, 1)) | (
                        am.reduce_dimensionality(m_final, z=order) &
                        models.Identity(1))
                model.inverse = models.Mapping((0, 1, 1)) | (
                        am.reduce_dimensionality(m_inverse, z=order) &
                        models.Identity(1))
                try:
                    frame_index = ext.wcs.available_frames.index("distortion_corrected")
                except ValueError:
                    pass
                else:
                    log.warning("Deleting existing distortion model in "
                                f"{ad.filename} order {spec_order}")
                    ext.wcs = ext.wcs.__class__(
                        ext.wcs.pipeline[:frame_index - 1] +
                        [(ext.wcs.pipeline[frame_index - 1].frame,
                          ext.wcs.pipeline[frame_index].transform)] +
                        ext.wcs.pipeline[frame_index + 1:]
                    )

                # We form a chain of coordinate frames: pixels -> rectified ->
                # slitpos -> distcorr_slitpos -> distortion_corrected. The
                # "distcorr_slitpos" frame is in (corrected_x, slitpos) units.
                # This may be useful later.
                distcorr_inter_frame = copy.copy(ext.wcs.slitpos)
                distcorr_inter_frame.name = "distcorr_slitpos"
                distcorr_frame = cf.Frame2D(name="distortion_corrected")
                ext.wcs.insert_frame('slitpos', model, distcorr_inter_frame)
                ext.wcs.insert_frame(distcorr_inter_frame.name,
                                     models.Identity(1) & models.Scale(slitlen_pix),
                                     distcorr_frame)

                # And update the "SKY" model to still give output coordinates
                # in arcseconds; the "distortion_corrected" frame is in
                # uniformly-sized pixels.
                ext.wcs = am.replace_submodel_in_gwcs(ext.wcs, "SKY",
                                                      models.Scale(pixscale))

                t = ext.wcs.get_transform("distcorr_slitpos", "distortion_corrected")

            del ad.MDF
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def determineWavelengthSolution(self, adinputs=None, **params):
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        nsum = params["nsum"]
        min_snr = params["min_snr"]
        in_vacuo = params["in_vacuo"]
        arc_file = params["linelist"]
        xdeg = params["dispersion_order"]
        ydeg = params["xdorder_order"]
        debug_plot = params["debug_plot"]
        sfx = params["suffix"]

        # The stddev for unresolved lines (empirically determined from fits)
        sigma_unres = 1.25

        # We do some hacking here so that we can use the Linelist class to
        # handle air-to-vacuum conversions. The "weights" column is actually
        # the Gaussian width
        if arc_file is not None:
            try:
                linelists = [wavecal.LineList(f) for f in arc_file.split(",")]
            except OSError:
                log.warning(f"Cannot read file {arc_file} - "
                            "using default linelist")
                arc_file = None
            else:
                log.stdinfo(f"Read arc line list(s) {arc_file}")
        if arc_file is None:
            linelists = self._get_linelist()

        # Construct arrays of line wavelengths in the units we want, and their
        # widths in pixels, and we don't need to access the LineList again
        ref_waves = np.hstack([l.wavelengths(in_vacuo=in_vacuo, units=u.nm)
                                     for l in linelists])
        idx = ref_waves.argsort()
        ref_waves = ref_waves[idx]
        ref_sigpx = np.hstack([[sigma_unres] * len(l) if l.weights is None else l.weights
                               for l in linelists])[idx]

        def _create_fitting_groups(pixloc, sigpx):
            sigma = sigpx[0]
            groups = [[0]]
            for i, (x, s) in enumerate(zip(pixloc[1:], sigpx[1:]), start=1):
                sigma = max(sigma, s)
                if x - pixloc[groups[-1][-1]] < 3 * sigma:
                    groups[-1].append(i)
                else:
                    groups.append([i])
            return groups

        npix = 2048
        pixels = np.arange(npix)
        xcorr = np.zeros((2*npix-1,))
        for ad in adinputs:
            slitlengths_asec = dict(zip(ad.MDF['specorder'],
                                        ad.MDF['slitlength_asec']))

            # First pass to extract spectra and get global shift
            spectra = []
            for ext in ad:
                data, mask, variance, extract_info = peak_finding.average_along_slit(
                    ext, center=npix//2, nsum=nsum)
                spectra.append(ext.nddata.__class__(data=data, mask=mask, variance=variance))

                # We can't just put all the wavelengths into wave_model.inverse()
                # because that may not be monotonic when extrapolated
                wave_model = am.get_named_submodel(ext.wcs.forward_transform, "WAVE")
                in_order = np.logical_and(ref_waves >= wave_model(0),
                                          ref_waves <= wave_model(npix-1))
                if in_order.sum() == 0:  # nothing to see here
                    continue

                pixloc = wave_model.inverse(ref_waves[in_order])
                # Since we haven't matched the lines yet, we don't know their
                # widths, so just assume stddev=1.5 pixels for all
                expected = np.zeros((2048,)) + np.sum([np.exp(-(cntr - pixels) ** 2 / 4.5)
                                   for cntr in pixloc], axis=0)
                xcorr += np.correlate(expected, data, mode='full')

            # -ve means line peaks are "ahead" of expected
            global_shift = xcorr.argmax() - (npix-1)
            log.stdinfo(f"{ad.filename}: global shift of {global_shift} "
                        "pixels from cross-correlation")

            fitted_lines = []
            debug_data = []
            for ext, spec, spec_order in zip(ad, spectra, ad.hdr['SPECORDR']):
                # For debug purposes
                slit_edges = [am.table_to_model(row) for row in ext.SLITEDGE]
                edges = lambda x: (slit_edges[0](x) + ext.detector_section().y1,
                                      slit_edges[1](x) + ext.detector_section().y1)

                wave_model = (models.Shift(global_shift) |
                              am.get_named_submodel(ext.wcs.forward_transform, "WAVE"))
                in_order = np.logical_and(ref_waves >= wave_model(0),
                                          ref_waves <= wave_model(npix-1))
                if in_order.sum() == 0:
                    continue

                # Get the lines that appear in this order, and group them if
                # they're close together so that we can fit them simultaneously
                order_waves = ref_waves[in_order]
                order_sigma = ref_sigpx[in_order]
                pixloc = wave_model.inverse(order_waves)
                groups = _create_fitting_groups(pixloc, order_sigma)
                log.stdinfo(f"Order {spec_order}: Fitting {len(order_waves)} "
                            f"lines in {len(groups)} groups")

                for group in groups:
                    linestr = "+".join(f"{order_waves[line]:.2f}" for line in group)
                    x1 = int(pixloc[group[0]] - 5*order_sigma[group[0]] - 5)
                    x2 = int(pixloc[group[-1]] + 5*order_sigma[group[-1]] + 6)
                    if x1 < 0 or x2 >= npix:  # too close to edge, skip
                        continue
                    x = np.arange(x1, x2)
                    y = np.ma.masked_array(spec.data[x1:x2], mask=spec.mask[x1:x2])
                    if y.size - y.count() > 2:  # max number of masked pixels
                        log.stdinfo(f"Rejecting line(s) at {linestr}nm in "
                                    f"order {spec_order} due to excessive masking")
                        continue
                    m_init = models.Const1D()  # continuum
                    for i, line in enumerate(group, start=1):
                        g = models.Gaussian1D(amplitude=max(spec.data[int(pixloc[line])], 0),
                                              mean=pixloc[line], stddev=order_sigma[line])
                        if i > 1:  # fix the separations of the lines
                            g.mean.tied = lambda m: m.mean_1 + (pixloc[line] - pixloc[group[0]])
                            g.stddev.tied = lambda m: m.stddev_1
                        else:
                            g.mean.bounds=(pixloc[line] - 5, pixloc[line] + 5)
                        g.amplitude.bounds = (0, np.inf)
                        factor = 1.25 if order_sigma[line] == sigma_unres else 2.0
                        g.stddev.bounds=(order_sigma[line] / factor,
                                         order_sigma[line] * factor)
                        m_init += g

                    fit_it = fitting.TRFLSQFitter()
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=AstropyWarning)
                        m_final = fit_it(m_init, x, y, maxiter=1000)
                    if not fit_it.fit_info.success:
                        log.stdinfo(f"Rejecting line(s) at {linestr}nm in "
                                    f"order {spec_order} due to fit failure: "
                                    f"{fit_it.fit_info.message}")
                        continue

                    # Check that line passed min_snr requirement
                    lines_only = m_final(x) - m_final[0](x)
                    snr = lines_only.max() / np.sqrt(spec.variance[x[lines_only.argmax()]])
                    if snr < min_snr:
                        log.stdinfo(f"Rejecting line(s) at {linestr}nm in "
                                    f"order {spec_order} with SNR={snr:.1f}")
                        continue

                    # Check that multiple lines actually *are* being fit, and
                    # it's not fitting the whole thing with a single line and
                    # adding low-amplitude features
                    amplitudes = np.asarray([getattr(m_final, f"amplitude_{i}").value
                                             for i, _ in enumerate(group, start=1)])
                    if amplitudes.min() < 0.5 * amplitudes.max():
                        log.stdinfo(f"Rejecting line(s) at {linestr}nm in "
                                    f"order {spec_order} due to large "
                                    "amplitude differences")
                        continue

                    # Store order, pixel location, and wavelength. For a group
                    # with multiple lines, we store the mean pixel location and
                    # mean wavelength, since we've only fit a single parameter.
                    pixel = np.mean([getattr(m_final, f"mean_{i+1}").value
                                       for i in range(len(group))])
                    wavelength = np.mean([order_waves[line] for line in group])
                    sigma = np.mean([getattr(m_final, f"stddev_{i+1}").value
                                     for i in range(len(group))])
                    log.debug(f"Adding ({spec_order}, {pixel:.2f}, {linestr})"
                              " to fitted line list")
                    fitted_lines.append((spec_order, pixel, wavelength, sigma))

                    # debug information
                    for i, line in enumerate(group, start=1):
                        _tuple = (spec_order,
                                  order_waves[line],
                                  getattr(m_final, f"mean_{i}").value,
                                  getattr(m_final, f"amplitude_{i}").value,
                                  getattr(m_final, f"stddev_{i}").value,
                                  edges(getattr(m_final, f"mean_{i}").value),
                                  len(fitted_lines)-1)
                        debug_data.append(_tuple)

                    if spec_order == 0 and 0 < m_final.mean_1 < 2000:
                        for p in m_final.param_names:
                            print(f"{p}: {getattr(m_final, p).value}")
                        print("Ref wavelengths", [order_waves[line] for line in group])
                        print("Evaluated", wave_model([getattr(m_final, f"mean_{i}").value for i, _ in enumerate(group, start=1)]))
                        fig, ax = plt.subplots()
                        ax.plot(x, y, 'k-')
                        ax.plot(x, m_final(x), 'b-')
                        ax.plot(x, m_final[0](x), 'r-')
                        for i in range(1, m_final.n_submodels):
                            ax.plot(x, m_final[0](x) + m_final[i](x), 'r-')
                        plt.show()

            # Now perform the fit
            orders, pix, waves, sigmas = [np.asarray(x) for x in zip(*fitted_lines)]
            fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                                       outlier_func=sigma_clip)
            m_init = models.Chebyshev2D(x_degree=xdeg, y_degree=ydeg,
                                        x_domain=(0, 2047),
                                        y_domain=(min(orders), max(orders)))
            m_final, mask = fit_it(m_init, pix, orders, waves*orders)

            # This code is comparable to update_wcs_with_solution()
            tbl_orders = orders[~mask].astype(int)
            tbl_pix = pix[~mask].astype(np.float32)
            tbl_waves = waves[~mask].astype(np.float32)
            tbl_sigmas = sigmas[~mask].astype(np.float32)
            tbl_fitted = m_final(tbl_pix, tbl_orders) / tbl_orders
            rms = np.std(m_final(tbl_pix, tbl_orders) / tbl_orders - tbl_waves)
            tbl_pix += 1  # use 1-based for output

            temptable = am.model_to_table(m_final, xunit=u.pixel, yunit=None)
            temptable.add_columns([[2], [xdeg], [ydeg], [0], [2047], [min(orders)], [max(orders)], [rms]],
                                  names=("ndim", "xdegree", "ydegree", "xdomain_start",
                                         "xdomain_end", "ydomain_start", "ydomain_end", "rms"))

            pad_rows = tbl_orders.size - len(temptable.colnames)
            if pad_rows < 0:
                tbl_orders = list(tbl_orders) + [0] * (-pad_rows)
                tbl_pix = list(tbl_pix) + [0] * (-pad_rows)
                tbl_waves = list(tbl_waves) + [0] * (-pad_rows)
                pad_rows = 0

            fit_table = Table([temptable.colnames + [''] * pad_rows,
                               list(temptable[0].values()) + [0] * pad_rows,
                               tbl_orders, tbl_pix, tbl_waves,
                               tbl_fitted, tbl_sigmas],
                              names=("name", "coefficients", "xdorder", "peaks",
                                     "wavelengths", "fitted", "sigma"),
                              units=(None, None, None, u.pix, u.nm, u.nm, u.pix),
                              meta=temptable.meta)
            medium = "vacuo" if in_vacuo else "air"
            fit_table.meta['comments'] = [
                'coefficients are based on 0-indexing',
                'peaks column is 1-indexed',
                f'calibrated with wavelengths in {medium}']
            ad.WAVECAL = fit_table
            log.stdinfo(f"Rejected {mask.sum()}/{pix.size} lines/groups "
                        f"for final rms = {rms:.4f} nm")

            # Create the wavelength models
            for ext, spec_order in zip(ad, ad.hdr['SPECORDR']):
                spectral_frame = ext.wcs.output_frame.frames[0]
                axis_name = "WAVE" if in_vacuo else "AWAV"
                new_spectral_frame = cf.SpectralFrame(
                    axes_order=spectral_frame.axes_order,
                    unit=spectral_frame.unit, axes_names=(axis_name,),
                    name=adwcs.frame_mapping[axis_name].description)
                spatial_frame = cf.CoordinateFrame(naxes=1, axes_type="SPATIAL",
                                                   axes_order=(1,), unit=[u.arcsec],
                                                   name="SPATIAL")
                output_frame = cf.CompositeFrame([new_spectral_frame, spatial_frame], name="world")

                cheb1d = am.reduce_dimensionality(m_final, y=spec_order)
                for p, v in zip(cheb1d.param_names, cheb1d.parameters):
                    if p.startswith("c"):
                        setattr(cheb1d, p, v / spec_order)
                cheb1d.inverse = am.make_inverse_chebyshev1d(cheb1d, max_deviation=0.01)
                cheb1d.name = "WAVE"
                slit_model = models.Scale(slitlengths_asec[spec_order], name="SKY")
                transform = cheb1d & slit_model
                ext.wcs = gWCS(ext.wcs.pipeline[:-2] +
                               [(ext.wcs.pipeline[-2].frame, transform),
                                (output_frame, None)])

            if debug_plot:
                orders = ad.hdr['SPECORDR']
                fig, ax = plt.subplots()
                cmap = plt.get_cmap('jet')
                norm = mcolors.Normalize(vmin=-0.005, vmax=0.005)

                for ext in ad:
                    edge1, edge2 = [am.table_to_model(row) for row in ext.SLITEDGE]
                    xfill = list(pixels) + list(pixels[::-1])
                    yfill = (list(edge1(pixels) + ext.detector_section().y1) +
                             list(edge2(pixels[::-1]) + ext.detector_section().y1))
                    ax.fill(xfill, yfill, color="gainsboro", edgecolor=None)

                for order, wave, x, amp, sig, yends, id in debug_data:
                    linestyle = ':' if mask[id] else '-'
                    residual = ad[orders.index(order)].wcs(x, 0)[0] - wave
                    color = cmap(norm(residual))
                    off = 0
                    ax.plot([x-off, x+off], yends,
                            color=color, ls=linestyle)

                ax.set_xlim(0, 2047)
                ax.set_ylim(0, 2047)
                ax.set_aspect('equal')
                ax.set_title(f"{ad.filename}\n{pix.size} lines "
                             f"({mask.sum()} rejected) RMS = {rms:.4f}nm")
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                fig.colorbar(sm, ax=ax, label="Fitted residual (nm)")
                plt.tight_layout(pad=1)
                plt.savefig(ad.filename.replace(".fits", "_wavecal.pdf"))

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def distortionCorrect(self, adinputs=None, **params):
        """
        Corrects optical distortion in science frames, using a distortion map
        that has previously been attached to each input's WCS by
        attachWavelengthSolution.

        This IGRINS-2 specific version copes with the fact that the origin of
        the distortion-corrected frame is in the middle of the slit, so having
        the origin of the output frame at (0, 0) will result in the bottom
        half of the slit being cropped. We therefore specify an origin when
        resampling.

        Parameters
        ----------
       suffix : str
            Suffix to be added to output files.
        interpolant : str
            Type of interpolant
        subsample : int
            Pixel subsampling factor.
        dq_threshold : float
            The fraction of a pixel's contribution from a DQ-flagged pixel to
            be considered 'bad' and also flagged.
       """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        sfx = params["suffix"]
        interpolant = params["interpolant"]
        subsample = params["subsample"]
        do_cal = params["do_cal"]
        dq_threshold = params["dq_threshold"]

        if do_cal == 'skip':
            log.warning('Distortion correction has been turned off.')
            return adinputs

        origin = (-int(slitpos_to_pix(0.) + 0.5), 0)
        output_shape = (abs(origin[0]) * 2 + 1, 2048)

        fail = False
        adoutputs = []
        for ad in adinputs:
            if not all(['distortion_corrected' in ext.wcs.available_frames
                        for ext in ad]):
                log.warning(f"{ad.filename}: cannot perform distortion "
                            "correction as one or more extensions do not "
                            "have a distortion map")
                fail = True
                adoutputs.append(ad)
                continue

            # for ext in ad:
            #     frame_idx = ext.wcs.available_frames.index('distortion_corrected')
            #     model = models.Shift(0) & models.Shift(-25)
            #     model.inverse = ext.wcs.get_transform(
            #         ext.wcs.available_frames[frame_idx],
            #         ext.wcs.input_frame)
            #     # To avoid continually checking whether we're correcting to
            #     # "distortion_corrected" or "rectified", rename the frame to
            #     # which we're correcting as "correction_endpoint"
            #     endpoint_frame = ext.wcs.pipeline[frame_idx].frame
            #     endpoint_frame.name = "correction_endpoint"
            #     ext.wcs = gWCS([(ext.wcs.input_frame, model),
            #                     (endpoint_frame, ext.wcs.pipeline[frame_idx].transform)]
            #                    + ext.wcs.pipeline[frame_idx+1:])

            # Cut the output to the size we want
            for i, ext in enumerate(ad):
                temp_out = transform.resample_from_wcs(
                    ext, 'distortion_corrected', interpolant=interpolant,
                    subsample=subsample, parallel=False, threshold=dq_threshold,
                    output_shape=output_shape, origin=origin)
                if i == 0:
                    ad_out = astrodata.create(temp_out.phu)
                ad_out.append(temp_out[0].nddata)

            # Timestamp and update the filename
            gt.mark_history(ad_out, primname=self.myself(),
                            keyword=timestamp_key)
            ad_out.update_filename(suffix=sfx, strip=True)
            adoutputs.append(ad_out)

        if fail:
            raise ValueError("One or more input(s) missing distortion "
                             "calibration; run attachWavelengthSolution first")

        return adoutputs

    def extractSpectra(self, adinputs=None, **params):
        """
        Create a one-dimensional spectrum by extracting the signal along each
        order.

        Parameters
        ----------
       suffix : str
            Suffix to be added to output files.
        method : str [aperture|optimal|default]
            Extraction method to use
        sigma : float
            Standard deviation threshold for cosmic ray rejection
        debug_order : int/None
            Echelle order for producing a debugging plot
        debug_pixel : int/None
            Pixel in "debug_order" for producing a debugging plot
        debug_min_frac : float
            Minimum fraction of good pixels needed in each columns in order
            to extract the flux
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        method = params["method"]
        sigmasq = params["sigma"] ** 2
        debug_order = params["debug_order"]
        debug_pixel = params["debug_pixel"]
        min_frac = params["debug_min_frac"]

        adoutputs = []
        for ad in adinputs:
            adout = astrodata.create(ad.phu)
            this_method = method if method != "default" else (
                "optimal" if 'STANDARD' in ad.tags else "aperture")
            for ext in ad:
                data = np.zeros((ext.shape[1],), dtype=ext.data.dtype)
                mask = np.full((ext.shape[1],), DQ.no_data, dtype=ext.mask.dtype)
                var = np.zeros_like(data)
                y, x = np.mgrid[:ext.shape[0], :ext.shape[1]]

                m_init = models.Polynomial1D(degree=1,
                                             bounds={"c0": (0, np.inf),
                                                     "c1": (0, np.inf)})
                m_init.linear = False  # quickest way to suppress warnings
                fit_it = fitting.FittingWithOutlierRemoval(fitting.DogBoxLSQFitter(),
                                                           sigma_clip)
                good = np.logical_and(ext.mask == 0, ext.variance > 0)
                m_var, _ = fit_it(m_init, abs(ext.data[good]), ext.variance[good])
                log.debug(f"{ext.hdr['SPECORDR']} estimated var: "
                          f"{m_var.c0.value:.2f} {m_var.c1.value:.2f}")

                # Construct an image of the slit position, which is used
                # to determine the extraction weight of each pixel
                # y coordinate is the same in 'slitpos' and 'distcorr_slitpos'
                t = ext.wcs.get_transform("pixels", "distcorr_slitpos")
                y_slitpos = t(x, y)[1]
                ext.SLITPOS = y_slitpos

                # Compute the model needed to horizontally shift each row
                # so as to make the sky lines vertical. We don't care about
                # the subsequent WCS since we're not going to use that, only
                # the resampled pixel values.
                xx = t.inverse(x, y_slitpos)[0]
                t = models.Identity(2)
                t.inverse = (models.Mapping((0, 1, 1)) |
                             models.Tabular2D(lookup_table=xx.T, bounds_error=False,
                                              fill_value=0) & models.Identity(1))
                xshifted = astrodata.wcs.pixel_frame(naxes=2, name="xshifted")
                ext.wcs.insert_frame(ext.wcs.input_frame, t, xshifted)
                ext_xshifted = transform.resample_from_wcs(
                    ext, "xshifted",
                    attributes=["data", "mask", "variance", "SLITPOS"])[0]

                # Development confidence check
                pixels_for_extraction = ext_xshifted.SLITPOS[ext_xshifted.mask == 0]
                assert pixels_for_extraction.min() >= -0.5
                assert pixels_for_extraction.max() <= 0.5
                del pixels_for_extraction

                goodpix = np.logical_and.reduce([ext_xshifted.variance > 0,
                                                 ext_xshifted.mask == 0,
                                                 abs(ext_xshifted.SLITPOS) < 0.5])
                slitpos_samples = slitpos_to_pix.inverse(np.arange(ext.SLITPROF.shape[0]))

                # For aperture extraction, we want to know where to extract.
                # The data are effectively noiseless, so we don't use
                # peak_finding.get_extrema() here.
                if this_method == "aperture":
                    slitprof = np.median(ext.SLITPROF, axis=1)
                    spl = make_interp_spline(slitpos_samples, slitprof, k=3)
                    ppoly = PPoly.from_spline(spl.tck)
                    roots = np.r_[-0.5, ppoly.roots(extrapolate=False), 0.5]
                    extrema = ppoly.derivative().roots(extrapolate=False)
                    extrema_values = ppoly(extrema)
                    i = bisect_left(roots, extrema[extrema_values.argmax()])
                    extraction_regions = [(roots[i-1], roots[i], 1.0)]
                    log.debug("Extracting positive beam "
                              f"(order {ext.hdr['SPECORDR']} between "
                              f"{extraction_regions[0][0]:.3f} and "
                              f"{extraction_regions[0][1]:.3f}")
                    # Are there A and B beams?
                    ab = extrema_values.min() < -0.5 * extrema_values.max()
                    if ab:
                        i = bisect_left(roots, extrema[extrema_values.argmin()])
                        extraction_regions.append((roots[i-1], roots[i], -1.0))
                        log.debug("Extracting negative beam between "
                                  f"{extraction_regions[1][0]:.3f} and "
                                  f"{extraction_regions[1][1]:.3f}")
                    pixels = np.arange(ext.shape[0])
                elif method == "optimal":
                    # We want to normalize the profile along each column. Since
                    # there's probably both a +ve and -ve beam, we can't simply
                    # use the sum of the profile.
                    spl = make_interp_spline(slitpos_samples, ext.SLITPROF, k=3, axis=0)
                    t, c, k = spl.tck

                for i, (slitpos, coldata, colvar, colgood) in enumerate(zip(ext_xshifted.SLITPOS.T,
                                                                            ext_xshifted.data.T,
                                                                            ext_xshifted.variance.T,
                                                                            goodpix.T)):
                    debug = ext.hdr['SPECORDR'] == debug_order and i == debug_pixel
                    if debug:
                        fig, ax = plt.subplots()
                        ax.plot(slitpos, coldata, 'k-')
                        ax.plot(slitpos, np.sqrt(colvar), 'r-')
                        ax.plot(slitpos[~colgood], coldata[~colgood], 'ko')
                    if colgood.sum() and ext.hdr['SPECORDR'] > 70:
                        if this_method == "optimal":
                            ppoly = PPoly.from_spline((t, c[:, i], k))
                            prof = ppoly(slitpos)
                            if prof.max() > 0:
                                # Renormalize the profile sum to unity.
                                # Summing the pixels directly accounts for the
                                # different pixel scale between the resampled
                                # SLITPROF and the actual data.
                                pixels_in_slit = abs(slitpos) <= 0.5
                                prof /= abs(prof[pixels_in_slit]).sum()
                                flux = np.sum(abs(coldata[colgood]))
                                while True:
                                    if colgood.sum() >= min_frac * pixels_in_slit.sum():
                                        colvar = m_var(abs(flux * prof))
                                        with warnings.catch_warnings(category=RuntimeWarning,
                                                                     action="ignore"):
                                            ivar = np.where(colgood, 1. / colvar, 0)
                                        flux = (prof * coldata * ivar).sum() / (prof * prof * ivar).sum()
                                        new_masked = np.logical_and(
                                            (coldata - flux * prof) ** 2 > sigmasq * colvar,
                                            colgood)
                                        if new_masked.sum() == 0:
                                            data[i] = flux
                                            mask[i] = DQ.good
                                            var[i] = abs(prof[colgood]).sum() / (prof * prof * ivar).sum()
                                            if debug:
                                                ax.plot(slitpos[colgood], prof[colgood] * flux, 'b-')
                                                ax.plot(slitpos[new_masked], coldata[new_masked], 'bo')
                                                print("FLUX and RAW", flux, np.sum(abs(coldata[colgood])))
                                            break
                                        colgood[new_masked] = False
                                    else:  # rejected all pixels, can't assign data
                                        break
                        elif this_method == "aperture":
                            mask[i] = DQ.good
                            for s1, s2, sign in extraction_regions:
                                x1, x2 = np.interp([s1, s2], slitpos, pixels,
                                                   left=np.nan, right=np.nan)
                                if np.isnan(x1) or np.isnan(x2):
                                    mask[i] = DQ.no_data
                                    continue

                                _slice = slice(int(np.ceil(x1-0.5))+1, int(np.ceil(x2-0.5)))
                                data[i] += sign * coldata[_slice].sum()
                                mask[i] |= np.logical_or.reduce(ext_xshifted.mask[_slice, i])
                                var[i] += colvar[_slice].sum()
                        else:
                            data[i] = np.abs(coldata[colgood]).sum()
                            mask[i] = np.logical_or.reduce(ext_xshifted.mask[colgood, i])
                            var[i] = colvar[colgood].sum()

                    if debug:
                        ax.set_xlim(-0.5, 0.5)
                        plt.show()

                if np.all(mask & DQ.no_data):
                    log.warning(f"No good pixels found for extraction in order {ext.hdr['SPECORDR']}")
                    continue

                wave_model = am.get_named_submodel(ext.wcs.forward_transform, "WAVE")
                for f in ext.wcs.output_frame.frames:
                    if isinstance(f, cf.SpectralFrame):
                        output_frame = f
                        break
                else:
                    raise ValueError("Cannot find spectral frame in WCS")

                wcs1d = gWCS([(adwcs.pixel_frame(naxes=1), wave_model),
                              (output_frame, None)])
                adout.append(ext.nddata.__class__(data=data, mask=mask, variance=var, wcs=wcs1d,
                                                  meta={'header': ext.hdr.copy()}))
                if any(np.isnan(data)):
                    log.warning(f"NaNs in {ad.filename} order {ext.hdr['SPECORDR']}")

            if len(adout) == 0:
                raise ValueError(f"Unable to extract any orders in {ad.filename}")

            adout.hdr['APERTURE'] = 1
            # Timestamp and update the filename
            gt.mark_history(adout, primname=self.myself(),
                            keyword=timestamp_key)
            adout.update_filename(suffix=sfx, strip=True)
            adoutputs.append(adout)

        return adoutputs

    def flexureCorrect(self, adinputs=None, **params):
        """
        Corrects for flexure in the science frames.

        This works by cross-correlated each echelle order (collapsed along
        the slit) in the science frame with the corresponding order in an
        arc/sky frame. Before performing the cross-correlation, the regions
        of the slit containing objects are masked out.

        Parameters
        ----------
        suffix : str
            Suffix to be added to output files.
        arc : str or AstroData or None
            arc/sky frame to use for flexure correction. If None, the
            calibration service will be used
        do_cal : str [procmode|force|skip]
            perform the flexure correction?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        sfx = params["suffix"]
        arc = params["arc"]
        do_cal = params["do_cal"]
        subsample = 50

        if do_cal == 'skip':
            log.warning('Distortion correction has been turned off.')
            return adinputs

        if arc is None:
            arc_list = self.caldb.get_processed_arc(adinputs)
            if len(set(arc_list.files)) == 1:
                log.stdinfo(f"Using the arc {arc_list.files[0]} (obtained from"
                            f" {arc_list.origins[0]}) for flexure correction")
                arc = arc_list.files[0]
            else:
                raise ValueError("Cannot find a single arc/sky frame for flexure correction")
        else:
            log.stdinfo(f"Using the arc {arc} for flexure correction")

        hist_bins = np.arange(-0.5, 0.51, 0.02)
        xpixels = np.arange(2048)
        if isinstance(arc, str):
            arc = astrodata.open(arc)
        detsecs = arc.detector_section()

        ad_shifts = {}

        # Make a dict of arc line locations for each order?
        arc_lines = {}
        for spec_order, peak in zip(arc.WAVECAL['xdorder'].data,
                                    arc.WAVECAL['peaks'].data):
            try:
                arc_lines[spec_order].append(peak-1)  # 1-indexed in WAVECAL
            except KeyError:
                arc_lines[spec_order] = [peak-1]

        from datetime import datetime
        for ad in adinputs:
            profile = np.zeros(hist_bins.size - 1)
            # We first want to know which parts of the echellogram are free
            # from objects, so construct a crude slit profile histogram
            for ext_arc, ext_section in zip(arc, detsecs):
                y, x = np.mgrid[:ext_arc.shape[0], :ext_arc.shape[1]]
                t = ext_arc.wcs.get_transform("pixels", "distcorr_slitpos")
                y_slitpos = t(x, y)[1]
                # Find the mean pixel value in each bin. Ignore orders that
                # fall off the top or bottom of the detector
                histsum = np.histogram(y_slitpos, bins=hist_bins,
                                       weights=ad[0].data[ext_section.asslice()])[0]
                histnum = np.histogram(y_slitpos, bins=hist_bins)[0]
                if all(histnum > 0):
                    profile += histsum / histnum

            peak, loc = profile.max(), profile.argmax()
            top = bottom = loc
            frac = 0.05
            bkgd = np.percentile(profile, 10)
            while top < profile.size - 1 and profile[top] > frac * (profile[loc] - bkgd) + bkgd:
                top +=1
            while bottom > 0 and profile[bottom] > frac * (profile[loc] - bkgd) + bkgd:
                bottom -= 1
            if bottom > profile.size - top:
                limits = (0, bottom-2)
            else:
                limits = (top+2, profile.size-1)
            slitpos_limits = slitpos_to_pix.inverse(limits)
            if slitpos_limits[1] - slitpos_limits[0] < 0.1:
                log.warning(f"{ad.filename}: Unable to find a region of the "
                            "slit free from objects")
                continue
            log.stdinfo(f"Extracting sky between slit positions {slitpos_limits}")

            # Now do the cross-correlation to find the flexure shift
            shifts = []
            for ext_arc, ext_section in zip(arc, detsecs):
                ndd = ad[0].nddata[ext_section.asslice()]
                y, x = np.mgrid[:ext_arc.shape[0], :ext_arc.shape[1]]
                t = ext_arc.wcs.get_transform("pixels", "distcorr_slitpos")
                y_slitpos = t(x, y)[1]
                mask = np.logical_or.reduce([ndd.mask, ext_arc.mask,
                                             y_slitpos < slitpos_limits[0],
                                             y_slitpos > slitpos_limits[1]])
                not_blank = mask.sum(axis=1) < 1024
                rows = np.flatnonzero(not_blank)
                if not rows.size:
                    continue
                collapsed_image = np.ma.median(np.ma.masked_where(mask, ndd.data), axis=0)
                collapsed_arc = np.ma.median(np.ma.masked_where(mask, ext_arc.data), axis=0)
                collapsed_mask = collapsed_image.mask  # same for both
                unmasked_indices = np.flatnonzero(~collapsed_mask)
                spl_image = make_interp_spline(xpixels[~collapsed_mask],
                                               collapsed_image[~collapsed_mask], k=3)
                spl_arc = make_interp_spline(xpixels[~collapsed_mask],
                                             collapsed_arc[~collapsed_mask], k=3)
                narclines = list(arc.WAVECAL['xdorder'].data).count(ext_arc.hdr['SPECORDR'])
                xeval = np.arange(unmasked_indices[0], unmasked_indices[-1] + 0.01, 1. / subsample)

                corrfunc = lambda dx: -np.sum(spl_arc(xeval) * spl_image(xeval + dx))
                from scipy.optimize import minimize
                result = minimize(corrfunc, 0, bounds=[(-4, 4)])
                log.stdinfo(f"{ad.filename} order {ext_arc.hdr['SPECORDR']}: shift={result.x[0]:.3f}")
                if abs(result.x[0]) < 3:
                    shifts.append(result.x[0])
                if ext_arc.hdr['SPECORDR'] == 0:
                    fig, ax = plt.subplots()
                    x = np.arange(-4, 4, 0.05)
                    #y = [corrfunc(dx) for dx in x]
                    ax.plot(xpixels, spl_image(xpixels), 'k-')
                    ax.plot(xpixels, spl_arc(xpixels), 'r-')
                    #ax.plot(x, y, 'k-')
                    plt.show()

            if shifts:
                # Use the median
                shift = sigma_clipped_stats(shifts, sigma=2, maxiters=5)[1]
                ad_shifts[ad.filename] = shift

        avg_shift = np.median(list(ad_shifts.values()))

        for ad in adinputs:
            ad.phu['FLEXSHFT'] = (avg_shift, "Measured flexure shift (pixels)")
            log.stdinfo(f"Recording shift of {avg_shift:.3f} pixels for {ad.filename}")

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def makeAB(self, adinputs=None, **params):
        """
        This performed the same work as the makeAB primitive, but doesn't do
        the flexure correction, which now lives in its own primitive. If all
        works as desired, this will become the makeAB primitive and the old
        one will be removed.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        #timestamp_key = self.timestamp_keys[self.myself()]
        suffix = params["suffix"]
        frac_FOV = 1.0

        frametypes = [ad.phu.get("FRMTYPE") for ad in adinputs]
        if frametypes.count(None) == 0:
            log.stdinfo("Grouping by FRMTYPE keyword")
            frametype_set = set(frametypes)
            assert len(frametype_set) == 2
            in_group_a = np.array([ft == frametypes[0] for ft in frametypes])
            if frametype_set.intersection({"A", "ON"}) and frametypes[0] not in ("A", "ON"):
                in_group_a = ~in_group_a
        else:
            log.stdinfo("Grouping by location on sky")
            groups = gt.group_exposures(adinputs, fields_overlap=self._fields_overlap,
                                        frac_FOV=frac_FOV)
            assert len(groups) == 2
            in_group_a = [ad in groups[0] for ad in adinputs]

        adinputsA = [ad for ad, in_a in zip(adinputs, in_group_a) if in_a]
        adinputsB = [ad for ad, in_a in zip(adinputs, in_group_a) if not in_a]
        if len(adinputsA) * len(adinputsB) == 0:
            raise ValueError("Cannot find two groups of exposures for subtraction")

        grp_a_list = "\n    ".join([ad.filename for ad in adinputsA])
        grp_b_list = "\n    ".join([ad.filename for ad in adinputsB])
        log.stdinfo(f"Exposures in group A:\n    {grp_a_list}")
        log.stdinfo(f"Exposures in group B:\n    {grp_b_list}")
        stackedA = self.stackFrames(adinputsA).pop()
        stackedB = self.stackFrames(adinputsB).pop()

        # FIXME should we better to create a new instance of AstroData?
        ad = stackedA.subtract(stackedB)
        # Timestamp and update filename
        #gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
        ad.update_filename(suffix=suffix, strip=True)

        return [ad]

    def removeObjectsLeaveSky(self, adinputs=None, **params):
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        #timestamp_key = self.timestamp_keys[self.myself()]
        suffix = params["suffix"]
        frac_FOV = 1.0

        frametypes = [ad.phu.get("FRMTYPE") for ad in adinputs]
        if frametypes.count(None) == 0:
            log.stdinfo("Grouping by FRMTYPE keyword")
            frametype_set = set(frametypes)
            assert len(frametype_set) == 2
            in_group_a = np.array([ft == frametypes[0] for ft in frametypes])
            if frametype_set.intersection({"A", "ON"}) and frametypes[0] not in ("A", "ON"):
                in_group_a = ~in_group_a
        else:
            log.stdinfo("Grouping by location on sky")
            groups = gt.group_exposures(adinputs, fields_overlap=self._fields_overlap,
                                        frac_FOV=frac_FOV)
            assert len(groups) == 2
            in_group_a = [ad in groups[0] for ad in adinputs]

        adoutputs = []
        adinputsA, adinputsB = [], []
        print(in_group_a)
        for ad, in_a in zip(adinputs, in_group_a):
            if in_a:
                adinputsA.append(ad)
            else:
                adinputsB.append(ad)
            if len(adinputsA) == len(adinputsB):
                grp_a_list = "\n    ".join([ad.filename for ad in adinputsB])
                grp_b_list = "\n    ".join([ad.filename for ad in adinputsB])
                log.stdinfo(f"Exposures in group A:\n    {grp_a_list}")
                log.stdinfo(f"Exposures in group B:\n    {grp_b_list}")
                stackedA = self.stackFrames(adinputsA).pop()
                stackedB = self.stackFrames(adinputsB).pop()
                stackedA_copy = copy.deepcopy(stackedA)
                ad = stackedA.subtract(stackedB)
                for ext in ad:
                    ext.data = -abs(ext.data)
                ad.add(stackedA_copy)
                ad.add(stackedB)
                ad.update_filename(suffix=suffix, strip=True)
                adoutputs.append(ad)

        return adoutputs

    def maskVignettedRegions(self, adinputs=None, **params):
        """
        This primitive attempts to find and mask the regions of each echelle
        order that are vignetted and so create a rapid drop-off in the
        response function. This dtop-off is hard to fit in the fitTelluric()
        primitive and better results are obtained by masking these regions.

        The primitive using a Savitzky-Golay filter to identify the decrease
        in the first derivative caused by each "knee" or "shoulder", and then
        a linear fit to these locations as a function of echelle order is
        performed to avoid issues from misidentifications. The points beyond
        these locations are then masked as DQ.overlap (which isn't strictly
        correct, but it's not entirely wrong either and it makes it easier to
        unmask the pixels later if desired).

        Parameters
        ----------
        suffix : str
            Suffix to be added to output files
        debug_halfwidth : int
            halfwidth of Savitzky-Golay filter
        debug_order : int
            order of Savitzky-Golay polynomial
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        #timestamp_key = self.timestamp_keys[self.myself()]
        timestamp_key = "MASKVIGN"
        suffix = params["suffix"]
        halfwidth = params["debug_halfwidth"]
        polyorder = params["debug_order"]

        for ad in adinputs:
            if ad.wavelength_band() == "H":
                log.stdinfo(f"{ad.filename}: H-band data are not vignetted, skipping")
                continue

            limits = []
            for ext in ad:
                masked_data = np.ma.masked_array(ext.data, mask=ext.mask, copy=False)
                spek = np.ma.median(masked_data, axis=0)
                # Get second derivative; we want to find minima in this
                ddspek = savgol_filter(spek, window_length=2 * halfwidth + 1,
                                       polyorder=polyorder, deriv=2)
                ddspek = np.ma.masked_where(spek.mask, ddspek)
                smoothed_ddspek = gaussian_filter(ddspek, halfwidth)
                # Mask regions where the S-G filter went off the edge
                masked_slices = np.ma.clump_masked(spek)
                start = 0 if not masked_slices or masked_slices[0].start > 0 \
                    else masked_slices[0].stop
                ddspek.mask[start:start + halfwidth] = True
                end = spek.size if (not masked_slices or
                                    masked_slices[-1].stop < spek.size) else masked_slices[-1].start
                ddspek.mask[end - halfwidth:end] = True
                smoothed_ddspek = np.ma.masked_where(ddspek.mask, smoothed_ddspek)

                # Take the locations of the knees/shoulders as the first
                # minimum in the second derivative, provided it's not the
                # first unmaksed pixel
                masked_slices = np.ma.clump_masked(ddspek)
                limit1 = np.ma.argmin(smoothed_ddspek[:1024])
                if limit1 == masked_slices[0].stop:
                    limit1 = np.nan
                limit2 = 1024 + np.ma.argmin(smoothed_ddspek[1024:])
                if limit2 == masked_slices[-1].start - 1:
                    limit2 = np.nan
                limits.append((limit1, limit2))
                log.debug(f"{ad.filename} order {ext.hdr['SPECORDR']} limits = {limit1}, {limit2}")

            # Now fit a linear function to each set of limits
            limits = np.asarray(limits).T
            orders = np.asarray(ad.hdr['SPECORDR'])
            fit_it = fitting.FittingWithOutlierRemoval(
                fitting.LinearLSQFitter(), sigma_clip, niter=2, sigma=2)
            m_init = models.Polynomial1D(degree=1)
            good = ~np.isnan(limits[0])
            good[:2] = False
            good[-2:] = False
            m_final, _ = fit_it(m_init, orders[good], limits[0, good])
            left_limits = np.maximum(m_final(orders).astype(int), 0)
            good = ~np.isnan(limits[1])
            good[:2] = False
            good[-2:] = False
            m_final, _ = fit_it(m_init, orders[good], limits[1, good])
            right_limits = np.minimum(m_final(orders).astype(int), 2048)
            for ext, left, right in zip(ad, left_limits, right_limits):
                log.debug(f"Masking <{left} and >={right} in order {ext.hdr['SPECORDR']}")
                ext.mask[:, :left] |= DQ.overlap
                ext.mask[:, right:] |= DQ.overlap

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)

        return adinputs

    def measureSlitProfile(self, adinputs=None, **params):
        """
        Follows the method of Cushing, Vacca, & Rayner (2004, PASP, 116, 362)
        to measure the slit profile. The primitive requires an image that has
        been distortion-corrected so that the spatial axis is vertical and the
        wavelength axis is horizontal, and the spatial profile is constructed
        at each wavelength pixel. For each echelle order, a median is taken in
        the dispersion direction and this median profile scaled to fit each
        row using a least-squares fit. The scaled rows are then median-combined
        to create equally-normalized spatial profiles. Finally, to allow for
        modest variations in the profile with wavelength, a polynomial is fit
        to each row. This synthetic profile is then attached as an array called
        'SLITPROF' to each extension.

        Parameters
        ----------
        suffix : str
            Suffix to be added to output files.
        order : int
            Order of the polynomial to fit along each resampled row
        lsigma : float
            Lower sigma threshold for rejection of outliers in the fit
        hsigma : float
            Upper sigma threshold for rejection of outliers in the fit
        niter : int
            Number of iterations for rejection of outliers in the fit
        threshold : float
            Threshold for rejecting columns with low signal when fitting
            along each row
        debug_goodfrac : float
            Minimum fraction of good pixels in a row for it to be included
            in the profile (this rejects rows on the edge of the profile
        use_variance : bool
            Weight each pixel by th einverse variance when fitting the
            profile to each column?
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        suffix = params["suffix"]
        order = params["order"]
        lsigma = params["lsigma"]
        hsigma = params["hsigma"]
        niter = params["niter"]
        threshold = params["threshold"]
        goodfrac = params["debug_goodfrac"]
        use_var = params["use_variance"]

        for ad in adinputs:
            if self.timestamp_keys['distortionCorrect'] not in ad.phu:
                raise ValueError(f"{ad.filename} has not been distortion-"
                                 "corrected and so the slit profile cannot"
                                 " be measured.")

            for ext in ad:
                npix = ext.shape[1]
                masked_data = np.ma.masked_array(ext.data, ext.mask, copy=True)
                weights = 1
                if use_var:
                    if ext.variance is None:
                        log.warning(f"{ad.filename} order {ext.hdr['SPECORDR']}"
                                    " has no variance array, so cannot use it "
                                    "to weight the slit profile")
                    else:
                        weights = at.divide0(1., ext.variance)

                for _ in range(1):  # allow iteration (4.2.2 of Cushing+ 2004)
                    profile = np.ma.median(masked_data, axis=1)
                    masked_profile = np.ma.masked_array(
                        profile[:, np.newaxis].repeat(npix, axis=1), ext.mask)
                    # Accounts for the mask since calls np.ma.sum()
                    scale_factors = ((weights * masked_data * masked_profile).sum(axis=0) /
                                     (weights * masked_profile ** 2).sum(axis=0))
                    masked_data /= scale_factors

                masked_data.mask |= (scale_factors < threshold * scale_factors.max())

                fit1d = fit_1D(masked_data, function="chebyshev",
                               domain=(0, npix-1), order=order,
                               sigma_lower=lsigma, sigma_upper=hsigma,
                               niter=niter)
                synth_data = fit1d.evaluate()
                synth_data[:, np.sum(ext.mask & DQ.no_data, axis=0) > 0] = 0
                synth_data[ext.mask.astype(bool).sum(axis=1) > goodfrac * npix] = 0

                ext.SLITPROF = synth_data.astype(np.float32)

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)

        return adinputs

    def normalizeFlat(self, adinputs=None, **params):
        """
        Performs the standard normalizeFlat primitive, and then unmasks the
        regions beyond the knees/shoulders
        """
        adinputs = super().normalizeFlat(adinputs, **params)
        if params["debug_unmask_vignetted"]:
            self.log.debug("Unmasking vignetted regions in all files")
            for ad in adinputs:
                for ext in ad:
                    ext.mask &= (DQ.max ^ DQ.overlap)
        return adinputs

    def standardizeWCS(self, adinputs=None, suffix=None):
        """
        We need to add a gWCS object to each extension since there are no
        standard FITS WCS keywords in the headers.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        for ad in adinputs:
            pixscale = ad.pixel_scale().pop()
            wcs_dict = {"CRVAL1": ad.ra(), "CRVAL2": ad.dec(),
                        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
                        "CUNIT1": "deg", "CUNIT2": "deg",
                        "RADESYS": "FK5", "EQUINOX": ad.phu['EQUINOX']}
            wcs_dict['CRPIX1'] = 1024 - ad.phu['POFFSET'] / pixscale
            wcs_dict['CRPIX2'] = 1024 - ad.phu['QOFFSET'] / pixscale
            pa = ad.phu['PA'] * np.pi / 180
            wcs_dict['CD1_1'] = pixscale * np.cos(pa) / 3600
            wcs_dict['CD1_2'] = -pixscale * np.sin(pa) / 3600
            wcs_dict['CD2_1'] = pixscale * np.sin(pa) / 3600
            wcs_dict['CD2_2'] = pixscale * np.cos(pa) / 3600
            ad[0].wcs = astrodata.wcs.fitswcs_to_gwcs(fits.Header(wcs_dict), silent=False)
            self._add_longslit_wcs(ad, pointing="center")

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def _fields_overlap(self, ad1, ad2, frac_FOV=1.0):
        offset = ad1.phu["QOFFSET"] - ad2.phu["QOFFSET"]
        print(ad1.filename, ad2.filename, offset)
        return abs(offset) <= 1.0

    def _get_linelist(self, wave_model=None, *args, **kwargs):
        """
        Returns a list of wavelengths of the arc reference lines used by the
        primitive `determineWavelengthSolution()`, if the user parameter
        `linelist=None` (i.e., the default list is requested).

        Parameters
        ----------
        wave_model : astroy.modeling.models.Chebyshev1D instance
            model (with domain) defining the wavelength (range) required

        Returns
        -------
        gempy.library.wavecal.LineList object
            arc line wavelengths (and optional weights)
        """
        lookup_dir = os.path.dirname(import_module('.__init__',
                                                   self.inst_lookups).__file__)
        filename = os.path.join(lookup_dir, 'linelist.dat')
        return [wavecal.LineList(os.path.join(lookup_dir, filename))
                for filename in ('OH.dat', 'HITRAN.dat')]
