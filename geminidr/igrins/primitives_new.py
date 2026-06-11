import os
from importlib import import_module

import numpy as np
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy import units as u

from gwcs.wcs import WCS as gWCS
from gwcs import coordinate_frames as cf

import astrodata
from astrodata import wcs as adwcs
from gempy.gemini import gemini_tools as gt
from gempy.library import astromodels as am
from gempy.library import peak_finding, tracing, transform, wavecal

from geminidr.gemini.lookups import DQ_definitions as DQ
from gempy.library.astromodels import reduce_dimensionality
from .primitives_igrins import IGRINS
from ..core.primitives_crossdispersed import CrossDispersed, Spect

from recipe_system.utils.decorators import parameter_override
from . import parameters_new

from .procedures.apertures import Apertures
from .procedures.correct_distortion import get_rectified_2dspec
from .procedures.flexure_correction import estimate_flexure
from .procedures.iraf_helper import get_wat_header
from .procedures.readout_pattern.readout_pattern_helper import remove_pattern


@parameter_override
class IGRINSNew(IGRINS, CrossDispersed, Spect):
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
            mdf_table['slitlength_asec'] = slitlen_pix * ad.pixel_scale()[0]
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
        #self.writeOutputs(adinputs, suffix="_intermediateCut", strip=True)
        x = np.arange(2048)
        for ad in adinputs:
            for ext in ad:
                edge_models = [am.table_to_model(row) for row in ext.SLITEDGE]
                t = ext.wcs.get_transform('pixels', 'rectified')
                y1 = edge_models[0](x)
                y2 = edge_models[1](x)
                center = 0.5 * np.mean(t(x, y2)[1] + t(x, y1)[1])
                width = np.mean(t(x, y2)[1] - t(x, y1)[1])  # of rectified slit
                m = models.Identity(1) & (models.Shift(-center) | models.Scale(1. / width))
                ext.wcs.insert_transform("rectified", m, after=False)
                wave_model = am.get_named_submodel(ext.wcs.forward_transform, "WAVE")
                sky_model = am.get_named_submodel(ext.wcs.forward_transform, "SKY")
                current_pixscale = 3600 * np.sqrt(np.linalg.det(sky_model[-3].matrix))
                sky_model[-3].matrix *= 5. / current_pixscale  # slit is 5 arcsec
                # We then want to remove all shifts (there'll be the CRPIX2
                # shift, and a second shift from the origin reset after cutting).
                # We do it like this because the CD matrix may have been rotated
                # by 90 degrees if saved as FITS and read back in.
                for m in sky_model:
                    if m.__class__ == models.Shift:
                        m.offset = 0
                ext.wcs.set_transform("rectified", "world", wave_model & sky_model)

                # super().applySlitModel() will have done the housekeeping

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

    def createDataCube(self, adinputs=None, **params):
        """
        Create the data cube.

        This is currently just saveTwodspec but taking things from the main
        stream.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        #timestamp_key = self.timestamp_keys[self.myself()]
        suffix = params["suffix"]
        height_2dspec = params["height_2dspec"]
        conserve_flux = True
        # height_2dspec = 100 # obsset.get_recipe_parameter("height_2dspec")
        wavelength_increasing_order = params["wavelength_increasing_order"]

        adoutputs = []
        for ad in adinputs:
            ad_sky = self._get_ad_sky(ad)
            wat_table = ad_sky[0].WAT_HEADER

            # make sure you apply convert_data to the output. If get_wat_header is
            # called with wavelength_increasing_order=True, convert_data will rearrange
            # the data to the correct order.
            wvl_header, convert_data = get_wat_header(wat_table,
                                                      wavelength_increasing_order)

            ordermap = ad_sky[0].ORDERMAP
            # FIXME we should use proper badpixel mask.
            ordermap_bpixed = np.ma.array(ordermap, mask=ad_sky[0].mask).filled(0)
            ap = Apertures(ad_sky[0].SLITEDGE)

            _ = get_rectified_2dspec(ad[0].data, ordermap_bpixed, ap,  # bottom_up_solutions,
                                     conserve_flux=conserve_flux, height=height_2dspec)
            d0_shft_list, msk_shft_list, height = _
            with np.errstate(invalid="ignore"):
                d = np.array(d0_shft_list) / np.array(msk_shft_list)

            d = convert_data(d.astype("float32"))
            hdu_spec2d = fits.ImageHDU(header=wvl_header, data=d)

            ad_out = astrodata.create(ad.phu)
            ad_out.append(hdu_spec2d)

            _ = get_rectified_2dspec(ad[0].variance, ordermap, ap,  # bottom_up_solutions,
                                     conserve_flux=conserve_flux, height=height)
            d0_shft_list, msk_shft_list, _ = _
            with np.errstate(invalid="ignore"):
                d = np.array(d0_shft_list) / np.array(msk_shft_list)

            ad_out[0].variance = d.astype(np.float32)
            ad_out[0].WAVELENGTHS = np.array(ad_sky[0].WVLSOL["wavelengths"], dtype=np.float32)

            ad_out.update_filename(suffix=suffix, strip=True)
            adoutputs.append(ad_out)

        return adoutputs

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
        spatial_order = params["spatial_order"]
        spectral_order = params["spectral_order"]
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

        for ad in adinputs:
            data_to_fit = []
            xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
            for ext in ad:
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
                spec_order = ext.hdr['SPECORDR']
                extname = f'{ad.filename} Order {spec_order}'
                start = ext.shape[1 - dispaxis] // 2

                # This is identical to the code in determineWavelengthSolution()
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

                # Find peaks; convert width FWHM to sigma
                widths = 0.42466 * fwidth * np.arange(0.75, 1.26, 0.05)  # TODO!
                initial_peaks, peak_values, _ = peak_finding.find_wavelet_peaks(
                    data, widths=widths, mask=mask & DQ.not_signal,
                    variance=variance, min_snr=min_snr, reject_bad=debug_reject_bad)

                if len(initial_peaks):
                    rwidth = 0.42466 * fwidth
                    slit_length_frac = 50 / ext.shape[1 - dispaxis]

                    traces = []
                    for peak, peak_value in zip(initial_peaks, peak_values):
                        # Need to start midway along the slit, which varies
                        # along the dispersion axis. `extract_info` here is the
                        # polynomial describing that midway line.
                        start = extract_info(peak)
                        traces.extend(tracing.trace_lines(
                            ext, axis=1 - dispaxis,
                            start=start, initial=[peak],
                            rwidth=rwidth, halfwidth=max(fwidth // 2, 2), step=step,
                            nsum=nsum, max_missed=max_missed,
                            max_shift=max_shift * ybin / xbin,
                            viewer=self.viewer if debug else None,
                            min_line_length=min_line_length * slit_length_frac,
                            min_peak_value=min_relative_height * peak_value))

                    # Remove traces with too few points
                    traces = [trace for trace in traces if len(trace) >= min_points]
                    log.stdinfo(f"Traced {len(traces)} lines from "
                                f"{len(initial_peaks)} peaks")

                    # Traces can extend beyond the edge of the illuminated slit
                    # if adaptive binning has occurred and there's a feature in
                    # the unilluminated region (for GNIRS notably the quadrant
                    # boundary halfway up the detector)
                    edge_models = [am.table_to_model(row) for row in ext.SLITEDGE]
                    # Go through each trace, find the location of the slit
                    # edge across from each point, and remove the point if
                    # it's beyond the edge (with a small tolerance).
                    for trace in traces:
                        inco = trace.input_coordinates(reverse=False)
                        limits = np.asarray([m([point[1] for point in inco])
                                             for m in edge_models]).T
                        for point, lim in zip(inco, limits):
                            if (point[0] < min(lim) - 0.5 * step or
                                    point[0] > max(lim) + 0.5 * step):
                                trace.remove_point(point)

                    # List of traced peak positions
                    in_coords = np.array([coord for trace in traces for
                                          coord in trace.input_coordinates()]).T
                    if in_coords.size == 0:
                        log.warning("Failed to trace any lines for "
                                    f"{ad.filename}:{ext.id}")
                        continue

                else:
                    log.warning("Failed to find any peaks in "
                                f"{ad.filename}:{ext.id}")
                    continue

                # List of "reference" positions (i.e., the coordinate
                # perpendicular to the line remains constant at its initial value
                ref_coords = np.array([coord for trace in traces for
                                       coord in trace.reference_coordinates()]).T

                # Convert all coordinates from peaks to centroids
                in_coords = np.asarray(convert_to_centroid(*in_coords))
                ref_coords = np.asarray(convert_to_centroid(*ref_coords))

                shift = in_coords[1-dispaxis] - ref_coords[1-dispaxis]
                slitpos = ((in_coords[dispaxis] - edge_models[0](in_coords[1-dispaxis])) /
                           (edge_models[1](in_coords[1-dispaxis]) - edge_models[0](in_coords[1-dispaxis])))
                data_to_fit.extend(list(zip(*[ref_coords[1-dispaxis], [spec_order] * len(shift), slitpos-0.5, shift])))

            x = np.asarray(data_to_fit).T
            linefit = Table([y for y in x], names=["initial_mean_pixel", "order", "slit_center", "shift"],
                            dtype=[np.float32, int, np.float32, np.float32])
            ad.LINEFIT = linefit
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def determineSlitEdgesNew(self, adinputs=None, **params):
        return Spect([]).determineSlitEdges(adinputs, **params)

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
        sfx = params["suffix"]
        debug_reject_bad = False

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
        ref_sigpx = np.hstack([[1.5] * len(l) if l.weights is None else l.weights
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

        fit_it = fitting.TRFLSQFitter()
        npix = 2048
        pixels = np.arange(npix)
        xcorr = np.zeros((2*npix-1,))
        for ad in adinputs:
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
            loop_pass = 0
            for ext, spec, spec_order in zip(ad, spectra, ad.hdr['SPECORDR']):
                loop_pass += 1
                wave_model = (models.Shift(global_shift) |
                              am.get_named_submodel(ext.wcs.forward_transform, "WAVE"))
                in_order = np.logical_and(ref_waves >= wave_model(0),
                                          ref_waves <= wave_model(npix-1))
                if in_order.sum() == 0:
                    continue

                # Get the lines that appear in this order, and group them if
                # they're close together so that we can fit them simultaneously
                order_waves = ref_waves[in_order]
                pixloc = wave_model.inverse(order_waves)
                groups = _create_fitting_groups(pixloc, ref_sigpx[in_order])
                log.stdinfo(f"Order {spec_order}: Fitting {len(order_waves)} "
                            f"lines in {len(groups)} groups")

                for group in groups:
                    x1 = max(int(pixloc[group[0]] - 5*ref_sigpx[group[0]] - 5), 0)
                    x2 = min(int(pixloc[group[-1]] + 5*ref_sigpx[group[-1]] + 6), npix-1)
                    x = np.arange(x1, x2)
                    y = np.ma.masked_array(spec.data[x1:x2], mask=spec.mask[x1:x2])
                    m_init = models.Polynomial1D(degree=1)  # continuum
                    for i, line in enumerate(group):
                        g = models.Gaussian1D(amplitude=spec.data[int(pixloc[line])],
                                              mean=pixloc[line], stddev=ref_sigpx[line])
                        if i > 1:  # fix the separations of the lines
                            g.mean.tied = lambda m: m.mean_1 + (pixloc[line] - pixloc[group[0]])
                        else:
                            g.mean.bounds=(pixloc[line] - 5, pixloc[line] + 5)
                        g.stddev.bounds=(0.5 * ref_sigpx[line], 2 * ref_sigpx[line])
                        m_init += g
                    m_final = fit_it(m_init, x, y, maxiter=1000)
                    linestr = "+".join(f"{order_waves[line]:.2f}" for line in group)
                    if not fit_it.fit_info.success:
                        print(fit_it.fit_info.message)
                        continue

                    # Check that line passed min_snr requirement
                    lines_only = m_final(x) - m_final[0](x)
                    snr = lines_only.max() / np.sqrt(spec.variance[x[lines_only.argmax()]])
                    if snr < min_snr:
                        log.stdinfo(f"Rejecting line(s) at {linestr}nm in "
                                  f"order {spec_order} with SNR={snr:.1f}")
                        continue

                    # Store order, pixel location, and wavelength. For a group
                    # with multiple lines, we store the mean pixel location and
                    # mean wavelength, since we've only fit a single parameter.
                    pixel = np.mean([getattr(m_final, f"mean_{i+1}").value
                                       for i in range(len(group))])
                    wavelength = np.mean([order_waves[line] for line in group])
                    log.stdinfo(f"Adding ({spec_order}, {pixel:.2f}, {linestr})"
                              " to fitted line list")
                    fitted_lines.append((spec_order, pixel, wavelength))

            # Now perform the fit
            orders, pixels, waves = [np.asarray(x) for x in zip(*fitted_lines)]
            fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                                       outlier_func=sigma_clip)
            m_init = models.Chebyshev2D(x_degree=xdeg, y_degree=ydeg,
                                        x_domain=(0, 2047),
                                        y_domain=(min(orders), max(orders)))
            m_final, mask = fit_it(m_init, pixels, orders, waves*orders)

            # This code is comparable to update_wcs_with_solution()
            tbl_orders = orders[~mask].astype(int)
            tbl_pixels = pixels[~mask].astype(np.float32)
            tbl_waves = waves[~mask].astype(np.float32)
            rms = np.std(m_final(tbl_pixels, tbl_orders) / tbl_orders - tbl_waves)
            tbl_pixels += 1  # use 1-based for output

            temptable = am.model_to_table(m_final, xunit=u.pixel, yunit=None)
            temptable.add_columns([[2], [xdeg], [ydeg], [0], [2047], [min(orders)], [max(orders)], [rms]],
                                  names=("ndim", "xdegree", "ydegree", "xdomain_start",
                                         "xdomain_end", "ydomain_start", "ydomain_end", "rms"))

            pad_rows = tbl_orders.size - len(temptable.colnames)
            if pad_rows < 0:
                tbl_orders = list(tbl_orders) + [0] * (-pad_rows)
                tbl_pixels = list(tbl_pixels) + [0] * (-pad_rows)
                tbl_waves = list(tbl_waves) + [0] * (-pad_rows)
                pad_rows = 0

            fit_table = Table([temptable.colnames + [''] * pad_rows,
                               list(temptable[0].values()) + [0] * pad_rows,
                               tbl_orders, tbl_pixels, tbl_waves],
                              names=("name", "coefficients", "xdorder", "peaks", "wavelengths"),
                              units=(None, None, None, u.pix, u.nm),
                              meta=temptable.meta)
            medium = "vacuo" if in_vacuo else "air"
            fit_table.meta['comments'] = [
                'coefficients are based on 0-indexing',
                'peaks column is 1-indexed',
                f'calibrated with wavelengths in {medium}']
            ad.WAVECAL = fit_table

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

                cheb1d = reduce_dimensionality(m_final, y=spec_order)
                for p, v in zip(cheb1d.param_names, cheb1d.parameters):
                    if p.startswith("c"):
                        setattr(cheb1d, p, v / spec_order)
                cheb1d.inverse = am.make_inverse_chebyshev1d(cheb1d, max_deviation=0.01)
                cheb1d.name = "WAVE"
                transform = cheb1d & models.Scale(5.0)  # slit is 5 arcsec
                ext.wcs = gWCS(ext.wcs.pipeline[:-2] +
                               [(ext.wcs.pipeline[-2].frame, transform),
                                (output_frame, None)])

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def distortionCorrect(self, adinputs=None, **params):
        """
        Corrects the wavelength distortion by shifting all rows so that the
        lines of constant wavelength become vertical. This can currently use
        the IGRINS-2 PLP code or the DRAGONS transform module.

        TODO: This can ultimately be removed once the WCS is properly
        implemented.

        NB. This also flatfields the data.

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
        dq_threshold = params["dq_threshold"]
        use_dragons = params["use_dragons"]

        adoutputs = []
        for ad in adinputs:
            ad_sky = self._get_ad_sky(ad)
            ap = Apertures(ad_sky[0].SLITEDGE)

            if use_dragons:  # use existing DRAGONS transform module
                x = np.meshgrid(*(np.arange(length) for length in ad[0].shape[::-1]))[1]
                t = models.Identity(2)  # ensure array size doesn't change
                t.inverse = (models.Mapping((0, 1, 1)) |
                             models.Tabular2D(lookup_table=x+ad_sky[0].SLITOFFSETMAP.T, bounds_error=False,
                                              fill_value=0) & models.Identity(1))
                if ad[0].wcs is None:
                    ad[0].wcs = gWCS([(astrodata.wcs.pixel_frame(naxes=2), t),
                                      (astrodata.wcs.pixel_frame(naxes=2, name="xshifted"), None)])
                else:
                    ad[0].wcs = gWCS([(ad[0].wcs.pipeline[0].frame, t),
                                      (astrodata.wcs.pixel_frame(naxes=2, name="xshifted"),
                                       ad[0].wcs.pipeline[0].transform),
                                      ] + ad[0].wcs.pipeline[1:])

                ad_out = transform.resample_from_wcs(
                    ad, 'xshifted', interpolant=interpolant,
                    subsample=subsample, parallel=False,
                    threshold=dq_threshold
                )

            else:
                _ = ap.get_shifted_images(ad[0].SLITPROFILE_MAP,
                                          ad[0].variance, ad[0].data,
                                          slitoffset_map=ad_sky[0].SLITOFFSETMAP,
                                          debug=False)
                data_shft, variance_map_shft, profile_map_shft, msk1_shft = _
                ad_out = astrodata.create(ad.phu)
                new_image = ad[0].nddata.__class__(data=data_shft.astype(np.float32),
                                                   mask=(~msk1_shft).astype(ad[0].mask.dtype),
                                                   variance=variance_map_shft.astype(np.float32))
                ad_out.append(new_image)
                ad_out[0].SLITPROFILE_MAP = profile_map_shft

            # Timestamp and update the filename
            gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
            ad_out.update_filename(suffix=sfx, strip=True)
            adoutputs.append(ad_out)

        return adoutputs

    def extractSpectrumUsingProfile(self, adinputs=None, **params):
        """
        Extract 1D stellar spectra from 2D spectral data using optimal extraction.

        This method performs optimal extraction of stellar spectra from 2D
        spectral data, taking into account the spatial profile of the star
        and the noise characteristics of the detector. The extraction can be
        performed using different methods and parameters to optimize the
        signal-to-noise ratio.

        The method performs the following steps:
        1. Loads flat field and sky data for calibration
        2. Applies flat field correction
        3. Performs optimal extraction using the specified method
        4. Calculates wavelength solution and signal-to-noise ratios
        5. Returns the extracted 1D spectrum with associated metadata

        Returns
        -------
        AstroData
            A new AstroData object containing the extracted 1D spectrum with
            the following extensions:
            - Primary HDU: The extracted 1D spectrum
            - Variance array: The variance of the extracted spectrum
            - Wavelengths: The wavelength solution for the spectrum
            - SN_PER_RESEL: Signal-to-noise ratio per resolution element

        Notes
        -----
        - The method requires flat field and sky data to be available through
          the `_get_ad_flat` and `_get_ad_sky` methods.
        - The extraction uses the SLITEDGE information to define the extraction
          apertures.
        - The wavelength solution is taken from the WVLFIT_RESULTS attribute
          of the sky data.
        - The output spectrum includes WCS information in the header for
          wavelength calibration.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys["extractSpectra"]
        sfx = params["suffix"]
        extraction_mode = params["extraction_mode"]
        pixel_per_res_element = params["pixel_per_res_element"]

        adoutputs = []
        for ad in adinputs:
            ad_flat = self._get_ad_flat(ad)
            ad_sky = self._get_ad_sky(ad)

            ap = Apertures(ad_sky[0].SLITEDGE)
            ordermap = ad_sky[0].ORDERMAP
            ordermap_bpixed = np.ma.array(ordermap, mask=ad_flat[0].mask > 0).filled(0)

            weight_thresh = None
            remove_negative = False
            s_list, v_list = ap.extract_stellar_from_shifted(
                ordermap_bpixed, ad[0].SLITPROFILE_MAP, ad[0].variance,
                ad[0].data, ~(ad[0].mask.astype(bool)), weight_thresh=weight_thresh,
                remove_negative=remove_negative)

            ad_out = astrodata.create(ad.phu)
            new_image = ad[0].nddata.__class__(data=np.array(s_list, dtype=np.float32),
                                               variance=np.array(v_list, dtype=np.float32))
            ad_out.append(new_image)

            # Timestamp and update the filename
            gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
            ad_out.update_filename(suffix=sfx, strip=True)
            adoutputs.append(ad_out)

        return adoutputs

    def flagDiscrepantPixels(self, adinputs=None, **params):
        """
        Flag discrepant pixels in the extracted spectrum.

        This method identifies and flags pixels in the extracted 1D spectrum
        that are discrepant based on a specified threshold. The flagged pixels
        can be used to exclude them from further analysis or to apply special
        handling during subsequent processing steps.

        Parameters
        ----------
        threshold : float
            The threshold for identifying discrepant pixels. Pixels with values
            that deviate from the median by more than this threshold will be flagged.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        #timestamp_key = self.timestamp_keys["flagDiscrepantPixels"]
        sfx = params["suffix"]
        discrepant_pixel_threshold = params["discrepant_pixel_threshold"]

        for ad in adinputs:
            for ext in ad:
                discrepant_mask = np.where(np.abs(ext.data - ext.SYNTHMAP) /
                                           np.sqrt(ext.variance) > discrepant_pixel_threshold,
                                           DQ.cosmic_ray, DQ.good)
                if ext.mask is None:
                    ext.mask = discrepant_mask.astype(DQ.datatype)
                else:
                    ext.mask |= discrepant_mask

            # Timestamp and update the filename
            #gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def flatCorrect(self, adinputs=None, suffix=None, flat=None, do_cal=None):
        # We have to delete the mask and variance because IGRINSDR doesn't do
        # anything with these and they're probably junk.
        for ad in adinputs:
            ad_flat = self._get_ad_flat(ad)
            ad_flat[0].mask = None
            ad_flat[0].variance = None
            ad.divide(ad_flat)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def makeABNew(self, adinputs=None, **params):
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
        remove_level = params["remove_level"]
        remove_amp_wise_var = params["remove_amp_wise_var"]
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

        log.stdinfo("Exposures in group A: {}".format([ad.filename for ad, in_a in zip(adinputs, in_group_a) if in_a]))
        log.stdinfo("Exposures in group B: {}".format([ad.filename for ad, in_a in zip(adinputs, in_group_a) if not in_a]))
        adinputsA = [ad for ad, in_a in zip(adinputs, in_group_a) if in_a]
        adinputsB = [ad for ad, in_a in zip(adinputs, in_group_a) if not in_a]
        stackedA = self.stackFrames(adinputsA).pop()
        stackedB = self.stackFrames(adinputsB).pop()

        ad_sky = self._get_ad_sky(adinputs[0])
        mask = ad_sky[0].ORDERMAP != 0

        # FIXME should we better to create a new instance of AstroData?
        ad = stackedA.subtract(stackedB)
        ad[0].data = remove_pattern(stackedA[0].data, mask=mask,
                                    remove_level=remove_level,
                                    remove_amp_wise_var=remove_amp_wise_var)
        ad[0].variance = remove_pattern(stackedA[0].variance, remove_level=1,
                                        remove_amp_wise_var=False)

        # Timestamp and update filename
        #gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
        ad.update_filename(suffix=suffix, strip=True)

        return [ad]

    def makeSyntheticImage(self, adinputs=None, **params):
        """
        Make a synthetic 2D spectrum image based on the slit profile and order map.

        Parameters
        ----------
        suffix : str
            Suffix to be added to output files.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        sfx = params["suffix"]

        adoutputs = []
        for ad in adinputs:
            ad_sky = self._get_ad_sky(ad)
            ap = Apertures(ad_sky[0].SLITEDGE)

            synth_map = ap.make_synth_map(ad_sky[0].ORDERMAP, ad_sky[0].SLITPOSMAP,
                                          ad[0].SLITPROFILE_MAP, ad[0].data,
                                          slitoffset_map=ad_sky[0].SLITOFFSETMAP)

            adout = astrodata.create(ad.phu)
            adout.append(synth_map.astype(np.float32))
            adout.update_filename(suffix=sfx, strip=True)
            adoutputs.append(adout)

        return adoutputs

    def normalizeFlatNew(self, adinputs=None, **params):
        return Spect([]).normalizeFlat(adinputs, **params)

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
