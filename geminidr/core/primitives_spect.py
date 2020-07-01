# Copyright(c) 2016-2020 Association of Universities for Research in Astronomy, Inc.
#
#
#                                                                  gemini_python
#
#                                                             primtives_spect.py
# ------------------------------------------------------------------------------
import os
import re
from copy import copy, deepcopy
from datetime import datetime
from importlib import import_module

import numpy as np
from astropy import units as u
from astropy import visualization as vis
from astropy.io.registry import IORegistryError
from astropy.io.ascii.core import InconsistentTableError
from astropy.modeling import models, fitting, Model
from astropy.stats import sigma_clip
from astropy.table import Table
from gwcs import coordinate_frames as cf
from gwcs.wcs import WCS as gWCS
from matplotlib import pyplot as plt
from matplotlib import gridspec
from numpy.ma.extras import _ezclump
from scipy import spatial, optimize
from scipy.interpolate import BSpline
from specutils import SpectralRegion
from functools import reduce

import astrodata
from astrodata import NDAstroData
from geminidr import PrimitivesBASE
from geminidr.gemini.lookups import DQ_definitions as DQ, extinction_data as extinct
from gempy.gemini import gemini_tools as gt
from gempy.library import astromodels, matching, tracing
from gempy.library import transform
from gempy.library.astrotools import array_from_list
from gempy.library.nddops import NDStacker
from gempy.library.spectral import Spek1D
from recipe_system.utils.decorators import parameter_override
from . import parameters_spect


# ------------------------------------------------------------------------------
@parameter_override
class Spect(PrimitivesBASE):
    """
    This is the class containing all of the pre-processing primitives
    for the `Spect` level of the type hierarchy tree.
    """
    tagset = {"GEMINI", "SPECT"}

    def __init__(self, adinputs, **kwargs):
        super().__init__(adinputs, **kwargs)
        self._param_update(parameters_spect)

    def adjustWCSToReference(self, adinputs=None, **params):
        """
        Compute offsets along the slit by cross-correlation, or use offset
        from the headers (QOFFSET). The computed offset is stored in the
        SLITOFF keyword.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Wavelength calibrated 1D or 2D spectra. Each extension must have a
            `.WAVECAL` table.
        suffix : str
            Suffix to be added to output files
        method : str ['correlation' | 'offsets']
            Method to use to compute offsets. 'correlation' uses a
            correlation of the slit profiles (the 2d images stacked
            on the dispersion axis), 'offsets' uses the QOFFSET keyword.
        tolerance : float
            Maximum distance from the header offset, for the correlation
            method (arcsec). If the correlation computed offset is too
            different from the header offset, then the latter is used.

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        methods = (params["method"], params["fallback"])
        tolerance = params["tolerance"]

        if len(adinputs) <= 1:
            log.warning("No correction will be performed, since at least two "
                        "input images are required")
            return adinputs

        if not all(len(ad) == 1 for ad in adinputs):
            raise OSError("All input images must have only one extension")

        if {len(ad[0].shape) for ad in adinputs} != {2}:
            raise OSError("All inputs must be two dimensional")

        def stack_slit(ext):
            dispaxis = 2 - ext.dispersion_axis()  # python sense
            data = np.ma.array(ext.data, mask=(ext.mask > 0))
            data = np.ma.masked_invalid(data)
            return data.mean(axis=dispaxis)

        # Use first image in list as reference
        refad = adinputs[0]
        ref_sky_model = astromodels.get_named_submodel(refad[0].wcs.forward_transform, 'SKY').copy()
        ref_sky_model.name = None
        log.stdinfo("Reference image: {}".format(refad.filename))
        refad.phu['SLITOFF'] = 0
        if any('sources' in m for m in methods):
            ref_profile = stack_slit(refad[0])
        if 'sources_wcs' in methods:
            world_coords = (refad[0].central_wavelength(asNanometers=True),
                            refad.target_ra(), refad.target_dec())
            ref_coords = refad[0].wcs.backward_transform(*world_coords)

        # The reference doesn't go through the loop so update it now
        gt.mark_history(adinputs[0], primname=self.myself(), keyword=timestamp_key)
        adinputs[0].update_filename(suffix=params["suffix"], strip=True)

        for ad in adinputs[1:]:
            for method in methods:
                adjust = True  # optimistic expectation
                dispaxis = 2 - ad[0].dispersion_axis()  # python sense

                # Calculate offset determined by header (WCS or offsets)
                if method == 'sources_wcs':
                    coords = ad[0].wcs.backward_transform(*world_coords)
                    hdr_offset = ref_coords[dispaxis] - coords[dispaxis]
                elif dispaxis == 1:
                    hdr_offset = refad.detector_y_offset() - ad.detector_y_offset()
                else:
                    hdr_offset = refad.detector_x_offset() - ad.detector_x_offset()

                # Cross-correlate to find real offset and compare
                if 'sources' in method:
                    profile = stack_slit(ad[0])
                    corr = np.correlate(ref_profile, profile, mode='full')
                    peak = tracing.pinpoint_peaks(corr, None, [np.argmax(corr)])[0]
                    offset = peak - ref_profile.shape[0] + 1

                    # Check that the offset is similar to the one from headers
                    offset_diff = hdr_offset - offset
                    if (tolerance is not None and
                            np.abs(offset_diff * ad.pixel_scale()) > tolerance):
                        log.warning("Offset for {} ({:.2f}) disagrees with "
                                    "expected value ({:.2f})".format(
                            ad.filename, offset, hdr_offset))
                        adjust = False
                elif method == 'offsets':
                    offset = hdr_offset

                if adjust:
                    wcs = ad[0].wcs
                    frames = wcs.available_frames
                    for input_frame, output_frame in zip(frames[:-1], frames[1:]):
                        t = wcs.get_transform(input_frame, output_frame)
                        try:
                            sky_model = astromodels.get_named_submodel(t, 'SKY')
                        except IndexError:
                            pass
                        else:
                            new_sky_model = models.Shift(offset) | ref_sky_model
                            new_sky_model.name = 'SKY'
                            ad[0].wcs.set_transform(input_frame, output_frame,
                                                    t.replace_submodel('SKY', new_sky_model))
                            break
                    else:
                        raise OSError("Cannot find 'SKY' model in WCS for "
                                      f"{ad.filename}")

                    log.stdinfo("Offset for image {} : {:.2f} pixels"
                                .format(ad.filename, offset))
                    ad.phu['SLITOFF'] = offset
                    break

            if not adjust:
                no_offset_msg = f"Cannot determine offset for {ad.filename}"
                if 'sq' in self.mode:
                    raise OSError(no_offset_msg)
                else:
                    log.warning(no_offset_msg)
            else:
                # Timestamp and update filename
                gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
                ad.update_filename(suffix=params["suffix"], strip=True)

        return adinputs

    def calculateSensitivity(self, adinputs=None, **params):
        """
        Calculates the overall sensitivity of the observation system
        (instrument, telescope, detector, etc) for each wavelength using
        spectrophotometric data. It is obtained using the ratio
        between the observed data and the reference look-up data.

        For that, it looks for reference data using the stripped and lower
        case name of the observed object inside :mod:`geminidr.gemini.lookups`,
        :mod:`geminidr.core.lookups` and inside the instrument lookup module.

        The reference data is fit using a Spline in order to match the input
        data sampling.

        See Also
        --------
        - :class:`~gempy.library.astromodels.UnivariateSplineWithOutlierRemoval`

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            1D spectra of spectrophotometric standard stars

        suffix :  str, optional
            Suffix to be added to output files (default: _sensitivityCalculated).

        filename: str or None, optional
            Location of spectrophotometric data file. If it is None, uses
            look up data based on the object name stored in OBJECT header key
            (default).

        order: int
            Order of the spline fit to be performed
        order: int, optional
            Order of the spline fit to be performed (default: 6)

        individual: bool - TODO - Not in calculateSensitivityConfig
            Calculate sensitivity for each AD spectrum individually?

        bandpass: float, optional
            default bandpass width (in nm) to use if not present in the
            spectrophotometric data table (default: 5.)

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            The same input list is used as output but each object now has a
            `.SENSFUNC` table appended to each of its extensions. This table
            provides details of the fit which describes the sensitivity as
            a function of wavelength.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        datafile = params["filename"]
        order = params["order"]
        bandpass = params["bandpass"]
        debug_plot = params["debug_plot"]

        # We're going to look in the generic (gemini) module as well as the
        # instrument module, so define that
        module = self.inst_lookups.split('.')
        module[-2] = 'gemini'
        gemini_lookups = '.'.join(module)

        for ad in adinputs:
            if datafile is None:
                filename = '{}.dat'.format(ad.object().lower().replace(' ', ''))
                for module in (self.inst_lookups, gemini_lookups, 'geminidr.core.lookups'):
                    try:
                        path = import_module('.', module).__path__[0]
                    except (ImportError, ModuleNotFoundError):
                        continue
                    full_path = os.path.join(path, 'spectrophotometric_standards', filename)
                    try:
                        spec_table = self._get_spectrophotometry(full_path)
                    except (FileNotFoundError, InconsistentTableError):
                        pass
                    else:
                        break
                else:
                    log.warning("Cannot read spectrophotometric data table. "
                                "Unable to determine sensitivity for {}".
                                format(ad.filename))
                    continue
            else:
                try:
                    spec_table = self._get_spectrophotometry(datafile)
                except FileNotFoundError:
                    log.warning(f"Cannot find spectrophotometric data table {datafile}."
                                f"Unable to determine sensitivity for {ad.filename}")
                    continue
                except InconsistentTableError:
                    log.warning(f"Cannot read spectrophotometric data table {datafile}."
                                f"Unable to determine sensitivity for {ad.filename}")
                    continue

            exptime = ad.exposure_time()
            if 'WIDTH' not in spec_table.colnames:
                log.warning("Using default bandpass of {} nm".format(bandpass))
                spec_table['WIDTH'] = bandpass * u.nm

            # We can only calculate the sensitivity for one extension in
            # non-XD data, so keep track of this in case it's not the first one
            calculated = False
            for ext in ad:
                if len(ext.shape) != 1:
                    log.warning("{}:{} is not a 1D spectrum".
                                format(ad.filename, ext.hdr['EXTVER']))
                    continue

                if calculated and 'XD' not in ad.tags:
                    log.warning("Found additional 1D extensions in non-XD data."
                                " Ignoring.")
                    break

                spectrum = Spek1D(ext) / (exptime * u.s)
                wave, zpt, zpt_err = [], [], []

                # Compute values that are counts / (exptime * flux_density * bandpass)
                for w0, dw, fluxdens in zip(spec_table['WAVELENGTH'].quantity,
                                            spec_table['WIDTH'].quantity, spec_table['FLUX'].quantity):
                    region = SpectralRegion(w0 - 0.5 * dw, w0 + 0.5 * dw)
                    data, mask, variance = spectrum.signal(region)
                    if mask == 0 and fluxdens > 0:
                        # Regardless of whether FLUX column is f_nu or f_lambda
                        flux = fluxdens.to(u.Unit('erg cm-2 s-1 nm-1'),
                                           equivalencies=u.spectral_density(w0)) * dw.to(u.nm)
                        wave.append(w0)
                        # This is (counts/s) / (erg/cm^2/s), in magnitudes (like IRAF)
                        zpt.append(u.Magnitude(data / flux))
                        zpt_err.append(u.Magnitude(1 + np.sqrt(variance) / data))

                # TODO: Abstract to interactive fitting
                wave = array_from_list(wave, unit=u.nm)
                zpt = array_from_list(zpt)
                zpt_err = array_from_list(zpt_err)
                spline = astromodels.UnivariateSplineWithOutlierRemoval(wave.value, zpt.value,
                                                                        w=1./zpt_err.value,
                                                                        order=order)
                knots, coeffs, degree = spline.tck
                sensfunc = Table([knots * wave.unit, coeffs * zpt.unit],
                                 names=('knots', 'coefficients'))
                ext.SENSFUNC = sensfunc
                calculated = True

                if debug_plot:
                    plt.ioff()
                    fig, ax = plt.subplots()
                    ax.plot(wave, zpt, 'bo')
                    ax.plot(wave[spline.mask], zpt[spline.mask], 'ko')
                    x = np.linspace(min(wave), max(wave), ext.shape[0])
                    ax.plot(x, spline(x), 'r-')
                    plt.show()

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
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
        adinputs : list of :class:`~astrodata.AstroData`
            Arc data as 2D spectral images with a WAVECAL table.

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

        debug: bool
            plot arc line traces on image display window?

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            The same input list is used as output but each object now has the
            appropriate `nddata.wcs` defined for each of its extensions. This
            provides details of the 2D Chebyshev fit which maps the distortion.
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
        debug = params["debug"]

        orders = (spectral_order, spatial_order)

        for ad in adinputs:
            xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
            for ext in ad:
                if debug:
                    self.viewer.display_image(ext, wcs=False)
                    self.viewer.width = 2

                dispaxis = 2 - ext.dispersion_axis()  # python sense
                direction = "row" if dispaxis == 1 else "column"

                # Here's a lot of input-checking
                extname = '{}:{}'.format(ad.filename, ext.hdr['EXTVER'])
                start = 0.5 * ext.shape[1 - dispaxis]
                initial_peaks = None
                try:
                    wavecal = ext.WAVECAL
                except AttributeError:
                    log.warning("Cannot find a WAVECAL table on {} - "
                                "identifying lines in middle {}".
                                format(extname, direction))
                else:
                    try:
                        index = list(wavecal['name']).index(direction)
                    except ValueError:
                        log.warning("Cannot find starting {} in WAVECAL "
                                    "table on {} - identifying lines in "
                                    "middle {}. Wavelength calibration may "
                                    "not be correct.".format(direction, extname,
                                                             direction))
                    else:
                        start = wavecal['coefficients'][index]
                    if id_only:
                        try:
                            # Peak locations in pixels are 1-indexed
                            initial_peaks = (ext.WAVECAL['peaks'] - 1)
                        except KeyError:
                            log.warning("Cannot find peak locations in {} "
                                        "- identifying lines in middle {}".
                                        format(extname, direction))

                # This is identical to the code in determineWavelengthSolution()
                if initial_peaks is None:
                    data, mask, variance, extract_slice = _average_along_slit(ext, center=None, nsum=nsum)
                    log.stdinfo("Finding peaks by extracting {}s {} to {}".
                                format(direction, extract_slice.start + 1, extract_slice.stop))

                    if fwidth is None:
                        fwidth = tracing.estimate_peak_width(data)
                        log.stdinfo("Estimated feature width: {:.2f} pixels".format(fwidth))

                    # Find peaks; convert width FWHM to sigma
                    widths = 0.42466 * fwidth * np.arange(0.8, 1.21, 0.05)  # TODO!
                    initial_peaks, _ = tracing.find_peaks(data, widths, mask=mask,
                                                          variance=variance, min_snr=min_snr)
                    log.stdinfo("Found {} peaks".format(len(initial_peaks)))

                # The coordinates are always returned as (x-coords, y-coords)
                ref_coords, in_coords = tracing.trace_lines(ext, axis=1 - dispaxis,
                                                            start=start, initial=initial_peaks, width=5, step=step,
                                                            nsum=nsum, max_missed=max_missed,
                                                            max_shift=max_shift * ybin / xbin,
                                                            viewer=self.viewer if debug else None)

                ## These coordinates need to be in the reference frame of a
                ## full-frame unbinned image, so modify the coordinates by
                ## the detector section
                # x1, x2, y1, y2 = ext.detector_section()
                # ref_coords = np.array([ref_coords[0] * xbin + x1,
                #                       ref_coords[1] * ybin + y1])
                # in_coords = np.array([in_coords[0] * xbin + x1,
                #                      in_coords[1] * ybin + y1])

                # The model is computed entirely in the pixel coordinate frame
                # of the data, so it could be used as a gWCS object
                m_init = models.Chebyshev2D(x_degree=orders[1 - dispaxis],
                                            y_degree=orders[dispaxis],
                                            x_domain=[0, ext.shape[1]],
                                            y_domain=[0, ext.shape[0]])
                # x_domain = [x1, x1 + ext.shape[1] * xbin - 1],
                # y_domain = [y1, y1 + ext.shape[0] * ybin - 1])
                # Find model to transform actual (x,y) locations to the
                # value of the reference pixel along the dispersion axis
                fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                                           sigma_clip, sigma=3)
                m_final, _ = fit_it(m_init, *in_coords, ref_coords[1 - dispaxis])
                m_inverse, masked = fit_it(m_init, *ref_coords, in_coords[1 - dispaxis])

                # TODO: Some logging about quality of fit
                # print(np.min(diff), np.max(diff), np.std(diff))

                if dispaxis == 1:
                    model = models.Mapping((0, 1, 1)) | (m_final & models.Identity(1))
                    model.inverse = models.Mapping((0, 1, 1)) | (m_inverse & models.Identity(1))
                else:
                    model = models.Mapping((0, 0, 1)) | (models.Identity(1) & m_final)
                    model.inverse = models.Mapping((0, 0, 1)) | (models.Identity(1) & m_inverse)

                self.viewer.color = "blue"
                spatial_coords = np.linspace(ref_coords[dispaxis].min(), ref_coords[dispaxis].max(),
                                             ext.shape[1 - dispaxis] // (step * 10))
                spectral_coords = np.unique(ref_coords[1 - dispaxis])
                for coord in spectral_coords:
                    if dispaxis == 1:
                        xref = [coord] * len(spatial_coords)
                        yref = spatial_coords
                    else:
                        xref = spatial_coords
                        yref = [coord] * len(spatial_coords)
                    mapped_coords = np.array(model.inverse(xref, yref)).T
                    if debug:
                        self.viewer.polygon(mapped_coords, closed=False, xfirst=True, origin=0)

                columns = []
                for m in (m_final, m_inverse):
                    model_dict = astromodels.chebyshev_to_dict(m)
                    columns.append(list(model_dict.keys()))
                    columns.append(list(model_dict.values()))
                # If we're genuinely worried about the two models, they might
                # have different orders and we might need to pad one
                ext.FITCOORD = Table(columns, names=("name", "coefficients",
                                                     "inv_name", "inv_coefficients"))

                # Put this model before the first step if there's an existing WCS
                if ext.wcs is None:
                    ext.wcs = gWCS([(cf.Frame2D(name="pixels"), model),
                                    (cf.Frame2D(name="world"), None)])
                else:
                    # TODO: use insert_frame here
                    ext.wcs = gWCS([(ext.wcs.input_frame, model),
                                    (cf.Frame2D(name="distortion_corrected"), ext.wcs.pipeline[0][1])]
                                   + ext.wcs.pipeline[1:])

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def distortionCorrect(self, adinputs=None, **params):
        """
        Corrects optical distortion in science frames using a `processed_arc`
        with attached distortion map (a Chebyshev2D model).

        If the input image requires mosaicking, then this is done as part of
        the resampling, to ensure one, rather than two, interpolations.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            2D spectral images.
        suffix : str
            Suffix to be added to output files.
        arc : :class:`~astrodata.AstroData` or str or None
            Arc(s) containing distortion map.
        order : int (0 - 5)
            Order of interpolation when resampling. Default: 3.
        subsample : int
            Pixel subsampling factor. Default: 1.

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Modified input objects with distortion correct applied.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        sfx = params["suffix"]
        arc = params["arc"]
        order = params["order"]
        subsample = params["subsample"]

        # Get a suitable arc frame (with distortion map) for every science AD
        if arc is None:
            self.getProcessedArc(adinputs, refresh=False)
            arc_list = self._get_cal(adinputs, 'processed_arc')
        else:
            arc_list = arc

        adoutputs = []
        # Provide an arc AD object for every science frame
        for ad, arc in zip(*gt.make_lists(adinputs, arc_list, force_ad=True)):
            # We don't check for a timestamp since it's not unreasonable
            # to do multiple distortion corrections on a single AD object

            len_ad = len(ad)
            if arc is None:
                if 'qa' in self.mode:
                    # TODO: Think about this when we have MOS/XD/IFU
                    if len(ad) == 1:
                        log.warning("No changes will be made to {}, since no "
                                    "arc was specified".format(ad.filename))
                        adoutputs.append(ad)
                    else:
                        log.warning("{} will only be mosaicked, since no "
                                    "arc was specified".format(ad.filename))
                        adoutputs.extend(self.mosaicDetectors([ad]))
                    continue
                else:
                    raise OSError('No processed arc listed for {}'.
                                  format(ad.filename))

            len_arc = len(arc)
            if len_arc not in (1, len_ad):
                log.warning("Science frame {} has {} extensions and arc {} "
                            "has {} extensions.".format(ad.filename, len_ad,
                                                        arc.filename, len_arc))
                adoutputs.append(ad)
                continue

            xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
            if arc.detector_x_bin() != xbin or arc.detector_y_bin() != ybin:
                log.warning("Science frame and arc have different binnings.")
                adoutputs.append(ad)
                continue

            # Read all the arc's distortion maps. Do this now so we only have
            # one block of reading and verifying them
            distortion_models, wave_models = [], []
            for ext in arc:
                wcs = ext.nddata.wcs

                # Any failures must be handled in the outer loop processing
                # ADs, so just set the found transforms to empty and present
                # the warning at the end
                try:
                    if 'distortion_corrected' in wcs.available_frames:
                        pipeline = wcs.pipeline
                    else:
                        distortion_models = []
                        break
                except AttributeError:
                    distortion_models = []
                    break

                # We could just pass the forward transform of the WCS, which
                # already has its inverse attached, but the code currently
                # relies on a no-op forward model to size the output correctly:
                m_distcorr = models.Identity(2)
                input_frame = wcs.input_frame
                m_distcorr.inverse = wcs.get_transform(input_frame, 'distortion_corrected').inverse
                distortion_models.append(m_distcorr)

                try:
                    wave_model = astromodels.get_named_submodel(wcs.forward_transform, 'WAVE')
                except IndexError:
                    wave_models.append(None)
                else:
                    wave_models.append(wave_model)

            if not distortion_models:
                log.warning("Could not find a 'distortion_corrected' frame "
                            f"in arc {arc.filename}:{ext.hdr['EXTVER']} - "
                            "continuing")
                continue

            # Determine whether we're producing a single-extension AD
            # or keeping the number of extensions as-is
            if len_arc == 1:
                arc_detsec = arc.detector_section()[0]
                ad_detsec = ad.detector_section()
                if len_ad > 1:
                    # We need to apply the mosaicking geometry, and add the
                    # same distortion correction to each input extension.
                    geotable = import_module('.geometry_conf', self.inst_lookups)
                    transform.add_mosaic_wcs(ad, geotable)
                    for ext in ad:
                        # TODO: use insert_frame() method
                        new_pipeline = []
                        for item in ext.wcs.pipeline:
                            if item[0].name == 'mosaic':
                                new_pipeline.extend([(item[0], m_distcorr),
                                                     (cf.Frame2D(name='distortion_corrected'), item[1])])
                            else:
                                new_pipeline.append(item)
                        ext.wcs = gWCS(new_pipeline)

                    # We need to consider the different pixel frames of the
                    # science and arc. The input->mosaic transform of the
                    # science maps to the default pixel space, but the arc
                    # will have had an origin shift before the distortion
                    # correction was calculated.
                    shifts = [c2 - c1 for c1, c2 in zip(np.array(ad_detsec).min(axis=0),
                                                        arc_detsec)]
                    xoff1, yoff1 = shifts[0] / xbin, shifts[2] / ybin  # x1, y1
                    if xoff1 or yoff1:
                        log.debug(f"Found a shift of ({xoff1},{yoff1}) "
                                  f"pixels between {ad.filename} and the "
                                  f"calibration {arc.filename}")
                    shifts = [c2 - c1 for c1, c2 in zip(np.array(ad_detsec).max(axis=0),
                                                        arc_detsec)]
                    xoff2, yoff2 = shifts[1] / xbin, shifts[3] / ybin  # x2, y2
                    nzeros = [xoff1, xoff2, yoff1, yoff2].count(0)
                    if nzeros < 2:
                        raise ValueError("I don't know how to process the "
                                         f"offsets between {ad.filename} "
                                         f"and {arc.filename}")

                    arc_ext_shapes = [(ext.shape[0] - yoff1 + yoff2,
                                       ext.shape[1] - xoff1 + xoff2) for ext in ad]
                    arc_corners = np.concatenate([transform.get_output_corners(
                        ext.wcs.get_transform(ext.wcs.input_frame, 'mosaic'),
                        input_shape=arc_shape, origin=(yoff1, xoff1))
                        for ext, arc_shape in zip(ad, arc_ext_shapes)], axis=1)
                    arc_origin = tuple(np.ceil(min(corners)) for corners in arc_corners)

                    # So this is what was applied to the ARC to get the
                    # mosaic frame to its pixel frame, in which the distortion
                    # correction model was calculated. Convert coordinates
                    # from python order to Model order.
                    origin_shift = reduce(Model.__and__, [models.Shift(-origin)
                                          for origin in arc_origin[::-1]])

                    for ext in ad:
                        ext.wcs.insert_transform('mosaic', origin_shift, after=True)
                        #ext.wcs.insert_transform('distortion_corrected',
                        #                         origin_shift.inverse, after=False)

                    # ARC and AD aren't the same size
                    if nzeros < 4:
                        ad_corners = np.concatenate([transform.get_output_corners(
                            ext.wcs.get_transform(ext.wcs.input_frame, 'mosaic'),
                            input_shape=ext.shape) for ext in ad], axis=1)
                        ad_origin = tuple(np.ceil(min(corners)) for corners in ad_corners)

                        # But a full-frame ARC and subregion AD may have different
                        # origin shifts. We only care about the one in the
                        # wavelength direction, since we need the AD to be on the
                        # same pixel basis before applying the new wave_model
                        offsets = tuple(o_ad - o_arc
                                        for o_ad, o_arc in zip(ad_origin, arc_origin))[::-1]
                        # len(arc)=1 so we only have one wave_model, but need to
                        # update the entry in the list, which gets used later
                        if wave_model is not None:
                            offset = offsets[ext.dispersion_axis()-1]
                            if offset != 0:
                                wave_model.name = None
                                wave_models[0] = models.Shift(offset) | wave_model
                                wave_models[0].name = 'WAVE'

                else:
                    # Single-extension AD, with single Transform
                    ad_detsec = ad.detector_section()[0]
                    if ad_detsec != arc_detsec:
                        if self.timestamp_keys['mosaicDetectors'] in ad.phu:
                            log.warning("Cannot distortionCorrect mosaicked "
                                        "data unless calibration has the "
                                        "same ROI. Continuing.")
                            adoutputs.append(ad)
                            continue
                        # No mosaicking, so we can just do a shift
                        m_shift = (models.Shift((ad_detsec.x1 - arc_detsec.x1) / xbin) &
                                   models.Shift((ad_detsec.y1 - arc_detsec.y1) / ybin))
                        m_distcorr = m_shift | m_distcorr
                    # TODO: use insert_frame method
                    new_pipeline = [(ad[0].wcs.input_frame, m_distcorr),
                                    (cf.Frame2D(name='distortion_corrected'), ad[0].wcs.pipeline[0][1])]
                    new_pipeline.extend(ad[0].wcs.pipeline[1:])
                    ad[0].wcs = gWCS(new_pipeline)

                ad_out = transform.resample_from_wcs(ad, 'distortion_corrected',
                                                     order=order, subsample=subsample,
                                                     parallel=False)

                if wave_model is None:
                    log.warning(f"{arc.filename} has no wavelength solution")

            else:
                log.warning("Distortion correction with multiple-extension "
                            "arcs has not been tested.")
                for i, (ext, ext_arc, dist_model) in enumerate(zip(ad, arc, distortion_models)):
                    # Shift science so its pixel coords match the arc's before
                    # applying the distortion correction
                    shifts = [c1 - c2 for c1, c2 in zip(ext.detector_section(),
                                                        ext_arc.detector_section())]
                    dist_model = (models.Shift(shifts[0] / xbin) &
                                  models.Shift(shifts[1] / ybin)) | dist_model
                    # TODO: use insert_frame method
                    new_pipeline = [(ext.wcs.input_frame, dist_model),
                                    (cf.Frame2D(name='distortion_corrected'), ext.wcs.pipeline[0][1])]
                    new_pipeline.extend(ext.wcs.pipeline[1:])
                    ext.wcs = gWCS(new_pipeline)
                    if i == 0:
                        ad_out = transform.resample_from_wcs(ext, order=order,
                                                             subsample=subsample,
                                                             parallel=False)
                    else:
                        ad_out.append(transform.resample_from_wcs(ext, order=order,
                                                                  subsample=subsample,
                                                                  parallel=False))
                    if wave_model is None:
                        log.warning(f"{arc.filename}:{0} has no wavelength "
                                    "solution".format(ext.hdr['EXTVER']))

            for i, (ext, wave_model) in enumerate(zip(ad_out, wave_models)):
                # TODO: remove this; for debugging purposes only
                if arc is not None:
                    try:
                        ad_out[i].WAVECAL = arc[i].WAVECAL
                    except AttributeError:
                        pass
                sky_model = astromodels.get_named_submodel(ext.wcs.forward_transform, 'SKY')
                if ext.dispersion_axis() == 1:
                    t = wave_model & sky_model
                else:
                    t = sky_model & wave_model
                ext.wcs = gWCS([(ext.wcs.input_frame, t),
                                (ext.wcs.output_frame, None)])
            # Timestamp and update the filename
            gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
            ad_out.update_filename(suffix=sfx, strip=True)
            adoutputs.append(ad_out)

        return adoutputs

    def determineWavelengthSolution(self, adinputs=None, **params):
        """
        Determines the wavelength solution for an ARC and stores it as an
        attached `.WAVECAL` :class:`~astropy.table.Table`.

        2D input images are converted to 1D by collapsing a slice of the image
        along the dispersion direction, and peaks are identified. These are then
        matched to an arc line list, using :class:`~gempy.library.matching.KDTreeFitter`.
        This class has a strong dependency with the `fwith` parameter and works
        better if a good initial value is provided.

        The `.WAVECAL` table contains four columns:
            ["name", "coefficients", "peaks", "wavelengths"]

        The `name` and the `coefficients` columns contain information to
        re-create an Chebyshev1D object. The `peaks` column contains the position
        of the lines that were matched to the catalogue, and the `wavelengths`
        column contains the wavelengths that were matched to that line.

        Parameters
        ----------

        adinputs : list of :class:`~astrodata.AstroData`
             Mosaicked Arc data as 2D spectral images or 1D spectra.

        suffix : str, optional
            Suffix to be added to output files
            (default: "_wavelengthSolutionDetermined").

        center : None or int, optional
            Central row/column for 1D extraction (None => use middle).

        nsum : int, optional
            Number of rows/columns to average (default: 10).

        order : int, optional
            Order of Chebyshev fitting function (default: 2).

        min_snr : float, optional
            Minimum S/N ratio in line peak to be used in fitting (default: 10).

        fwidth : None or float, optional
            Expected width of arc lines in pixels. It tells how far the
            KDTreeFitter should look for when matching detected peaks with
            reference arcs lines. If None (the default), `fwidth` is
            determined using `tracing.estimate_peak_width`.

        linelist : None or str, optional
            Name of file containing arc lines. If None, then a default look-up
            table will be used.

        weighting : {'natural', 'relative', 'none'}
            How to weight the detected peaks.

        nbright : int
            Number of brightest lines to cull before fitting (default: 0).

        plot : bool
            Enable plots for debugging.

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Updated objects with the `.WAVECAL` attribute on each appropriated
            extension.

        See Also
        --------
        :class:`~geminidr.core.primitives_visualize.Visualize.mosaicDetectors`,
        :class:`~gempy.library.matching.KDTreeFitter`,
        :class:`~gempy.library.matching.Chebyshev1DMatchBox`.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        center = params["center"]
        nsum = params["nsum"]
        order = params["order"]
        min_snr = params["min_snr"]
        fwidth = params["fwidth"]
        arc_file = params["linelist"]
        weighting = params["weighting"]
        nbright = params.get("nbright", 0)

        debug = params["debug"]
        plt.ioff()

        # TODO: This decision would prevent MOS data being reduced so need
        # to think a bit more about what we're going to do. Maybe make
        # central_wavelength() return a one-per-ext list? Or have the GMOS
        # determineWavelengthSolution() recipe check the input has been
        # mosaicked before calling super()?
        #
        # Top-level decision for this to only work on single-extension ADs
        # if not all(len(ad)==1 for ad in adinputs):
        #    raise ValueError("Not all inputs are single-extension AD objects")

        # Get list of arc lines (probably from a text file dependent on the
        # input spectrum, so a private method of the primitivesClass)
        linelists = {}
        if arc_file is not None:
            try:
                arc_lines = np.loadtxt(arc_file, usecols=[0])
            except (OSError, TypeError):
                log.warning("Cannot read file {} - using default linelist".format(arc_file))
                arc_file = None
            else:
                log.stdinfo("Read arc line list {}".format(arc_file))
                try:
                    arc_weights = np.sqrt(np.loadtxt(arc_file, usecols=[1]))
                except IndexError:
                    arc_weights = None
                else:
                    log.stdinfo("Read arc line relative weights")

        for ad in adinputs:
            log.info("Determining wavelength solution for {}".format(ad.filename))
            for ext in ad:
                if len(ad) > 1:
                    log.info("Determining solution for EXTVER {}".format(ext.hdr['EXTVER']))

                # Determine direction of extraction for 2D spectrum
                if ext.data.ndim > 1:
                    dispaxis = 2 - ext.dispersion_axis()  # python sense
                    direction = "row" if dispaxis == 1 else "column"
                    data, mask, variance, extract_slice = _average_along_slit(ext, center=center, nsum=nsum)
                    log.stdinfo("Extracting 1D spectrum from {}s {} to {}".
                                format(direction, extract_slice.start + 1, extract_slice.stop))
                else:
                    data = ext.data
                    mask = ext.mask
                    variance = ext.variance

                # Mask bad columns but not saturated/non-linear data points
                if mask is not None:
                    mask = mask & (65535 ^ (DQ.saturated | DQ.non_linear))
                    data[mask > 0] = 0.

                cenwave = params["central_wavelength"] or ext.central_wavelength(asNanometers=True)
                dw = params["dispersion"] or ext.dispersion(asNanometers=True)
                w1 = cenwave - 0.5 * len(data) * abs(dw)
                w2 = cenwave + 0.5 * len(data) * abs(dw)
                log.stdinfo("Using central wavelength {:.1f} nm and dispersion "
                            "{:.3f} nm/pixel".format(cenwave, dw))

                if fwidth is None:
                    fwidth = tracing.estimate_peak_width(data)
                    log.stdinfo("Estimated feature width: {:.2f} pixels".format(fwidth))

                # Don't read linelist if it's the one we already have
                # (For user-supplied, we read it at the start, so don't do this at all)
                if arc_file is None:
                    arc_lines, arc_weights = self._get_arc_linelist(ext, w1=w1, w2=w2, dw=dw)

                # arc_weights = None

                if min(arc_lines) > cenwave + 0.5 * len(data) * abs(dw):
                    log.warning("Line list appears to be in Angstroms; converting to nm")
                    arc_lines *= 0.1

                # Find peaks; convert width FWHM to sigma
                widths = 0.42466 * fwidth * np.arange(0.8, 1.21, 0.05)  # TODO!
                peaks, peak_snrs = tracing.find_peaks(data, widths, mask=mask,
                                                      variance=variance, min_snr=min_snr)
                log.stdinfo('{}: {} peaks and {} arc lines'.
                            format(ad.filename, len(peaks), len(arc_lines)))

                # Compute all the different types of weightings so we can
                # change between them as needs require
                weights = {'none': np.ones((len(peaks),)),
                           'natural': peak_snrs}

                # The "relative" weights compares each line strength to
                # those of the lines close to it
                tree = spatial.cKDTree(np.array([peaks]).T)
                # Find lines within 10% of the array size
                indices = tree.query(np.array([peaks]).T, k=10,
                                     distance_upper_bound=abs(0.1 * len(data) * dw))[1]
                snrs = np.array(list(peak_snrs) + [np.nan])[indices]
                # Normalize weights by the maximum of these lines
                weights['relative'] = peak_snrs / np.nanmedian(snrs, axis=1)

                plot_data = data

                # Some diagnostic plotting
                yplot = 0
                plt.ioff()

                # The very first model is called "m_final" because at each
                # iteration the next initial model comes from the fitted
                # (final) model of the previous iteration
                m_final = models.Chebyshev1D(degree=1, c0=cenwave,
                                             c1=0.5 * dw * len(data), domain=[0, len(data) - 1])
                if debug:
                    plot_arc_fit(data, peaks, arc_lines, arc_weights, m_final, "Initial model")
                log.stdinfo('Initial model: {}'.format(repr(m_final)))

                kdsigma = fwidth * abs(dw)
                peaks_to_fit = peak_snrs > min_snr
                peaks_to_fit[np.argsort(peak_snrs)[len(peaks) - nbright:]] = False

                # Temporary code to help with testing
                try:
                    sequence = self.fit_sequence
                except AttributeError:

                    # FixMe: toggle commented lines bellow to make tests pass

                    sequence = (((1, 'none', 'basinhopping', ['c1']),
                                 (1, 'none', 'basinhopping', ['c0'])) +
                                tuple((order, 'none', 'Nelder-Mead') for order in range(2, order+1)))

                    # sequence = (((1, 'none', 'basinhopping', ['c1']), (2, 'none', 'basinhopping', ['c1'])) +
                    #             tuple((order, 'relative', 'Nelder-Mead') for order in range(2, order + 1)))

                # Now make repeated fits, increasing the polynomial order
                for item in sequence:
                    if len(item) == 3:
                        ord, weight_type, method = item
                        fixems = None
                    else:
                        ord, weight_type, method, fixems = item
                    in_weights = weights[weight_type]

                    # TODO: Can probably remove when this is optimized
                    if ord > order:
                        continue

                    # Create new initial model based on latest model
                    m_init = models.Chebyshev1D(degree=ord, domain=m_final.domain)
                    for i in range(ord + 1):
                        param = 'c{}'.format(i)
                        setattr(m_init, param, getattr(m_final, param, 0))

                    # Set some bounds; this may need to be abstracted for
                    # different instruments? TODO
                    dw = abs(2 * m_init.c1 / np.diff(m_init.domain)[0])

                    # FixMe: "0.02 * cenwave" makes tests for determineWavelengthSolution
                    #  fail. Use "0.05 * cenwave" to make them pass.
                    c0_unc = 0.02 * cenwave

                    m_init.c0.bounds = (m_init.c0 - c0_unc, m_init.c0 + c0_unc)
                    c1_unc = 0.005 * abs(m_init.c1)
                    m_init.c1.bounds = tuple(sorted([m_init.c1 - c1_unc, m_init.c1 + c1_unc]))
                    for i in range(2, ord + 1):
                        getattr(m_init, 'c{}'.format(i)).bounds = (-5, 5)

                    if fixems is not None:
                        for fx in fixems:
                            getattr(m_init, fx).fixed = True

                    fit_it = matching.KDTreeFitter(sigma=kdsigma, maxsig=10, k=3, method=method)
                    if method == 'basinhopping':
                        kwargs = {'minimizer_kwargs': {'tol': 0.01}}
                    else:
                        kwargs = {}
                    m_final = fit_it(m_init, peaks[peaks_to_fit], arc_lines,
                                     in_weights=in_weights[peaks_to_fit],
                                     ref_weights=None if weight_type == 'none' else arc_weights,
                                     **kwargs)
                    #                 method='basinhopping' if weight_type is 'none' else 'Nelder-Mead')
                    #                 options={'xatol': 1.0e-7, 'fatol': 1.0e-8})

                    log.stdinfo('{} {}'.format(repr(m_final), fit_it.statistic))
                    if debug:
                        plot_arc_fit(plot_data, peaks, arc_lines, arc_weights, m_final,
                                     "KDFit model order {} KDsigma = {}".format(ord, kdsigma))

                    match_radius = 2 * fwidth * abs(m_final.c1) / len(data)  # fwidth pixels
                    for p in m_final.param_names:
                        getattr(m_final, p).bounds = (None, None)

                    #m = matching.Chebyshev1DMatchBox.create_from_kdfit(peaks, arc_lines,
                    #                                                   model=m_final, match_radius=match_radius,
                    #                                                   sigma_clip=3)
                    # kdsigma = m.rms_output
                    # print("New kdsigma {}".format(kdsigma))
                    kdsigma = fwidth * abs(dw)
                    yplot += 1

                # Remove bounds from the model
                for p in m_final.param_names:
                    getattr(m_final, p).bounds = (None, None)

                # FixMe: using "4 * fwidth" breaks tests in test_gmos_spect_ls_wavelength_calibration.
                #   Use "2 * fwidth" to make them pass.
                match_radius = 4 * fwidth * abs(m_final.c1) / len(data)  # 2*fwidth pixels

                # match_radius = kdsigma
                m = matching.Chebyshev1DMatchBox.create_from_kdfit(peaks, arc_lines,
                                                                   model=m_final, match_radius=match_radius,
                                                                   sigma_clip=3)
                log.stdinfo('{} {} {}'.format(repr(m.forward), len(m.input_coords), m.rms_output))
                if debug:
                    plot_arc_fit(plot_data, peaks, arc_lines, arc_weights, m.forward,
                                 f"MatchBox model order {ord}")
                    #for incoord, outcoord in zip(m.forward(m.input_coords), m.output_coords):
                    #    ax.text(incoord, yplot, '{:.4f}'.format(outcoord), rotation=90,
                    #            ha='center', va='top')

                # Choice of kdsigma can have a big effect. This oscillates
                # around the initial choice, with increasing amplitude.
                # kdsigma = 10.*abs(dw) * (((1.0+0.1*((kditer+1)//2)))**((-1)**kditer)
                #                    if kditer<21 else 1)

                if debug:
                    plot_arc_fit(plot_data, peaks, arc_lines, arc_weights, m_final, "Final fit")
                    m.display_fit(show=True)
                    m.display_fit(show=False)
                    plt.savefig(ad.filename.replace('.fits', '.jpg'))

                m_final = m.forward
                rms = m.rms_output
                nmatched = len(m.input_coords)
                log.stdinfo(m_final)
                log.stdinfo("Matched {} lines with rms = {:.3f} nm.".format(nmatched, rms))

                max_rms = rms / dw  # in pixels
                max_dev = 3 * max_rms
                m_inverse = astromodels.make_inverse_chebyshev1d(m_final, rms=max_rms,
                                                                 max_deviation=max_dev)
                log.stdinfo(m_inverse)
                inv_rms = np.std(m_inverse(m_final(m.input_coords)) - m.input_coords)
                log.stdinfo("Inverse model has rms = {:.3f} pixels.".format(inv_rms))
                m_final.name = "WAVE"
                m_final.inverse = m_inverse

                m.sort()
                # Add 1 to pixel coordinates so they're 1-indexed
                incoords = np.float32(m.input_coords) + 1
                outcoords = np.float32(m.output_coords)
                model_dict = astromodels.chebyshev_to_dict(m_final)
                model_dict['rms'] = rms
                # Add information about where the extraction took place
                if ext.data.ndim > 1:
                    model_dict[direction] = 0.5 * (extract_slice.start + extract_slice.stop)

                # Ensure all columns have the same length
                pad_rows = nmatched - len(model_dict)
                if pad_rows < 0:  # Really shouldn't be the case
                    incoords = list(incoords) + [0] * (-pad_rows)
                    outcoords = list(outcoords) + [0] * (-pad_rows)
                    pad_rows = 0

                fit_table = Table([list(model_dict.keys()) + [''] * pad_rows,
                                   list(model_dict.values()) + [0] * pad_rows,
                                   incoords, outcoords],
                                  names=("name", "coefficients", "peaks", "wavelengths"))
                fit_table.meta['comments'] = ['coefficients are based on 0-indexing',
                                              'peaks column is 1-indexed']
                ext.WAVECAL = fit_table

                if ext.data.ndim == 1:
                    ext.wcs.set_transform(ext.wcs.input_frame,
                                          ext.wcs.output_frame, m_final)
                else:
                    # Write out a simplified WCS model so it's easier to
                    # extract what we need later
                    spatial_frame = cf.CoordinateFrame(naxes=1, axes_type="SPATIAL",
                                                       axes_order=(1,), unit=u.pix,
                                                       name="SPATIAL")
                    output_frame = cf.CompositeFrame([ext.wcs.output_frame.frames[0],
                                                      spatial_frame], name='world')
                    try:
                        slit_model = ext.wcs.forward_transform[f'crpix{dispaxis + 1}']
                    except IndexError:
                        slit_model = models.Identity(1)
                    slit_model.name = 'SKY'
                    if dispaxis == 1:
                        transform = m_final & slit_model
                    else:
                        transform = slit_model & m_final
                    ext.wcs = gWCS([(ext.wcs.input_frame, transform),
                                    (output_frame, None)])

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def extract1DSpectra(self, adinputs=None, **params):
        """
        Extracts one or more 1D spectra from a 2D spectral image, according to
        the contents of the `.APERTURE` table.

        If the `skyCorrectFromSlit()` primitive has not been performed, then a
        1D sky spectrum is constructed from a nearby region of the image, and
        subtracted from the source spectrum.

        Each 1D spectrum is stored as a separate extension in a new AstroData
        object. The `.WAVECAL` table (if it exists) is copied from the parent.

        These new AD objects are placed in a separate stream from the
        parent 2D images, which are returned in the default stream.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            2D spectral images with a `.APERTURE` table.
        suffix : str
            Suffix to be added to output files.
        method : {'standard', 'weighted', 'optimal'}
            Extraction method.
        width : float or None
            Width of extraction aperture in pixels. Default: None.
        grow : float
            Avoidance region around each source aperture if a sky aperture
            is required. Default: 10.
        debug: bool
            draw apertures on image display window?

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Extracted spectra as 1D data.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        method = params["method"]
        width = params["width"]
        grow = params["grow"]
        debug = params["debug"]

        colors = ("green", "blue", "red", "yellow", "cyan", "magenta")
        offset_step = 2

        ad_extracted = []
        # This is just cut-and-paste code from determineWavelengthSolution()
        for ad in adinputs:
            ad_spec = astrodata.create(ad.phu)
            ad_spec.filename = ad.filename
            ad_spec.orig_filename = ad.orig_filename
            skysub_needed = self.timestamp_keys['skyCorrectFromSlit'] not in ad.phu
            if skysub_needed:
                log.stdinfo("Sky subtraction has not been performed on {} "
                            "- extracting sky from separate apertures".
                            format(ad.filename))

            for ext in ad:
                extname = "{}:{}".format(ad.filename, ext.hdr['EXTVER'])
                if debug:
                    self.viewer.display_image(ext, wcs=False)
                if len(ext.shape) == 1:
                    log.warning(f"{extname} is already one-dimensional")
                    continue

                try:
                    aptable = ext.APERTURE
                except AttributeError:
                    log.warning("{} has no APERTURE table. Cannot extract "
                                "spectra.".format(extname))
                    continue

                num_spec = len(aptable)
                if num_spec == 0:
                    log.warning("{} has an empty APERTURE table. Cannot "
                                "extract spectra.".format(ad.filename))
                    continue

                try:
                    wave_model = astromodels.get_named_submodel(ext.wcs.forward_transform, 'WAVE')
                except (AttributeError, IndexError):
                    log.warning(f"Cannot find wavelength solution for {extname}")
                    wave_model = None

                log.stdinfo("Extracting {} spectra from {}".format(num_spec,
                                                                   extname))
                dispaxis = 2 - ext.dispersion_axis()  # python sense
                direction = "row" if dispaxis == 1 else "column"

                # We loop twice so we can construct the aperture mask if needed
                apertures = []
                for row in aptable:
                    model_dict = dict(zip(aptable.colnames, row))
                    trace_model = astromodels.dict_to_chebyshev(model_dict)
                    aperture = tracing.Aperture(trace_model,
                                                aper_lower=model_dict['aper_lower'],
                                                aper_upper=model_dict['aper_upper'])
                    if width is not None:
                        aperture.width = width
                    apertures.append(aperture)

                if skysub_needed:
                    apmask = np.logical_or.reduce([ap.aperture_mask(ext, width=width, grow=grow)
                                                   for ap in apertures])

                for apnum, aperture in enumerate(apertures, start=1):
                    log.stdinfo(f"    Extracting spectrum from aperture {apnum}")
                    self.viewer.width = 2
                    self.viewer.color = colors[(apnum-1) % len(colors)]
                    ndd_spec = aperture.extract(ext, width=width,
                                                method=method, viewer=self.viewer if debug else None)

                    # This whole (rather large) section is an attempt to ensure
                    # that sky apertures don't overlap with source apertures
                    if skysub_needed:
                        self.viewer.width = 1
                        # We're going to try to create half-size apertures
                        # equidistant from the source aperture on both sides
                        sky_width = 0.5 * aperture.width
                        sky_spectra = []

                        min_, max_ = aperture.limits()
                        for direction in (-1, 1):
                            offset = (direction * (0.5 * sky_width + grow) +
                                      (aperture.aper_upper if direction > 0 else aperture.aper_lower))
                            ok = False
                            while not ok:
                                if ((min_ + offset - 0.5 * sky_width < -0.5) or
                                        (max_ + offset + 0.5 * sky_width > ext.shape[1 - dispaxis] - 0.5)):
                                    break

                                sky_trace_model = aperture.model | models.Shift(offset)
                                sky_aperture = tracing.Aperture(sky_trace_model)
                                sky_spec = sky_aperture.extract(apmask, width=sky_width, dispaxis=dispaxis)
                                if np.sum(sky_spec.data) == 0:
                                    sky_spectra.append(sky_aperture.extract(ext, width=sky_width,
                                                                            viewer=self.viewer if debug else None))
                                    ok = True
                                offset += direction * offset_step

                        if sky_spectra:
                            # If only one, add it to itself (since it's half-width)
                            sky_spec = sky_spectra[0].add(sky_spectra[-1])
                            ad_spec.append(ndd_spec.subtract(sky_spec, handle_meta='first_found',
                                                             handle_mask=np.bitwise_or))
                        else:
                            log.warning("Difficulty finding sky aperture. No sky"
                                        " subtraction for aperture {}".format(apnum))
                            ad_spec.append(ndd_spec)
                    else:
                        ad_spec.append(ndd_spec)

                    # Create a new gWCS and add header keywords with the
                    # extraction location. All extracted spectra will have the
                    # same gWCS but that could change.
                    ext = ad_spec[-1]
                    if wave_model is not None:
                        in_frame = cf.CoordinateFrame(naxes=1, axes_type=['SPATIAL'],
                                                      axes_order=(0,), unit=u.pix,
                                                      axes_names=('x',), name='pixels')
                        out_frame = cf.SpectralFrame(unit=u.nm, name='world')
                        ext.wcs = gWCS([(in_frame, wave_model),
                                        (out_frame, None)])
                    ext.hdr[ad._keyword_for('aperture_number')] = apnum
                    center = aperture.model.c0.value
                    ext.hdr['XTRACTED'] = (center, "Spectrum extracted "
                                                    "from {} {}".format(direction, int(center + 0.5)))
                    ext.hdr['XTRACTLO'] = (aperture._last_extraction[0],
                                           'Aperture lower limit')
                    ext.hdr['XTRACTHI'] = (aperture._last_extraction[1],
                                           'Aperture upper limit')
                    # TODO: remove after testing
                    try:
                        ext.WAVECAL = ext.WAVECAL
                    except AttributeError:
                        pass

            # Don't output a file with no extracted spectra
            if len(ad_spec) > 0:
                try:
                    del ad_spec.hdr['RADECSYS']
                except KeyError:
                    pass
                gt.mark_history(ad_spec, primname=self.myself(), keyword=timestamp_key)
                ad_spec.update_filename(suffix=sfx, strip=True)
                ad_extracted.append(ad_spec)

        # Only return extracted spectra
        return ad_extracted

    def findSourceApertures(self, adinputs=None, **params):
        """
        Finds sources in 2D spectral images and store them in an APERTURE table
        for each extension. Each table will, then, be used in later primitives
        to perform aperture extraction.

        The primitive operates by first collapsing the 2D spectral image in
        the spatial direction to identify sky lines as regions of high
        pixel-to-pixel variance, and the regions between the sky lines which
        consist of at least `min_sky_pix` pixels are selected. These are then
        collapsed in the dispersion direction to produce a 1D spatial profile,
        from which sources are identified using a peak-finding algorithm.

        The widths of the apertures are determined by calculating a threshold
        level relative to the peak, or an integrated flux relative to the
        total between the minima on either side and determining where a smoothed
        version of the source profile reaches this threshold.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Science data as 2D spectral images.
        suffix : str
            Suffix to be added to output files.
        max_apertures : int, default: None
            Maximum number of apertures expected to be found.
        threshold : float (0 - 1)
            height above background (relative to peak) at which to define
            the edges of the aperture. Default: 0.01.
        min_sky_pix : int
            minimum number of contiguous pixels between sky lines
            for a region to be added to the spectrum before collapsing to 1D.
            Default: 20.

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            The 2D spectral images with APERTURE tables attached

        See Also
        --------
        :meth:`~geminidr.core.primitives_spect.Spect.determineDistortion`,
        :meth:`~geminidr.cofe.primitives_spect.Spect.distortionCorrect`
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        max_apertures = params["max_apertures"]
        threshold = params["threshold"]
        min_sky_pix = params["min_sky_region"]

        limit_method = 'threshold'

        for ad in adinputs:
            if self.timestamp_keys['distortionCorrect'] not in ad.phu:
                log.warning("{} has not been distortion corrected".
                            format(ad.filename))
            for ext in ad:
                log.stdinfo("Searching for sources in {}:{}".
                            format(ad.filename, ext.hdr['EXTVER']))

                dispaxis = 2 - ext.dispersion_axis()  # python sense
                npix = ext.shape[dispaxis]

                # data, mask, variance are all arrays in the GMOS orientation
                # with spectra dispersed horizontally
                data, mask, variance = _transpose_if_needed(ext.data, ext.mask,
                                                            ext.variance, transpose=dispaxis == 0)
                direction = "column" if dispaxis == 0 else "row"

                # Collapse image along spatial direction to find noisy regions
                # (caused by sky lines, regardless of whether image has been
                # sky-subtracted or not)
                data1d, mask1d, var1d = NDStacker.mean(data, mask=mask,
                                                       variance=variance)
                ndd = NDAstroData(var1d, mask=mask1d)
                mean, sigma, _ = gt.measure_bg_from_image(ndd, sampling=1)

                # Mask sky-line regions and find clumps of unmasked pixels
                mask1d[var1d > mean + sigma] = 1
                slices = np.ma.clump_unmasked(np.ma.masked_array(var1d, mask1d))
                sky_regions = [slice_ for slice_ in slices
                               if slice_.stop - slice_.start >= min_sky_pix]

                sky_mask = np.ones_like(mask, dtype=np.uint16)
                for reg in sky_regions:
                    sky_mask[(slice(None), reg)] = 0
                sky_mask |= (mask > 0)

                # We probably don't want the median because of, e.g., a
                # Lyman Break Galaxy that may have signal for less than half
                # the dispersion direction.
                profile, prof_mask, prof_var = NDStacker.combine(data.T, mask=sky_mask.T,
                                                                 variance=None if variance is None else variance.T,
                                                                 rejector="none", combiner="mean")
                # profile, prof_mask, prof_var = NDStacker.combine((data / data1d).T, mask=sky_mask.T,
                #                                                 variance=None if variance is None else (variance / (data1d*data1d)).T,
                #                                                 rejector="sigclip", combiner="wtmean")

                # TODO: find_peaks might not be best considering we have no
                #   idea whether sources will be extended or not
                widths = np.arange(5, 30)
                peaks_and_snrs = tracing.find_peaks(profile, widths, mask=prof_mask,
                                                    variance=prof_var, reject_bad=False,
                                                    min_snr=3, min_frac=0.2)

                if peaks_and_snrs.size == 0:
                    log.warning("Found no sources")
                    try:
                        del ext.APERTURE
                    except AttributeError:
                        pass
                    continue

                # Reverse-sort by SNR and return only the locations
                locations = np.array(sorted(peaks_and_snrs.T, key=lambda x: x[1],
                                            reverse=True)[:max_apertures]).T[0]
                locstr = ' '.join(['{:.1f}'.format(loc) for loc in locations])
                log.stdinfo("Found sources at {}s: {}".format(direction, locstr))

                all_limits = tracing.get_limits(profile, prof_mask, peaks=locations,
                                                threshold=threshold, method=limit_method)

                all_model_dicts = []
                for loc, limits in zip(locations, all_limits):
                    cheb = models.Chebyshev1D(degree=0, domain=[0, npix - 1], c0=loc)
                    model_dict = astromodels.chebyshev_to_dict(cheb)
                    lower, upper = limits - loc
                    model_dict['aper_lower'] = lower
                    model_dict['aper_upper'] = upper
                    all_model_dicts.append(model_dict)
                    log.debug("Limits for source {:.1f} ({:.1f}, +{:.1f})".format(loc, lower, upper))

                aptable = Table([np.arange(len(locations)) + 1], names=['number'])
                for name in model_dict.keys():  # Still defined from above loop
                    aptable[name] = [model_dict.get(name, 0)
                                     for model_dict in all_model_dicts]
                ext.APERTURE = aptable

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)
        return adinputs

    def fluxCalibrate(self, adinputs=None, **params):
        """
        Performs flux calibration multiplying the input signal by the
        sensitivity function obtained from
        :meth:`~geminidr.core.primitives_spect.Spec.calculateSensitivity`.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            1D or 2D Spectra of targets that need to be flux-calibrated.
            2D spectra are expected to be distortion corrected and its
            dispersion axis should be along rows.

        suffix :  str, optional
            Suffix to be added to output files (default: _fluxCalibrated).

        standard: str or AstroData, optional
            Standard star spectrum containing one extension or the same number
            of extensions as the input spectra. Each extension must have a
            `.SENSFUNC` table containing information about the overall
            sensitivity. Right now, if this is not provided, it will raise a
            NotImplementedError since it needs implementation.

        units : str, optional
            Units for output spectrum (default: W m-2 nm-1).

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            The same input list is used as output but each object now has
            its pixel values in physical units.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        std = params["standard"]
        final_units = params["units"]

        # Expectation is that the SENSFUNC table will be in units
        # like (electron/s) / (W/m^2)
        flux_units = u.Unit("W m-2")

        # Get a suitable arc frame (with distortion map) for every science AD
        if std is None:
            self.getProcessedStandard(adinputs, refresh=False)
            std_list = self._get_cal(adinputs, 'processed_standard')
        else:
            std_list = std

        for ad, std in zip(*gt.make_lists(adinputs, std_list, force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by fluxCalibrate".
                            format(ad.filename))
                continue

            if std is None:
                if 'sq' in self.mode:
                    raise OSError('No processed standard listed for {}'.
                                  format(ad.filename))
                else:
                    log.warning("No changes will be made to {}, since no "
                                "standard was specified".format(ad.filename))

            len_std, len_ad = len(std), len(ad)
            if len_std not in (1, len_ad):
                log.warning("{} has {} extensions so cannot be used to "
                            "calibrate {} with {} extensions.".
                            format(std.filename, len_std, ad.filename, len_ad))
                continue

            # Since 2D flux calibration just uses the wavelength info for the
            # middle row/column, non-distortion-corrected data will have the
            # wrong wavelength solution in other columns/rows
            if (any(len(ext.shape) == 2 for ext in ad) and
                    not self.timestamp_keys['distortionCorrect'] in ad.phu):
                log.warning("{} has not been distortion corrected".format(ad.filename))

            exptime = ad.exposure_time()
            try:
                delta_airmass = ad.airmass() - std.airmass()
            except TypeError:  # if either airmass() returns None
                log.warning("Cannot determine airmass of target and/or standard."
                            " Not making an airmass correction.")
                delta_airmass = 0

            for index, ext in enumerate(ad):
                ext_std = std[max(index, len_std-1)]
                try:
                    sensfunc = ext_std.SENSFUNC
                except AttributeError:
                    log.warning("{}:{} has no SENSFUNC table. Cannot flux calibrate".
                                format(std.filename, ext_std.hdr['EXTVER']))
                    continue

                extver = '{}:{}'.format(ad.filename, ext.hdr['EXTVER'])

                # Try to confirm the science image has the correct units
                std_flux_unit = sensfunc['coefficients'].unit
                if isinstance(std_flux_unit, u.LogUnit):
                    std_flux_unit = std_flux_unit.physical_unit
                try:
                    sci_flux_unit = u.Unit(ext.hdr.get('BUNIT'))
                except:
                    sci_flux_unit = None
                if not (std_flux_unit is None or sci_flux_unit is None):
                    unit = sci_flux_unit / (std_flux_unit * flux_units)
                    if unit.is_equivalent(u.s):
                        log.fullinfo("Dividing {} by exposure time of {} s".
                                     format(extver, exptime))
                        ext /= exptime
                        sci_flux_unit /= u.s
                    elif not unit.is_equivalent(u.dimensionless_unscaled):
                        log.warning("{} has incompatible units ('{}' and '{}')."
                                    "Cannot flux calibrate".format(extver,
                                                                   sci_flux_unit, std_flux_unit))
                        continue
                else:
                    log.warning("Cannot determine units of data and/or SENSFUNC "
                                "table for {}, so cannot flux calibrate.".format(extver))
                    continue

                # Get wavelengths of all pixels
                ndim = len(ext.shape)
                dispaxis = 0 if ndim == 1 else 2 - ext.dispersion_axis()

                # Get wavelengths and pixel sizes of all the pixels along the
                # dispersion axis by calculating wavelengths in the middles and
                # edges of all pixels.
                all_coords = [0.5*(length - 1) for length in ext.shape]
                all_coords[dispaxis] = np.arange(-0.5, ext.shape[dispaxis], 0.5)
                all_waves = ext.wcs(*all_coords[::-1], with_units=True)
                if ndim > 1:
                    all_waves = all_waves[0]

                waves = all_waves[1::2]
                pixel_sizes = abs(np.diff(all_waves[::2]))

                # Reconstruct the spline and evaluate it at every wavelength
                spline = BSpline(sensfunc['knots'].data, sensfunc['coefficients'].data, 3)
                sens_factor = spline(waves.to(sensfunc['knots'].unit)) * sensfunc['coefficients'].unit
                try:  # conversion from magnitude/logarithmic units
                    sens_factor = sens_factor.physical
                except AttributeError:
                    pass

                # Apply airmass correction. If none is needed/possible, we
                # don't need to try to do this
                if delta_airmass != 0:
                    telescope = ad.telescope()
                    try:
                        extinction_correction = extinct.extinction(waves, telescope=telescope)
                    except KeyError:
                        log.warning("Telescope {} not recognized. "
                                    "Not making an airmass correction.".format(telescope))
                    else:
                        log.stdinfo("Correcting for difference of {:5.3f} "
                                    "airmasses".format(delta_airmass))
                        sens_factor *= 10**(0.4*delta_airmass * extinction_correction)

                final_sens_factor = (sci_flux_unit / (sens_factor * pixel_sizes)).to(final_units,
                                     equivalencies=u.spectral_density(waves)).value

                if ndim == 2 and dispaxis == 0:
                    ext *= final_sens_factor[:, np.newaxis]
                else:
                    ext *= final_sens_factor
                ext.hdr['BUNIT'] = final_units

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def linearizeSpectra(self, adinputs=None, **params):
        """
        Transforms 1D spectra so that the relationship between them and their
        respective wavelength calibration is linear.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Wavelength calibrated 1D spectra. Each extension must have a
            `.WAVECAL` table.

        suffix : str
            Suffix to be added to output files.
        w1 : float
            Wavelength of first pixel (nm). See Notes below.
        w2 : float
            Wavelength of last pixel (nm). See Notes below.
        dw : float
            Dispersion (nm/pixel). See Notes below.
        npix : int
            Number of pixels in output spectrum. See Notes below.
        conserve : bool
            Conserve flux (rather than interpolate)?
        order : int
            order of interpolation during the resampling

        Notes
        -----
        Exactly 0 or 3 of (w1, w2, dw, npix) must be specified.

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Linearized 1D spectra.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        w1 = params["w1"]
        w2 = params["w2"]
        dw = params["dw"]
        npix = params["npix"]
        conserve = params["conserve"]
        order = params["order"]

        # There are either 1 or 4 Nones, due to validation
        nones = [w1, w2, dw, npix].count(None)
        if nones == 1:
            # Work out the missing variable from the others
            if npix is None:
                npix = int(np.ceil((w2 - w1) / dw)) + 1
                w2 = w1 + (npix - 1) * dw
            elif w1 is None:
                w1 = w2 - (npix - 1) * dw
            elif w2 is None:
                w2 = w1 + (npix - 1) * dw
            else:
                dw = (w2 - w1) / (npix - 1)

        # We send the ADs through one-by-one so there's no attempt to
        # align them in the spatial direction
        adoutputs = []
        for ad in adinputs:
            ad_out = self.resampleToCommonFrame([ad], suffix=sfx, w1=w1, w2=w2, npix=npix,
                                                conserve=conserve, order=order,
                                                trim_data=False)[0]
            gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
            adoutputs.append(ad_out)

        return adoutputs

    def makeProcessedSlitIllum(self, adinputs=None, **params):
        """
        Creates the processed Slit Response Illumination Function by binning a
        2D spectrum along the dispersion direction, fitting a smooth function
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
        # timestamp_key = self.timestamp_keys[self.myself()] - Todo

        suffix = params["suffix"]
        bins = params["bins"]

        ad_outputs = []
        for ad in adinputs:

            # Mosaic data if has more than one extension temporarily
            if len(ad) > 1:
                log.info("Temporarily mosaicking multi-extension file")
                geotable = import_module('.geometry_conf', self.inst_lookups)
                _ad = deepcopy(ad)  # Prevent modifying input file inplace
                transform.add_mosaic_wcs(_ad, geotable)
                slit_response_ad = transform.resample_from_wcs(
                    _ad, "mosaic", attributes=None, order=1, process_objcat=False)

            else:
                log.info("Using single-extension file")
                slit_response_ad = ad

            log.info("Transposing data if needed")
            dispaxis = 2 - slit_response_ad[0].dispersion_axis()  # python sense

            data, mask, variance = _transpose_if_needed(
                slit_response_ad[0].data, slit_response_ad[0].mask,
                slit_response_ad[0].variance, transpose=dispaxis == 1)

            log.info("Using masked data")
            data = np.ma.masked_array(data, mask=mask)
            height = data.shape[0]
            width = data.shape[1]

            if bins is None:
                nbins = len(ad)
                bin_limits = np.linspace(0, data.shape[0], nbins + 1, dtype=int)
            elif isinstance(bins, int):
                nbins = bins
                bin_limits = np.linspace(0, data.shape[0], nbins + 1, dtype=int)
            else:
                # ToDo - Handle input bins as array
                raise TypeError("Expected None or Int for `bins`. "
                                "Found: {}".format(type(bins)))

            bin_top = bin_limits[1:]
            bin_bot = bin_limits[:-1]
            binned_data = np.zeros_like(data)

            log.info("Smooth binned data and normalize them by smoothed central value")
            for i, (b0, b1) in enumerate(zip(bin_bot, bin_top)):

                x = np.arange(width)
                y = np.ma.mean(data[b0:b1], axis=0)
                s = astromodels.UnivariateSplineWithOutlierRemoval(x, y, order=5)

                slit_central_value = s(x)[width // 2]
                binned_data[b0:b1] = s(x) / slit_central_value

            log.info("Reconstruct 2D mosaicked image")
            bin_center = np.array(0.5 * (bin_bot + bin_top), dtype=int)
            cols_fit, rows_fit = np.meshgrid(np.arange(width), bin_center)

            fit_p = fitting.SLSQPLSQFitter()
            model_init = models.Chebyshev2D(
                x_degree=6, x_domain=(0, width),
                y_degree=nbins - 1, y_domain=(0, height))

            slit_response_model = fit_p(model_init, cols_fit, rows_fit,
                                        binned_data[rows_fit, cols_fit])

            rows_val, cols_val = np.mgrid[:height, :width]
            slit_response_data = slit_response_model(cols_val, rows_val)

            del cols_fit, cols_val, rows_fit, rows_val

            _data, _mask, _variance = _transpose_if_needed(
                slit_response_data, mask, variance, transpose=dispaxis == 1)

            slit_response_ad[0].data = _data
            slit_response_ad[0].mask = _mask
            slit_response_ad[0].variance = _variance

            if len(ad) > 1:
                log.info("Map coordinates between multi-extension and mosaicked data")
                slit_response_ad = _split_mosaic_into_extensions(_ad, slit_response_ad)

            ad_outputs.append(slit_response_ad)

        return ad_outputs

    def normalizeFlat(self, adinputs=None, **params):
        """
        This primitive normalizes a spectroscopic flatfield, by fitting
        a cubic spline along the dispersion direction of an averaged
        combination of rows/columns (by default, in the center of the
        spatial direction). Each row/column is then divided by this spline.

        For multi-extension AstroData objects of MOS or XD, each extension
        is treated separately. For other multi-extension data,
        mosaicDetectors() is called to produce a single extension, and the
        spline fitting is performed with variable scaling parameters for
        each detector (identified within the mosaic from groups of DQ.no_data
        pixels). The spline fit is calculated in the mosaicked frame but it
        is evaluated for each pixel in each unmosaicked detector, so that
        the resultant flatfield always has the same format (i.e., number of
        extensions and their shape) as the input frame.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        center: int/None
            central row/column for 1D extraction (None => use middle)
        nsum: int
            number of rows/columns around center to combine
        spectral_order: int
            order of fit in spectral direction
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        spline_kwargs = params.copy()
        sfx = params.pop("suffix")
        spectral_order = params.pop("spectral_order")
        center = params.pop("center")
        nsum = params.pop("nsum")

        for ad in adinputs:
            # Don't mosaic if the multiple extensions are because the
            # data are MOS or cross-dispersed
            if len(ad) > 1 and not ({'MOS', 'XD'} & ad.tags):
                geotable = import_module('.geometry_conf', self.inst_lookups)
                adg = transform.create_mosaic_transform(ad, geotable)
                admos = adg.transform(attributes=None, order=1)
                mosaicked = True
            else:
                admos = ad
                mosaicked = False

            # This will loop over MOS slits or XD orders
            for ext in admos:
                dispaxis = 2 - ext.dispersion_axis()  # python sense
                direction = "row" if dispaxis == 1 else "column"

                data, mask, variance, extract_slice = _average_along_slit(ext, center=center, nsum=nsum)
                log.stdinfo("Extracting 1D spectrum from {}s {} to {}".
                            format(direction, extract_slice.start + 1, extract_slice.stop))
                mask |= (DQ.no_data * (variance == 0))  # Ignore var=0 points
                slices = _ezclump((mask & (DQ.no_data | DQ.unilluminated)) == 0)

                masked_data = np.ma.masked_array(data, mask=mask)
                weights = np.sqrt(np.where(variance > 0, 1. / variance, 0.))
                pixels = np.arange(len(masked_data))

                # We're only going to do CCD-to-CCD normalization if we've
                # done the mosaicking in this primitive; if not, we assume
                # the user has already taken care of it (if it's required).
                nslices = len(slices)
                if nslices > 1 and mosaicked:
                    coeffs = np.ones((nslices - 1,))
                    boundaries = list(slice_.stop for slice_ in slices[:-1])
                    result = optimize.minimize(QESpline, coeffs, args=(pixels, masked_data,
                                                                       weights, boundaries, spectral_order), tol=1e-7,
                                               method='Nelder-Mead')
                    if not result.success:
                        log.warning("Problem with spline fitting: {}".format(result.message))

                    # Rescale coefficients so centre-left CCD is unscaled
                    coeffs = np.insert(result.x, 0, [1])
                    coeffs /= coeffs[len(coeffs) // 2]
                    for coeff, slice_ in zip(coeffs, slices):
                        masked_data[slice_] *= coeff
                        weights[slice_] /= coeff
                    log.stdinfo("QE scaling factors: " +
                                " ".join("{:6.4f}".format(coeff) for coeff in coeffs))
                spline = astromodels.UnivariateSplineWithOutlierRemoval(pixels, masked_data,
                                                                        order=spectral_order, w=weights,
                                                                        **spline_kwargs)

                if not mosaicked:
                    flat_data = np.tile(spline.data, (ext.shape[dispaxis - 1], 1))
                    ext.divide(_transpose_if_needed(flat_data, transpose=(dispaxis == 2))[0])

            # If we've mosaicked, there's only one extension
            # We forward transform the input pixels, take the transformed
            # coordinate along the dispersion direction, and evaluate the
            # spline there.
            if mosaicked:
                for block, trans in adg:
                    trans.append(models.Shift(-adg.origin[1]) & models.Shift(-adg.origin[0]))
                    for ext, corner in zip(block, block.corners):
                        t = deepcopy(trans)
                        # Shift so coordinates are correct in this Block
                        t.prepend(models.Shift(corner[1]) & models.Shift(corner[0]))
                        geomap = transform.GeoMap(t, ext.shape, inverse=True)
                        flat_data = spline(geomap.coords[dispaxis])
                        ext.divide(flat_data)

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def resampleToCommonFrame(self, adinputs=None, **params):
        """
        Resample 1D or 2D spectra on a common frame, and optionally transform
        them so that the relationship between them and their respective
        wavelength calibration is linear.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Wavelength calibrated 1D or 2D spectra. Each extension must have a
            `.WAVECAL` table.
        suffix : str
            Suffix to be added to output files.
        w1 : float
            Wavelength of first pixel (nm). See Notes below.
        w2 : float
            Wavelength of last pixel (nm). See Notes below.
        dw : float
            Dispersion (nm/pixel). See Notes below.
        npix : int
            Number of pixels in output spectrum. See Notes below.
        conserve : bool
            Conserve flux (rather than interpolate)?
        order : int
            order of interpolation during the resampling
        trim_data : bool
            Trim spectra to size of reference spectra?
        force_linear : bool
            Force a linear output wavelength solution?

        Notes
        -----
        If ``w1`` or ``w2`` are not specified, they are computed from the
        individual spectra: if ``trim_data`` is True, this is the intersection
        of the spectra ranges, otherwise this is the union of all ranges,

        If ``dw`` or ``npix`` are specified, the spectra are linearized.

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Linearized 1D spectra.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        suffix = params["suffix"]
        w1 = params["w1"]
        w2 = params["w2"]
        dw = params["dw"]
        npix = params["npix"]
        conserve = params["conserve"]
        trim_data = params["trim_data"]
        force_linear = params["force_linear"]

        # Check that all ad objects are either 1D or 2D
        ndim = {len(ext.shape) for ad in adinputs for ext in ad}
        if len(ndim) != 1:
            raise ValueError('inputs must have the same dimension')
        ndim = ndim.pop()

        # For the 2D case check that all ad objects have only 1 extension
        if ndim > 1:
            adjust_key = self.timestamp_keys['adjustWCSToReference']
            for i, ad in enumerate(adinputs):
                if len(ad) != 1:
                    raise ValueError('inputs must have only 1 extension')
                if adjust_key not in ad.phu:
                    log.warning(
                        "{} has not offset, adjustWCSToReference "
                        "should be run first".format(ad.filename))
            # Store these values for later!
            refad = adinputs[0]
            ref_coords = (refad.central_wavelength(asNanometers=True),
                          refad.target_ra(), refad.target_dec())
            ref_pixels = refad[0].wcs.backward_transform(*ref_coords)

        # If only one variable is missing we compute it from the others
        nparams = sum(x is not None for x in (w1, w2, dw, npix))
        if nparams == 3:
            if npix is None:
                npix = int(np.ceil((w2 - w1) / dw)) + 1
                w2 = w1 + (npix - 1) * dw
            elif w1 is None:
                w1 = w2 - (npix - 1) * dw
            elif w2 is None:
                w2 = w1 + (npix - 1) * dw
            else:
                dw = (w2 - w1) / (npix - 1)

        # Gather information from all the spectra (Chebyshev1D model,
        # w1, w2, dw, npix), and compute the final bounds (w1out, w2out)
        # if there are not provided
        info = []
        w1out, w2out, dwout, npixout = w1, w2, dw, npix
        for ad in adinputs:
            adinfo = []
            for ext in ad:
                extname = "{}:{}".format(ad.filename, ext.hdr['EXTVER'])
                try:
                    model_info = _extract_model_info(ext)
                except ValueError:
                    raise ValueError("{} has no WAVECAL. Cannot linearize."
                                     .format(extname))
                adinfo.append(model_info)

                if w1 is None:
                    if w1out is None:
                        w1out = model_info['w1']
                    elif trim_data:
                        w1out = max(w1out, model_info['w1'])
                    else:
                        w1out = min(w1out, model_info['w1'])

                if w2 is None:
                    if w2out is None:
                        w2out = model_info['w2']
                    elif trim_data:
                        w2out = min(w2out, model_info['w2'])
                    else:
                        w2out = max(w2out, model_info['w2'])
            info.append(adinfo)

        if trim_data:
            if w1 is None:
                w1out = info[0][0]['w1']
            if w2 is None:
                w2out = info[0][0]['w2']
            if w1 is None or w2 is None:
                log.fullinfo("Trimming data to size of reference spectra")

        # linearize spectra only if the grid parameters are specified
        linearize = force_linear or npix is not None or dw is not None
        if linearize:
            if npixout is None and dwout is None:
                # if both are missing, use the reference spectrum
                dwout = info[0][0]['dw']

            if npixout is None:
                npixout = int(np.ceil((w2out - w1out) / dwout)) + 1
            elif dwout is None:
                dwout = (w2out - w1out) / (npixout - 1)

        if linearize:
            new_wave_model = models.Scale(dwout) | models.Shift(w1out)
        else:
            # compute the inverse model needed to go to the reference
            # spectrum grid. Due to imperfections in the Chebyshev inverse
            # we check whether the wavelength limits are the same as the
            # reference spectrum.
            wave_model_ref = info[0][0]['wave_model'].copy()
            wave_model_ref.name = None
            limits = wave_model_ref.inverse([w1out, w2out])
            if info[0][0]['w1'] == w1out:
                limits[0] = round(limits[0])
            if info[0][0]['w2'] == w2out:
                limits[1] = round(limits[1])
            pixel_shift = int(np.ceil(limits.min()))
            new_wave_model = models.Shift(pixel_shift) | wave_model_ref
            if info[0][0]['w2'] == w2out:
                npixout = info[0][0]['npix']
            else:
               npixout = int(np.floor(new_wave_model.inverse([w1out, w2out]).max()) + 1)
            dwout = (w2out - w1out) / (npixout - 1)

        new_wave_model.name = 'WAVE'
        if ndim == 1:
            new_wcs_model = new_wave_model
        else:
            new_wcs_model = refad[0].wcs.forward_transform.replace_submodel('WAVE', new_wave_model)

        adoutputs = []
        for i, ad in enumerate(adinputs):
            for iext, ext in enumerate(ad):
                wave_model = info[i][iext]['wave_model']
                extn = "{}:{}".format(ad.filename, ext.hdr['EXTVER'])
                wave_resample = wave_model | new_wave_model.inverse
                # TODO: This shouldn't really be needed, but it is
                wave_resample.inverse = new_wave_model | wave_model.inverse

                # Avoid performing a Cheb and its imperfect inverse
                if not linearize and new_wave_model[1:] == wave_model:
                    wave_resample = models.Shift(-pixel_shift)

                if ndim == 1:
                    dispaxis = 0
                    resampling_model = wave_resample
                else:
                    pixels = ext.wcs.backward_transform(*ref_coords)
                    dispaxis = 2 - ext.dispersion_axis()  # python sense
                    slit_offset = models.Shift(ref_pixels[dispaxis] - pixels[dispaxis])
                    if dispaxis == 0:
                        resampling_model = slit_offset & wave_resample
                    else:
                        resampling_model = wave_resample & slit_offset

                if i == 0 and not linearize:
                    log.fullinfo(f"{ad.filename}: No interpolation")
                msg = "Resampling"
                if linearize:
                    msg += " and linearizing"
                log.stdinfo("{} {}: w1={:.3f} w2={:.3f} dw={:.3f} npix={}"
                            .format(msg, extn, w1out, w2out, dwout, npixout))

                # If we resample to a coarser pixel scale, we may
                # interpolate over features. We avoid this by subsampling
                # back to the original pixel scale (approximately).
                input_dw = info[i][iext]['dw']
                subsample = int(np.ceil(abs(dwout / input_dw) - 0.1))
                attributes = [attr for attr in ('data', 'mask', 'variance')
                              if getattr(ext, attr) is not None]

                ext.wcs = gWCS([(ext.wcs.input_frame, resampling_model),
                                (cf.Frame2D(name='resampled'), new_wcs_model),
                                (ext.wcs.output_frame, None)])

                origin = (0,) * ndim
                output_shape = list(ext.shape)
                output_shape[dispaxis] = npixout
                new_ext = transform.resample_from_wcs(ext, 'resampled', subsample=subsample,
                                                      attributes=attributes, conserve=conserve,
                                                      origin=origin, output_shape=output_shape)
                if iext == 0:
                    ad_out = new_ext
                else:
                    ad_out.append(new_ext[0])
                if ndim == 2:
                    try:
                        offset = slit_offset.offset.value
                        ext.APERTURE['c0'] += offset
                        log.fullinfo("Shifting aperture locations by {:.2f} "
                                     "pixels".format(offset))
                    except AttributeError:
                        pass

            # Timestamp and update the filename
            gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
            ad_out.update_filename(suffix=suffix, strip=True)
            adoutputs.append(ad_out)

        return adoutputs

    def skyCorrectFromSlit(self, adinputs=None, **params):
        """
        Performs row-by-row/column-by-column sky subtraction of 2D spectra.

        For that, it fits the sky contribution using a Univariate Spline and
        builds a mask of rejected pixels during the fitting process. It also
        adds any apertures defined in the APERTURE table to this mask if it
        exists.

        If there are less than 4 good pixels on each row/column, then the fit
        is performed to every pixel.

        This primitive should be called on data free of distortion.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            2D science spectra loaded as :class:`~astrodata.AstroData` objects.
        suffix : str or None
            Suffix to be added to output files.
        order : int or None
            Order of piecewise cubic spline fit to each row/column. If `None`,
            it uses as many pieces as required to get chi^2=1. Else, it is
            reduced proportionately by the ratio between the number of good pixels
            in each row/column and the total number of pixels. Default: 5.
        grow : float
            Masking growth radius (in pixels) for each aperture. Default: 0.

        Returns
        -------
        adinputs : list of :class:`~astrodata.AstroData`
            Sky subtractd 2D spectral images.

        See Also
        --------
        :meth:`~geminidr.core.primitives_spect.Spect.determineDistortion`,
        :meth:`~geminidr.core.primitives_spect.Spect.distortionCorrect`,
        :meth:`~geminidr.core.primitives_spect.Spect.findSourceApertures`,
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        order = params["order"]
        grow = params["grow"]

        for ad in adinputs:
            if self.timestamp_keys['distortionCorrect'] not in ad.phu:
                log.warning("{} has not been distortion corrected. Sky "
                            "subtraction is likely to be poor.".format(ad.filename))

            start = datetime.now()
            for ext in ad:
                dispaxis = 2 - ext.dispersion_axis()  # python sense
                slitlen = ext.shape[1 - dispaxis]
                pixels = np.arange(slitlen)

                # We want to mask pixels in apertures in addition to the mask
                sky_mask = (np.zeros_like(ext.data, dtype=DQ.datatype)
                            if ext.mask is None else ext.mask.copy())

                # If there's an aperture table, go through it row by row,
                # masking the pixels
                try:
                    aptable = ext.APERTURE
                except AttributeError:
                    pass
                else:
                    for row in aptable:
                        model_dict = dict(zip(aptable.colnames, row))
                        trace_model = astromodels.dict_to_chebyshev(model_dict)
                        aperture = tracing.Aperture(trace_model, aper_lower=model_dict['aper_lower'],
                                                    aper_upper=model_dict['aper_upper'])
                        sky_mask |= aperture.aperture_mask(ext, grow=grow)

                # Transpose if needed so we're iterating along rows
                data, mask, var = _transpose_if_needed(ext.data, sky_mask, ext.variance, transpose=dispaxis == 1)
                sky_model = np.empty_like(data)
                sky_weights = (np.ones_like(data) if var is None
                               else np.sqrt(np.divide(1., var, out=np.zeros_like(data),
                                                      where=var > 0)))

                # Now fit the model for each row/column along dispersion axis
                for i, (data_row, mask_row, weight_row) in enumerate(zip(data, mask,
                                                                         sky_weights)):
                    sky = np.ma.masked_array(data_row, mask=mask_row)
                    if weight_row.sum() == 0:
                        weight_row = None

                    spline = astromodels.UnivariateSplineWithOutlierRemoval(pixels, sky, order=order,
                                                                            w=weight_row, grow=2)
                    # Spline fit has been returned so no need to recompute
                    sky_model[i] = spline.data

                ext.data -= (sky_model if dispaxis == 0 else sky_model.T)

            print(datetime.now() - start, ad.filename)
            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

    def traceApertures(self, adinputs=None, **params):
        """
        Traces apertures listed in the `.APERTURE` table along the dispersion
        direction, and estimates the optimal extraction aperture size from the
        spatial profile of each source.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Science data as 2D spectral images with a `.APERTURE` table attached
            to one or more of its extensions.
        suffix : str
            Suffix to be added to output files.
        trace_order : int
            Fitting order along spectrum. Default: 2
        step : int
            Step size for sampling along dispersion direction. Default: 10
        nsum : int
            Number of rows/columns to combine at each step. Default: 10
        max_missed : int
            Maximum number of interactions without finding line before line is
            considered lost forever. Default: 5
        max_shift : float
            Maximum perpendicular shift (in pixels) from pixel to pixel.
            Default: 0.05
        debug: bool
            draw aperture traces on image display window?

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Science data as 2D spectral images with the `.APERTURE` the updated
            to contain its upper and lower limits.

        See Also
        --------
        :meth:`~geminidr.core.primitives_spect.Spect.findSourceApertures`

        """

        def averaging_func(data, mask=None, variance=None):
            """Use a sigma-clipped mean to collapse in the dispersion
            direction, which should reject sky lines"""
            return NDStacker.mean(*NDStacker.sigclip(data, mask=mask,
                                                     variance=variance))

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        order = params["trace_order"]
        step = params["step"]
        nsum = params["nsum"]
        max_missed = params["max_missed"]
        max_shift = params["max_shift"]
        debug = params["debug"]

        for ad in adinputs:
            for ext in ad:
                try:
                    aptable = ext.APERTURE
                    locations = aptable['c0'].data
                except (AttributeError, KeyError):
                    log.warning("Could not find aperture locations in {}:{} - "
                                "continuing".format(ad.filename, ext.hdr['EXTVER']))
                    continue

                if debug:
                    self.viewer.display_image(ext, wcs=False)
                    self.viewer.width = 2
                dispaxis = 2 - ext.dispersion_axis()  # python sense

                # TODO: Do we need to keep track of where the initial
                # centering took place?
                start = 0.5 * ext.shape[dispaxis]

                # The coordinates are always returned as (x-coords, y-coords)
                all_ref_coords, all_in_coords = tracing.trace_lines(ext, axis=dispaxis,
                                                                    start=start, initial=locations, width=5, step=step,
                                                                    nsum=nsum, max_missed=max_missed,
                                                                    max_shift=max_shift,
                                                                    viewer=self.viewer if debug else None)

                self.viewer.color = "blue"
                spectral_coords = np.arange(0, ext.shape[dispaxis], step)
                all_column_names = []
                all_model_dicts = []
                for aperture in aptable:
                    location = aperture['c0']
                    # Funky stuff to extract the traced coords associated with
                    # each aperture (there's just a big list of all the coords
                    # from all the apertures) and sort them by coordinate
                    # along the spectrum
                    coords = np.array([list(c1) + list(c2)
                                       for c1, c2 in zip(all_ref_coords.T, all_in_coords.T)
                                       if c1[dispaxis] == location])
                    values = np.array(sorted(coords, key=lambda c: c[1 - dispaxis])).T
                    ref_coords, in_coords = values[:2], values[2:]

                    # Find model to transform actual (x,y) locations to the
                    # value of the reference pixel along the dispersion axis
                    m_init = models.Chebyshev1D(degree=order, c0=location,
                                                domain=[0, ext.shape[dispaxis] - 1])
                    fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                                               sigma_clip, sigma=3)
                    try:
                        m_final, _ = fit_it(m_init, in_coords[1 - dispaxis], in_coords[dispaxis])
                    except (IndexError, np.linalg.linalg.LinAlgError):
                        # This hides a multitude of sins, including no points
                        # returned by the trace, or insufficient points to
                        # constrain the request order of polynomial.
                        log.warning("Unable to trace aperture {}".format(aperture["number"]))
                        m_final = models.Chebyshev1D(degree=0, c0=location,
                                                     domain=[0, ext.shape[dispaxis] - 1])
                    else:
                        if debug:
                            plot_coords = np.array([spectral_coords, m_final(spectral_coords)]).T
                            self.viewer.polygon(plot_coords, closed=False,
                                                xfirst=(dispaxis == 1), origin=0)
                    model_dict = astromodels.chebyshev_to_dict(m_final)

                    # Recalculate aperture limits after rectification
                    apcoords = m_final(np.arange(ext.shape[dispaxis]))
                    model_dict['aper_lower'] = aperture['aper_lower'] + (location - np.min(apcoords))
                    model_dict['aper_upper'] = aperture['aper_upper'] - (np.max(apcoords) - location)
                    all_column_names.extend([k for k in model_dict.keys()
                                             if k not in all_column_names])
                    all_model_dicts.append(model_dict)

                for name in all_column_names:
                    aptable[name] = [model_dict.get(name, 0) for model_dict in all_model_dicts]
                # We don't need to reattach the Table because it was a
                # reference all along!

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)
        return adinputs

    def _get_arc_linelist(self, ext, w1=None, w2=None, dw=None, **kwargs):
        """
        Returns a list of wavelengths of the arc reference lines used by the
        primitive `determineWavelengthSolution()`, if the user parameter
        `linelist=None` (i.e., the default list is requested).

        Parameters
        ----------
        ext : single-slice AD object
            Extension being calibrated (allows descriptors to be calculated).

        w1 : float
            Approximate shortest wavelength (nm).

        w2 : float
            Approximate longest wavelength (nm).

        dw : float
            Approximate dispersion (nm/pixel).

        Returns
        -------
        array_like
            arc line wavelengths

        array_like or None
            arc line weights
        """
        lookup_dir = os.path.dirname(import_module('.__init__', self.inst_lookups).__file__)
        filename = os.path.join(lookup_dir, 'linelist.dat')
        arc_lines = np.loadtxt(filename, usecols=[0])
        try:
            weights = np.loadtxt(filename, usecols=[1])
        except IndexError:
            weights = None
        return arc_lines, weights

    def _get_spectrophotometry(self, filename):
        """
        Reads a file containing spectrophotometric data for a standard star
        and returns these data as a Table(), with unit information. We
        attempt to read a range of files and interpret them, either using
        metadata or guesswork. If there's no metadata, we assume that the
        first column is the wavelength, the second is the brightness data,
        there may then be additional columns with uncertainty information,
        and the width of the bandpass is always the last column.

        We ignore any uncertainty information because, for ground-based data,
        this will be swamped by limitations of the user's data.

        Parameters
        ----------
        filename: str
            name of file containing spectrophotometric data

        Returns
        -------
        Table:
            the spectrophotometric data, with columns 'WAVELENGTH',
            'WIDTH', and 'FLUX'

        Raises
        ------
        FileNotFoundError: if file does not exist
        InconsistentTableError: if the file can't be read as ASCII
        """
        log = self.log
        try:
            tbl = Table.read(filename)
        except IORegistryError:
            # Force ASCII
            tbl = Table.read(filename, format='ascii')

        # Create table, interpreting column names (or lack thereof)
        spec_table = Table()
        colnames = ('WAVELENGTH', 'WIDTH', 'MAGNITUDE')
        aliases = (('WAVE', 'LAMBDA', 'col1'),
                   ('FWHM', 'col3'),
                   ('MAG', 'ABMAG', 'FLUX', 'FLAM', 'FNU', 'col2', 'DATA'))

        for colname, alias in zip(colnames, aliases):
            for name in (colname,) + alias:
                if name in tbl.colnames:
                    spec_table[colname] = tbl[name]
                    orig_colname = name
                    break
            else:
                log.warning("Cannot find a column to convert to '{}' in "
                            "{}".format(colname.lower(), filename))

        # Now handle units
        for col in spec_table.itercols():
            try:
                unit = col.unit
            except AttributeError:
                unit = None
            if isinstance(unit, u.UnrecognizedUnit):
                # Try chopping off the trailing 's'
                try:
                    unit = u.Unit(re.sub(r's$', '', col.unit.name.lower()))
                except:
                    unit = None
            if unit is None:
                # No unit defined, make a guess
                if col.name == 'WAVELENGTH':
                    unit = u.AA if max(col.data) > 5000 else u.nm
                elif col.name == 'WIDTH':
                    unit = spec_table['WAVELENGTH'].unit
                else:
                    if orig_colname == 'FNU':
                        unit = u.Unit("erg cm-2 s-1 Hz-1")
                    elif orig_colname in ('FLAM', 'FLUX') or np.median(col.data) < 1:
                        unit = u.Unit("erg cm-2 s-1 AA-1")
                    else:
                        unit = u.mag
                col.unit = unit

            # We've created a column called "MAGNITUDE" but it might be a flux
            if col.name == 'MAGNITUDE':
                try:
                    unit.to(u.W / u.m ** 3, equivalencies=u.spectral_density(1. * u.m))
                except:
                    pass
                else:
                    col.name = 'FLUX'

        # If we don't have a flux column, create one
        if not 'FLUX' in spec_table.colnames:
            # Use ".data" here to avoid "mag" being in the unit
            spec_table['FLUX'] = (10 ** (-0.4 * (spec_table['MAGNITUDE'].data + 48.6))
                                  * u.Unit("erg cm-2 s-1") / u.Hz)
        return spec_table


# -----------------------------------------------------------------------------
def _average_along_slit(ext, center=None, nsum=None):
    """
    Calculates the average of long the slit and its pixel-by-pixel variance.

    Parameters
    ----------
    ext : `AstroData` slice
        2D spectral image from which trace is to be extracted.

    center : float or None
        Center of averaging region (None => center of axis).

    nsum : int
        Number of rows/columns to combine

    Returns
    -------
    data : array_like
        Averaged data of the extracted region.

    mask : array_like
        Mask of the extracted region.

    variance : array_like
        Variance of the extracted region based on pixel-to-pixel variation.

    extract_slice : slice
        Slice object for extraction region.
    """
    slitaxis = ext.dispersion_axis() - 1
    npix = ext.data.shape[slitaxis]

    if nsum is None:
        nsum = npix
    if center is None:
        center = 0.5 * npix

    extract_slice = slice(max(0, int(center - 0.5 * nsum)),
                          min(npix, int(center + 0.5 * nsum)))
    data, mask, variance = _transpose_if_needed(ext.data, ext.mask, ext.variance,
                                                transpose=(slitaxis == 1),
                                                section=extract_slice)

    # Create 1D spectrum; pixel-to-pixel variation is a better indicator
    # of S/N than the VAR plane

    # FixMe: "variance=variance" breaks test_gmos_spect_ls_distortion_determine.
    #  Use "variance=None" to make them pass again.
    data, mask, variance = NDStacker.mean(data, mask=mask, variance=variance)

    return data, mask, variance, extract_slice


def _transpose_if_needed(*args, transpose=False, section=slice(None)):
    """
    This function takes a list of arrays and returns them (or a section of them),
    either untouched, or transposed, according to the parameter.

    Parameters
    ----------
    args : sequence of arrays
        The input arrays.

    transpose : bool
        If True, return transposed versions.

    section : slice object
        Section of output data to return.

    Returns
    -------
    list of arrays
        The input arrays, or their transposed versions.
    """
    return list(None if arg is None
                else arg.T[section] if transpose else arg[section] for arg in args)


def _read_chebyshev_model(ext):
    try:
        wavecal = dict(zip(ext.WAVECAL["name"], ext.WAVECAL["coefficients"]))
    except (AttributeError, KeyError):
        raise ValueError('missing wave model')
    else:
        return astromodels.dict_to_chebyshev(wavecal)


def _extract_model_info(ext):
    if len(ext.shape) == 1:
        dispaxis = 0
        wave_model = ext.wcs.forward_transform
    else:
        dispaxis = 2 - ext.dispersion_axis()
        wave_model = astromodels.get_named_submodel(ext.wcs.forward_transform, 'WAVE')
    npix = ext.shape[dispaxis]
    limits = wave_model([0, npix])
    w1, w2 = min(limits), max(limits)
    dw = (w2 - w1) / (npix - 1)
    return {'wave_model': wave_model, 'w1': w1, 'w2': w2,
            'npix': npix, 'dw': dw}


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

    if len(mos_ad[0].shape) is not 2:
        raise ValueError("Expected ndim for `mos_ad` to be 2. "
                         "Found {:d}".format(len(mos_ad[0].shape)))

    # Get original relative shift
    origin_shift_x, origin_shift_y = \
        [int(s) for s in mos_ad.phu["origtran"][1:-1].split(',')]

    # Create shift transformation
    shift_x = models.Shift(origin_shift_x - border_size)
    shift_y = models.Shift(origin_shift_y - border_size)

    # Create empty AD
    ad_out = astrodata.create(ref_ad.phu)

    # Loop across all extensions
    for i, ref_ext in enumerate(ref_ad):

        # Create new transformation pipeline
        in_frame = ref_ext.wcs.input_frame
        mos_frame = cf.Frame2D(name="mosaic")

        mosaic_to_pixel = ref_ext.wcs.get_transform(mos_frame, in_frame)

        pipeline = [(mos_frame, mosaic_to_pixel),
                    (in_frame, None)]

        mos_ad[0].wcs = gWCS(pipeline)

        # Shift mosaic in order to set reference (0, 0) on Detector 2
        mos_ad[0].wcs.insert_transform(mos_frame, shift_x & shift_y, after=True)

        # Apply transformation
        temp_ad = transform.resample_from_wcs(
            mos_ad, in_frame.name, origin=(0, 0), output_shape=ref_ext.shape)

        ad_out.append(temp_ad[0])

    return ad_out


def QESpline(coeffs, xpix, data, weights, boundaries, order):
    """
    Fits a cubic spline to data, allowing scaling renormalizations of
    contiguous subsets of the data.

    Parameters
    ----------
    coeffs : array_like
        Scaling factors for CCDs 2+.

    xpix : array
        Pixel numbers (in general, 0..N).

    data : masked_array
        Data to be fit.

    weights: array
        Fitting weights (inverse standard deviations).

    boundaries: tuple
        The last pixel coordinate on each CCD.

    order: int
        Order of spline to fit.

    Returns
    -------
    float
        Normalized chi^2 of the spline fit.
    """
    scaling = np.ones_like(data, dtype=np.float64)
    for coeff, boundary in zip(coeffs, boundaries):
        scaling[boundary:] = coeff
    scaled_data = scaling * data
    scaled_weights = 1. / scaling if weights is None else (weights / scaling).astype(np.float64)
    spline = astromodels.UnivariateSplineWithOutlierRemoval(xpix, scaled_data,
                                                            order=order, w=scaled_weights, niter=1, grow=0)
    result = np.ma.masked_where(spline.mask, np.square((spline.data - scaled_data) *
                                                       scaled_weights)).sum() / (~spline.mask).sum()
    return result


def plot_arc_fit(data, peaks, arc_lines, arc_weights, model, title):
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})
    weights = np.full_like(arc_lines, 3) if arc_weights is None else arc_weights
    for line, wt in zip(arc_lines, weights):
        ax.plot([line, line], [0, 1], color='{}'.format(0.07 * (9 - wt)))
    for peak in model(peaks):
        ax.plot([peak, peak], [0, 1], 'r:')
    ax.plot(model(np.arange(len(data))), data / np.max(data), 'b-')
    limits = model([0, len(data)])
    ax.set_xlim(min(limits), max(limits))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative intensity")
    ax.set_title(title)
