#
#                                                                  gemini_python
#
#                                                             primtives_spect.py
# ------------------------------------------------------------------------------
from geminidr import PrimitivesBASE
from . import parameters_spect
import os
import re

import numpy as np
from numpy.ma.extras import _ezclump
from scipy import spatial, optimize
from scipy.interpolate import UnivariateSpline
from importlib import import_module

from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.io.registry import IORegistryError
from astropy.wcs import WCS
from astropy import units as u

from specutils import SpectralRegion

from matplotlib import pyplot as plt

from gempy.gemini import gemini_tools as gt
from gempy.library.astrotools import array_from_list
from gempy.library import astromodels, matching, tracing
from gempy.library.spectral import Spek1D
from gempy.library.nddops import NDStacker
from gempy.library import transform
from geminidr.gemini.lookups import DQ_definitions as DQ

import astrodata
from astrodata import NDAstroData

from datetime import datetime
from copy import deepcopy

from recipe_system.utils.decorators import parameter_override


# ------------------------------------------------------------------------------
@parameter_override
class Spect(PrimitivesBASE):
    """
    This is the class containing all of the pre-processing primitives
    for the `Spect` level of the type hierarchy tree.
    """
    tagset = set(["GEMINI", "SPECT"])

    def __init__(self, adinputs, **kwargs):
        super(Spect, self).__init__(adinputs, **kwargs)
        self._param_update(parameters_spect)


    def calculateSensitivity(self, adinputs=None, **params):
        """

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            1D spectra of spectrophotometric standard stars

        suffix :  str
            Suffix to be added to output files.

        order: int
            Order of the spline fit to be performed

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
        order = params["order"]

        for ad in adinputs:

            # TODO: Fix when we know how/where these will be stored
            iraf_dir = os.path.join(os.environ.get('iraf'), "noao",
                                    "lib", "onedstds")
            filename = os.path.join(iraf_dir, "spec50cal",
                                    "{}.dat".format(ad.object().lower()))
            spec_table = self._get_spectrophotometry(filename)
            if not spec_table:
                log.warning("Unable to determine sensitivity for {}".
                            format(ad.filename))
                continue

            exptime = ad.exposure_time()

            # Could be XD so iterate over extensions
            for ext in ad:
                if len(ext.shape) != 1:
                    log.warning("{}:{} is not a 1D spectrum".
                                format(ad.filename, ext.hdr['EXTVER']))
                    continue

                spectrum = Spek1D(ext) / (exptime * u.s)

                # Compute values that are counts / (exptime * flux_density * bandpass)
                wave, zpt, zpt_err = [], [], []
                for w0, dw, fluxdens in zip(spec_table['WAVELENGTH'].quantity,
                    spec_table['WIDTH'].quantity, spec_table['FLUX'].quantity):
                    region = SpectralRegion(w0 - 0.5*dw, w0 + 0.5*dw)
                    data, mask, variance = spectrum.signal(region)
                    if not mask and fluxdens > 0:
                        # Regardless of whether FLUX column is f_nu or f_lambda
                        flux = fluxdens.to(u.Unit('erg cm-2 s-1 AA-1'),
                                           equivalencies=u.spectral_density(w0)) * dw
                        wave.append(w0)
                        zpt.append(u.Magnitude(data / flux))
                        zpt_err.append(u.Magnitude(1 + np.sqrt(variance) / data))

                wave = array_from_list(wave)
                zpt = array_from_list(zpt)
                zpt_err = array_from_list(zpt_err)

                spline = astromodels.UnivariateSplineWithOutlierRemoval(wave.value,
                                            zpt.value, w=1./zpt_err.value, order=order)

                #plt.ioff()
                #fig, ax = plt.subplots()
                #ax.plot(wave, zpt, 'ko')
                #x = np.linspace(min(wave), max(wave), ext.shape[0])
                #ax.plot(x, spline(x), 'r-')
                #plt.show()

                knots, coeffs, degree = spline._eval_args
                sensfunc = Table([knots * wave.unit, coeffs * zpt.unit],
                                 names=('knots', 'coefficients'))
                ext.SENSFUNC = sensfunc

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

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            The same input list is used as output but each object now has a
            `.FITCOORD` table appended to each of its extensions. This table
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

        orders = (spectral_order, spatial_order)

        for ad in adinputs:
            xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
            for ext in ad:
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
                ref_coords, in_coords = tracing.trace_lines(ext, axis=1-dispaxis,
                        start=start, initial=initial_peaks, width=5, step=step,
                        nsum=nsum, max_missed=max_missed,
                        max_shift=max_shift*ybin/xbin, viewer=self.viewer)

                ## These coordinates need to be in the reference frame of a
                ## full-frame unbinned image, so modify the coordinates by
                ## the detector section
                #x1, x2, y1, y2 = ext.detector_section()
                #ref_coords = np.array([ref_coords[0] * xbin + x1,
                #                       ref_coords[1] * ybin + y1])
                #in_coords = np.array([in_coords[0] * xbin + x1,
                #                      in_coords[1] * ybin + y1])

                # The model is computed entirely in the pixel coordinate frame
                # of the data, so it could be used as a gWCS object
                m_init = models.Chebyshev2D(x_degree=orders[1-dispaxis],
                                            y_degree=orders[dispaxis],
                                            x_domain=[0, ext.shape[1]],
                                            y_domain=[0, ext.shape[0]])
                #x_domain = [x1, x1 + ext.shape[1] * xbin - 1],
                #y_domain = [y1, y1 + ext.shape[0] * ybin - 1])
                # Find model to transform actual (x,y) locations to the
                # value of the reference pixel along the dispersion axis
                fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                                           sigma_clip, sigma=3)
                m_final, _ = fit_it(m_init, *in_coords, ref_coords[1-dispaxis])
                m_inverse, masked = fit_it(m_init, *ref_coords, in_coords[1-dispaxis])

                # TODO: Some logging about quality of fit
                #print(np.min(diff), np.max(diff), np.std(diff))

                if dispaxis == 1:
                    model = models.Mapping((0, 1, 1)) | (m_final & models.Identity(1))
                    model.inverse = models.Mapping((0, 1, 1)) | (m_inverse & models.Identity(1))
                else:
                    model = models.Mapping((0, 0, 1)) | (models.Identity(1) & m_final)
                    model.inverse = models.Mapping((0, 0, 1)) | (models.Identity(1) & m_inverse)

                self.viewer.color = "blue"
                spatial_coords = np.linspace(ref_coords[dispaxis].min(), ref_coords[dispaxis].max(),
                                             ext.shape[1-dispaxis] // (step*10))
                spectral_coords = np.unique(ref_coords[1-dispaxis])
                for coord in spectral_coords:
                    if dispaxis == 1:
                        xref = [coord] * len(spatial_coords)
                        yref = spatial_coords
                    else:
                        xref = spatial_coords
                        yref = [coord] * len(spatial_coords)
                    #mapped_coords = (np.array(model.inverse(xref, yref)).T -
                    #                 np.array([x1, y1])) / np.array([xbin, ybin])
                    mapped_coords = np.array(model.inverse(xref, yref)).T
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
                #ext.COORDS = Table([*in_coords] + [*ref_coords], names=('xin','yin','xref','yref'))

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
            Order of interpolation when resampling.
        subsample : int
            Pixel subsampling factor.

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

        # The forward Transform is *only* used to determine the shape
        # of the output image, which we want to be the same as the input
        m_ident = models.Identity(2)

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
                    raise IOError('No processed arc listed for {}'.
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
            distortion_models = []
            for ext in arc:
                fitcoord = ext.FITCOORD
                model_dict = dict(zip(fitcoord['inv_name'],
                                      fitcoord['inv_coefficients']))
                m_inverse = astromodels.dict_to_chebyshev(model_dict)
                if not isinstance(m_inverse, models.Chebyshev2D):
                    log.warning("Could not read distortion model from arc {}:{}"
                                " - continuing".format(arc.filename, ext.hdr['EXTVER']))
                    adoutputs.append(ad)
                    continue

                # Use AstroPy Compound Models to have a model that can be applied to 2D data:
                # https://docs.astropy.org/en/stable/modeling/compound-models.html#advanced-mappings
                dispaxis = arc[0].dispersion_axis() - 1
                if dispaxis == 0:
                    m_ident.inverse = models.Mapping((0, 1, 1)) | (m_inverse & models.Identity(1))
                else:
                    m_ident.inverse = models.Mapping((0, 0, 1)) | (models.Identity(1) & m_inverse)
                distortion_models.append(m_ident.copy())

            # Determine whether we're producing a single-extension AD
            # or keeping the number of extensions as-is
            if len_arc == 1:
                arc_detsec = arc.detector_section()[0]
                ad_detsec = ad.detector_section()
                if len_ad > 1:
                    # We need to apply the mosaicking geometry, and add the
                    # same distortion correction to each input extension.
                    # The problem is that the arc may be a larger area than
                    # the science frame, and the arc's pixel coordinates have
                    # had the "origin shift" applied after mosaicking. So we
                    # need to work out what that was and apply it so that the
                    # science frame's pixel coords after mosaicking (i.e., in
                    # the middle of this transform) match those of the
                    # mosaicked arc. We assume that one of the shifts will be
                    # zero (i.e., that the science frame is a subregion of the
                    # arc only along one dimension).
                    geotable = import_module('.geometry_conf', self.inst_lookups)
                    adg = transform.create_mosaic_transform(ad, geotable)
                    shifts = [c1 - c2 for c1, c2 in zip(np.array(ad_detsec).min(axis=0),
                                                        arc_detsec)]
                    xshift, yshift = shifts[0] / xbin, shifts[2] / ybin  # x1, y1
                    if xshift or yshift:
                        log.stdinfo("Found a shift of ({},{}) pixels between "
                                    "{} and the calibration.".
                                    format(xshift, yshift, ad.filename))
                    add_shapes, add_transforms = [], []
                    for (arr, trans) in adg:
                        # Try to work out shape of this Block in the unmosaicked
                        # arc, and then apply a shift to align it with the
                        # science Block before applying the same transform.
                        if xshift == 0:
                            add_shapes.append(((arc_detsec.y2 - arc_detsec.y1) // ybin, arr.shape[1]))
                        else:
                            add_shapes.append((arr.shape[0], (arc_detsec.x2 - arc_detsec.x1) // xbin))
                        t = transform.Transform(models.Shift(-xshift) & models.Shift(-yshift))
                        t.append(trans)
                        add_transforms.append(t)
                    adg.calculate_output_shape(additional_array_shapes=add_shapes,
                                               additional_transforms=add_transforms)
                    # This tells us where the arc would have been in pixel
                    # space after mosaicking, so make sure the science frame
                    # has the same coordinates, and then apply the distortion
                    # correction transform
                    origin_shift = models.Shift(-adg.origin[1]) & models.Shift(-adg.origin[0])
                    for t in adg.transforms:
                        t.append([origin_shift, m_ident.copy()])
                    # And recalculate output_shape and origin properly
                    adg.calculate_output_shape()
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
                        adg = transform.AstroDataGroup(ad, [transform.Transform([m_shift, m_ident])])
                    else:
                        adg = transform.AstroDataGroup(ad, [transform.Transform(m_ident)])

                ad_out = adg.transform(order=order, subsample=subsample, parallel=False)
                try:
                    ad_out[0].WAVECAL = arc[0].WAVECAL
                except AttributeError:
                    log.warning("No WAVECAL table in {}".format(arc.filename))
            else:
                log.warning("Distortion correction with multiple-extension "
                            "arcs has not been tested.")
                for i, (ext, ext_arc, model) in enumerate(zip(ad, arc, distortion_models)):
                    # Shift science so its pixel coords match the arc's before
                    # applying the distortion correction
                    shifts = [c1 - c2 for c1, c2 in zip(ext.detector_section(),
                                                        ext_arc.detector_section())]
                    t = transform.Transform([models.Shift(shifts[0] / xbin) &
                                             models.Shift(shifts[1] / ybin), model])
                    adg = transform.AstroDataGroup([ext], t)
                    adg.set_reference()
                    if i == 0:
                        ad_out = adg.transform(order=order, subsample=subsample,
                                               parallel=False)
                    else:
                        ad_out.append(adg.transform(order=order, subsample=subsample,
                                                    parallel=False))
                    try:
                        ad_out[i].WAVECAL = arc[i].WAVECAL
                    except AttributeError:
                        log.warning("No WAVECAL table in {}:{}".
                                    format(arc.filename, arc[i].hdr['EXTVER']))

            # Timestamp and update the filename
            gt.mark_history(ad_out, primname=self.myself(), keyword=timestamp_key)
            ad_out.update_filename(suffix=sfx, strip=True)
            adoutputs.append(ad_out)

        return adoutputs

    def determineWavelengthSolution(self, adinputs=None, **params):
        """
        This primitive determines the wavelength solution for an ARC and
        stores it as an attached `.WAVECAL` table.

        2D input images are converted to 1D by collapsing a slice of the image
        along the dispersion direction, and peaks are identified. These are then
        matched to an arc line list, using the `KDTreeFitter`.

        For each `AstroData` object and for each extension within it, this
        primitive contains the following bigger steps:

        - Read Arc Lines;

        - Extract 1D spectrum;

        - Mask data skipping non-linear or saturated lines;

        - Use known central wavelength to calculate wavelength range and
          dispersion;

        - Estimate line widths to be used on peak detection;

        - Detect peaks using Continuous Wavelet Transform;

        - Create weight dictionary with different types of weights
          (intensity/flux);

        - Create iteration sequence with different weights methods;

        - Create 0th iteration using a 1D Chebyshev model;

        - Fit iteration using `KDTreeFitter`;

        - Matching using `Chebyshev1DMatchBox`;

        - Convert model into dictionary and, then, into a Table;

        - Store solution as a Table.


        Parameters
        ----------

        adinputs : list of :class:`~astrodata.AstroData`
             Mosaicked Arc data as 2D spectral images or 1D spectra.

        suffix : str
            Suffix to be added to output files.

        center : int or None
            Central row/column for 1D extraction (None => use middle).

        nsum : int
            Number of rows/columns to average.

        order : int
            Order of Chebyshev fitting function.

        min_snr : float
            Minimum S/N ratio in line peak to be used in fitting.

        fwidth : float
            Expected width of arc lines in pixels.

        linelist : str or None
            Name of file containing arc lines.

        weighting : {'none', 'natural', 'relative'}
            How to weight the detected peaks.

        nbright : int
            Number of brightest lines to cull before fitting.

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

        plot = params["plot"]
        plt.ioff()

        # TODO: This decision would prevent MOS data being reduced so need
        # to think a bit more about what we're going to do. Maybe make
        # central_wavelength() return a one-per-ext list? Or have the GMOS
        # determineWavelengthSolution() recipe check the input has been
        # mosaicked before calling super()?
        #
        # Top-level decision for this to only work on single-extension ADs
        #if not all(len(ad)==1 for ad in adinputs):
        #    raise ValueError("Not all inputs are single-extension AD objects")

        # Get list of arc lines (probably from a text file dependent on the
        # input spectrum, so a private method of the primitivesClass)
        linelists = {}
        if arc_file is not None:
            try:
                arc_lines = np.loadtxt(arc_file, usecols=[0])
            except (IOError, TypeError):
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
                    direction = "row" if ext.dispersion_axis() == 1 else "column"
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

                #arc_weights = None

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
                                     distance_upper_bound=abs(0.1*len(data)*dw))[1]
                snrs = np.array(list(peak_snrs) + [np.nan])[indices]
                # Normalize weights by the maximum of these lines
                weights['relative'] = peak_snrs / np.nanmedian(snrs, axis=1)

                plot_data = data

                # Some diagnostic plotting
                yplot = 0
                plt.ioff()
                if plot:
                    fig, ax = plt.subplots()

                # The very first model is called "m_final" because at each
                # iteration the next initial model comes from the fitted
                # (final) model of the previous iteration
                m_final = models.Chebyshev1D(degree=1, c0=cenwave,
                                             c1=0.5*dw*len(data), domain=[0, len(data)-1])
                if plot:
                    plot_arc_fit(data, peaks, arc_lines, arc_weights, m_final, "Initial model")
                log.stdinfo('Initial model: {}'.format(repr(m_final)))

                kdsigma = fwidth * abs(dw)
                peaks_to_fit = peak_snrs > min_snr
                peaks_to_fit[np.argsort(peak_snrs)[len(peaks)-nbright:]] = False

                # Temporary code to help with testing
                try:
                    sequence = self.fit_sequence
                except AttributeError:
                    sequence = (((1, 'none', 'basinhopping', ['c1']), (2, 'none', 'basinhopping', ['c1'])) +
                                tuple((order, 'relative', 'Nelder-Mead') for order in range(2, order+1)))

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
                    c0_unc = 0.05 * cenwave
                    m_init.c0.bounds = (m_init.c0 - c0_unc, m_init.c0 + c0_unc)
                    c1_unc = 0.005 * abs(m_init.c1)
                    m_init.c1.bounds = tuple(sorted([m_init.c1 - c1_unc, m_init.c1 + c1_unc]))
                    for i in range(2, ord + 1):
                        getattr(m_init, 'c{}'.format(i)).bounds = (-5, 5)

                    if fixems is not None:
                        for fx in fixems:
                            getattr(m_init, fx).fixed = True

                    fit_it = matching.KDTreeFitter(sigma=kdsigma, maxsig=10, k=3, method=method)
                    m_final = fit_it(m_init, peaks[peaks_to_fit], arc_lines,
                                     in_weights=in_weights[peaks_to_fit],
                                     ref_weights=None if weight_type is 'none' else arc_weights)
                    #                 method='basinhopping' if weight_type is 'none' else 'Nelder-Mead')
                    #                 options={'xtol': 1.0e-7, 'ftol': 1.0e-8})

                    log.stdinfo('{} {}'.format(repr(m_final), fit_it.statistic))
                    if plot:
                        plot_arc_fit(plot_data, peaks, arc_lines, arc_weights, m_final,
                                     "KDFit model order {} KDsigma = {}".format(ord, kdsigma))

                    match_radius = 2 * fwidth * abs(m_final.c1) / len(data)  # fwidth pixels
                    m_final._constraints['bounds'] = {p: (None, None)
                                                      for p in m_final.param_names}
                    m = matching.Chebyshev1DMatchBox.create_from_kdfit(peaks, arc_lines,
                                                                       model=m_final, match_radius=match_radius,
                                                                       sigma_clip=3)
                    #kdsigma = m.rms_output
                    #print("New kdsigma {}".format(kdsigma))
                    kdsigma = fwidth * abs(dw)
                    yplot += 1

                # Remove bounds from the model
                m_final._constraints['bounds'] = {p: (None, None)
                                                  for p in m_final.param_names}
                match_radius = 2 * fwidth * abs(m_final.c1) / len(data)  # fwidth pixels
                # match_radius = kdsigma
                m = matching.Chebyshev1DMatchBox.create_from_kdfit(peaks, arc_lines,
                                model=m_final, match_radius=match_radius, sigma_clip=3)
                if plot:
                    for incoord, outcoord in zip(m.forward(m.input_coords), m.output_coords):
                        ax.text(incoord, yplot, '{:.4f}'.format(outcoord), rotation=90,
                                ha='center', va='top')

                log.stdinfo('{} {} {}'.format(repr(m.forward), len(m.input_coords), m.rms_output))
                if plot:
                    plot_arc_fit(plot_data, peaks, arc_lines, arc_weights, m.forward,
                                 "MatchBox model order {ord}")

                # Choice of kdsigma can have a big effect. This oscillates
                # around the initial choice, with increasing amplitude.
                #kdsigma = 10.*abs(dw) * (((1.0+0.1*((kditer+1)//2)))**((-1)**kditer)
                #                    if kditer<21 else 1)

                m_final = m.forward
                rms = m.rms_output
                nmatched = len(m.input_coords)
                log.stdinfo(m_final)
                log.stdinfo("Matched {} lines with rms = {:.3f} nm.".format(nmatched, rms))

                if plot:
                    plot_arc_fit(plot_data, peaks, arc_lines, arc_weights, m_final, "Final fit")
                    m.display_fit()
                    plt.show()

                m.display_fit()

                if plot:
                    plt.savefig(ad.filename.replace('.fits', '.jpg'))

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

        width : float
            Width of extraction aperture (in pixels).

        grow : float or None
            Avoidance region around each source aperture if a sky aperture
            is required.

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
                self.viewer.display_image(ext, wcs=False)
                if len(ext.shape) == 1:
                    log.warning("{} is already one-dimensional".format(extname))
                    continue

                try:
                    aptable = ext.APERTURE
                except AttributeError:
                    log.warning("{} has no APERTURE table. Cannot extract "
                                "spectra.".format(extname))
                    continue

                num_spec = len(aptable)
                log.stdinfo("Extracting {} spectra from {}".format(num_spec,
                                                                   extname))

                dispaxis = 2 - ext.dispersion_axis()  # python sense
                direction = "row" if dispaxis == 1 else "column"
                # Create dict of wavelength keywords to add to new headers
                # TODO: Properly. Simply put the linear approximation here for now
                hdr_dict = {'CTYPE1': 'Wavelength',
                            'CUNIT1': 'nanometer'}

                try:
                    wavecal = dict(zip(ext.WAVECAL["name"],
                                       ext.WAVECAL["coefficients"]))
                except (AttributeError, KeyError):
                    log.warning("Could not read wavelength solution for {}:{}"
                                "".format(ad.filename, ext.hdr['EXTVER']))
                    hdr_dict.update({'CRPIX1': 0.5 * (ext.shape[dispaxis] + 1),
                                     'CRVAL1': ext.central_wavelength(asNanometers=True),
                                     'CDELT1': ext.dispersion(asNanometers=True)})
                else:
                    wave_model = astromodels.dict_to_chebyshev(wavecal)
                    hdr_dict.update({'CRPIX1': 0.5 * np.sum(wave_model.domain) + 1,
                                     'CRVAL1': wave_model.c0.value,
                                     'CDELT1': 2 * wave_model.c1.value / np.diff(wave_model.domain)[0]})
                hdr_dict['CD1_1'] = hdr_dict['CDELT1']

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

                for i, aperture in enumerate(apertures):
                    log.stdinfo("    Extracting spectrum from aperture {}".format(i + 1))
                    self.viewer.width = 2
                    self.viewer.color = colors[i % len(colors)]
                    ndd_spec = aperture.extract(ext, width=width,
                                                method=method, viewer=self.viewer)

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
                                     (max_ + offset + 0.5 * sky_width > ext.shape[1-dispaxis] - 0.5)):
                                    break

                                sky_trace_model = aperture.model | models.Shift(offset)
                                sky_aperture = tracing.Aperture(sky_trace_model)
                                sky_spec = sky_aperture.extract(apmask, width=sky_width, dispaxis=dispaxis)
                                if np.sum(sky_spec.data) == 0:
                                    sky_spectra.append(sky_aperture.extract(ext, width=sky_width,
                                                                            viewer=self.viewer))
                                    ok = True
                                offset += direction * offset_step

                        if sky_spectra:
                            # If only one, add it to itself (since it's half-width)
                            sky_spec = sky_spectra[0].add(sky_spectra[-1])
                            ad_spec.append(ndd_spec.subtract(sky_spec, handle_meta='first_found',
                                                             handle_mask=np.bitwise_or))
                        else:
                            log.warning("Difficulty finding sky aperture. No sky"
                                        " subtraction for aperture {}".format(i))
                            ad_spec.append(ndd_spec)
                    else:
                        ad_spec.append(ndd_spec)

                    # Copy wavelength solution and add a header keyword
                    # with the extraction location
                    try:
                        ad_spec[-1].WAVECAL = ext.WAVECAL
                    except AttributeError:  # That's OK, there wasn't one
                        pass
                    center = model_dict['c0']
                    ad_spec[-1].hdr['XTRACTED'] = (center, "Spectrum extracted "
                                "from {} {}".format(direction, int(center + 0.5)))

                    # Delete some header keywords
                    for kw in ("CTYPE", "CRPIX", "CRVAL", "CUNIT", "CD1_", "CD2_"):
                        for ax in (1, 2):
                            try:
                                del ad_spec[-1].hdr["{}{}".format(kw, ax)]
                            except KeyError:
                                pass

                    for k, v in hdr_dict.items():
                        ad_spec[-1].hdr[k] = (v, self.keyword_comments.get(k))

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

        max_apertures : int
            Maximum number of apertures expected to be found.

        threshold : float
            height above background (relative to peak) at which to define
            the edges of the aperture

        min_sky_pix : int
            minimum number of contiguous pixels between sky lines
            for a region to be added to the spectrum before collapsing to 1D

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
                mask1d[var1d > mean+sigma] = 1
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
                                                                 rejector="sigclip", combiner="mean")

                # TODO: find_peaks might not be best considering we have no
                # idea whether sources will be extended or not
                widths = np.arange(5, 30)
                peaks_and_snrs = tracing.find_peaks(profile, widths, mask=prof_mask,
                                                    variance=prof_var, reject_bad=False,
                                                    min_snr=3, min_frac=0.2)
                # Reverse-sort by SNR and return only the locations
                locations = np.array(sorted(peaks_and_snrs.T, key=lambda x: x[1],
                                            reverse=True)[:max_apertures]).T[0]
                log.stdinfo("Found sources at {}s: {}".format(direction,
                            ' '.join(['{:.1f}'.format(loc) for loc in locations])))

                all_limits = tracing.get_limits(profile, prof_mask, peaks=locations,
                                                threshold=threshold, method=limit_method)

                all_model_dicts = []
                for loc, limits in zip(locations, all_limits):
                    cheb = models.Chebyshev1D(degree=0, domain=[0, npix-1], c0=loc)
                    model_dict = astromodels.chebyshev_to_dict(cheb)
                    model_dict['aper_lower'] = limits[0] - loc
                    model_dict['aper_upper'] = limits[1] - loc
                    all_model_dicts.append(model_dict)

                aptable = Table([np.arange(len(locations))+1], names=['number'])
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

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            1D spectra of targets that need to be flux-calibrated

        suffix :  str
            Suffix to be added to output files.

        standard: str/AstroData/None
            Name/AD instance of the standard star spectrum with a SENSFUNC
            table attached

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

        flux_units = u.Unit("W m-2")

        # Get a suitable arc frame (with distortion map) for every science AD
        if std is None:
            raise NotImplementedError("Cannot perform automatic standard star"
                                      " association")
            #self.getProcessedStandard(adinputs, refresh=False)
            #std_list = self._get_cal(adinputs, 'processed_standard')
        else:
            std_list = std

        for ad, std in zip(*gt.make_lists(adinputs, std_list, force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by fluxCalibrate".
                            format(ad.filename))
                continue

            if len(ad) != len(std):
                log.warning("{} has {} extensions so cannot be used to "
                            "calibrate {} with {} extensions.".
                            format(std.filename, len(std), ad.filename, len(ad)))
                continue

            exptime = ad.exposure_time()

            for ext, ext_std in zip(ad, std):
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
                wcs = WCS(ext.hdr)
                ndim = len(ext.shape)
                dispaxis = 0 if ndim == 1 else 2 - ext.dispersion_axis()
                wave_unit = u.Unit(ext.hdr['CUNIT{}'.format(ndim-dispaxis)])

                # Get wavelengths and pixel sizes of all the pixels along the dispersion axis
                coords = np.arange(-0.5, ext.shape[dispaxis], 0.5)
                if ndim == 2:
                    other_axis = np.full_like(coords, 0.5*(ext.shape[1-dispaxis] - 1))
                    coords = [coords, other_axis] if dispaxis == 1 else [other_axis, coords]
                else:
                    coords = [coords]
                all_waves = wcs.all_pix2world(*coords, 0) * wave_unit
                if ndim == 2:
                    all_waves = all_waves[1-dispaxis]
                else:
                    all_waves = all_waves[0]
                waves = all_waves[1::2]
                pixel_sizes = np.diff(all_waves[::2])

                # Reconstruct the spline and evaluate it at every wavelength
                tck = (sensfunc['knots'].data, sensfunc['coefficients'].data, 3)
                spline = UnivariateSpline._from_tck(tck)
                sens_factor = spline(waves.to(sensfunc['knots'].unit)) * sensfunc['coefficients'].unit
                try:
                    sens_factor = sens_factor.physical
                except AttributeError:
                    pass
                final_sens_factor = (sci_flux_unit / (sens_factor * pixel_sizes)).to(final_units, equivalencies=u.spectral_density(waves)).value

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

        # There are either 1 or 4 Nones, due to validation
        nones = [w1, w2, dw, npix].count(None)
        if nones == 1:
            # Work out the missing variable from the others
            if npix is None:
                npix = int(np.ceil((w2 - w1) / dw)) + 1
                w2 = w1 + (npix-1) * dw
            elif w1 is None:
                w1 = w2 - (npix-1) * dw
            elif w2 is None:
                w2 = w1 + (npix-1) * dw
            else:
                dw = (w2 - w1) / (npix-1)

        for ad in adinputs:
            for ext in ad:
                extname = "{}:{}".format(ad.filename, ext.hdr['EXTVER'])

                attributes = [attr for attr in ('data', 'mask', 'variance')
                              if getattr(ext, attr) is not None]
                try:
                    wavecal = dict(zip(ext.WAVECAL["name"],
                                       ext.WAVECAL["coefficients"]))
                except (AttributeError, KeyError):
                    cheb = None
                else:
                    cheb = astromodels.dict_to_chebyshev(wavecal)
                if cheb is None:
                    log.warning("{} has no WAVECAL. Cannot linearize.".
                                format(extname))
                    continue

                if nones == 4:
                    npix = ext.data.size
                    limits = cheb([0, npix-1])
                    w1, w2 = min(limits), max(limits)
                    dw = (w2 - w1) / (npix - 1)

                log.stdinfo("Linearizing {}: w1={:.3f} w2={:.3f} dw={:.3f} "
                            "npix={}".format(extname, w1, w2, dw, npix))

                cheb.inverse = astromodels.make_inverse_chebyshev1d(cheb, rms=0.1)
                t = transform.Transform(cheb)

                # Linearization (and inverse)
                t.append(models.Shift(-w1))
                t.append(models.Scale(1. / dw))

                # If we resample to a coarser pixel scale, we may interpolate
                # over features. We avoid this by subsampling back to the
                # original pixel scale (approximately).
                input_dw = np.diff(cheb(cheb.domain))[0] / np.diff(cheb.domain)
                subsample = abs(dw / input_dw)
                if subsample > 1.1:
                    subsample = int(subsample + 0.5)

                dg = transform.DataGroup([ext], [t])
                dg.output_shape = (npix,)
                output_dict = dg.transform(attributes=attributes, subsample=subsample,
                                           conserve=conserve)
                for key, value in output_dict.items():
                    setattr(ext, key, value)

                ext.hdr["CRPIX1"] = 1.
                ext.hdr["CRVAL1"] = w1
                ext.hdr["CDELT1"] = dw
                ext.hdr["CD1_1"] = dw
                ext.hdr["CUNIT1"] = "nanometer"

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

        return adinputs

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

        TODO: There is an issue here because the spline knots are equally
        spaced but their separation should be much larger than any data-free
        inter-chip gap, so this effectively sets a maxmimum spline order
        which might not be very high. Can/should we set the spline knots only
        within each chip?

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
        sfx = params["suffix"]
        spectral_order = params["spectral_order"]
        center = params["center"]
        nsum = params["nsum"]

        for ad in adinputs:
            # Don't mosaic if the multiple extensions are because the
            # data are MOS or cross-dispersed
            if len(ad) > 1 and not({'MOS', 'XD'} & ad.tags):
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
                mask |= (DQ.no_data * (variance==0))  # Ignore var=0 points
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
                            weights, boundaries, spectral_order), tol=1e-7, method='Nelder-Mead')
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
                                                    order=spectral_order, w=weights)

                if not mosaicked:
                    flat_data = np.tile(spline.data, (ext.shape[dispaxis-1], 1))
                    ext.divide(_transpose_if_needed(flat_data, transpose=(dispaxis==2))[0])

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

        suffix : str
            Suffix to be added to output files.

        order : int or None
            Order of piecewise cubic spline fit to each row/column. If `None`,
            it uses as many pieces as required to get chi^2=1. Else, it is
            reduced proportionately by the ratio between the number of good pixels
            in each row/column and the total number of pixels.

        width : float or None
            Width in pixels for each aperture, if not specified in the
            APERTURE table. `None` will use the value in the aperture table
            and, if one doesn't exist there, will result in the optimal width
            being calculated for each aperture.

        grow : float
            Masking growth radius (in pixels) for each aperture.

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
        default_width = params["width"]
        grow = params["grow"]

        for ad in adinputs:
            if self.timestamp_keys['distortionCorrect'] not in ad.phu:
                log.warning("{} has not been distortion corrected. Sky "
                            "subtraction is likely to be poor.".format(ad.filename))

            start = datetime.now()
            for ext in ad:
                dispaxis = 2 - ext.dispersion_axis()  # python sense
                slitlen = ext.shape[1-dispaxis]
                pixels = np.arange(slitlen)

                # We want to mask pixels in apertures in addition to the mask
                sky_mask = (np.zeros_like(data, dtype=DQ.datatype)
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
                        aperture = tracing.Aperture(trace_model)
                        width = model_dict.get('width', default_width)
                        sky_mask |= aperture.aperture_mask(ext, width=width, grow=grow)

                # Transpose if needed so we're iterating along rows
                data, mask, var = _transpose_if_needed(ext.data, sky_mask, ext.variance, transpose=dispaxis==1)
                sky_model = np.empty_like(data)
                sky_weights = (np.ones_like(data) if var is None
                               else np.sqrt(np.divide(1., var, out=np.zeros_like(data),
                                                      where=var>0)))

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
            Fitting order along spectrum.

        step : int
            Step size for sampling along dispersion direction.

        nsum : int
            Number of rows/columns to combine at each step.

        max_missed : int
            Maximum number of interactions without finding line before line is
            considered lost forever.

        max_shift : float
            Maximum perpendicular shift (in pixels) from pixel to pixel.

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
        # timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        order = params["trace_order"]
        step = params["step"]
        nsum = params["nsum"]
        max_missed = params["max_missed"]
        max_shift = params["max_shift"]

        for ad in adinputs:
            for ext in ad:
                try:
                    aptable = ext.APERTURE
                    locations = aptable['c0'].data
                except (AttributeError, KeyError):
                    log.warning("Could not find aperture locations in {}:{} - "
                                "continuing".format(ad.filename, ext.hdr['EXTVER']))
                    continue

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
                                                                    max_shift=max_shift, viewer=self.viewer)

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
                    m_init = models.Chebyshev1D(degree=order,
                                                domain=[0, ext.shape[dispaxis] - 1])
                    fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                                               sigma_clip, sigma=3)
                    m_final, _ = fit_it(m_init, in_coords[1 - dispaxis], in_coords[dispaxis])
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
            # gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
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
        """
        log = self.log
        try:
            tbl = Table.read(filename)
        except FileNotFoundError:
            log.warning("File {} not found!".format(filename))
            return
        except IORegistryError:
            try:
                tbl = Table.read(filename, format='ascii')
            except:
                self.log.warning("Cannot read file {}".format(filename))
                return
        num_columns = len(tbl.columns)

        # Create table, interpreting column names (or lack thereof)
        spec_table = Table()
        colnames = ('WAVELENGTH', 'WIDTH', 'MAGNITUDE')
        aliases = (('WAVE', 'LAMBDA', 'col1'),
                   ('FWHM', 'col{}'.format(min(3, num_columns))),
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
                return

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
                        unit = u.Unit("erg cm-2 s-1") / u.Hz
                        col.name = 'FLUX'
                    elif orig_colname in ('FLAM', 'FLUX') or np.median(col.data) < 1:
                        unit = u.Unit("erg cm-2 s-1") / u.AA
                        col.name = 'FLUX'
                    else:
                        unit = u.mag
                col.unit = unit

        # If we don't have a flux column, create one
        if not 'FLUX' in spec_table.colnames:
            # Use ".data" here to avoid "mag" being in the unit
            spec_table['FLUX'] = (10**(-0.4*(spec_table['MAGNITUDE'].data + 48.6))
                                  * u.Unit("erg cm-2 s-1") / u.Hz)
        return spec_table

#-----------------------------------------------------------------------------

def _average_along_slit(ext, center=None, nsum=None):
    """
    Calculated the average of long the slit and its pixel-by-pixel variance.

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
    extraction = center or (0.5 * ext.data.shape[slitaxis])
    extract_slice = slice(max(0, int(extraction - 0.5 * nsum)),
                          min(ext.data.shape[slitaxis],
                              int(extraction + 0.5 * nsum)))
    data, mask, variance = _transpose_if_needed(ext.data, ext.mask, ext.variance,
                                                transpose=(slitaxis == 1), section=extract_slice)

    # Create 1D spectrum; pixel-to-pixel variation is a better indicator
    # of S/N than the VAR plane
    data, mask, variance = NDStacker.mean(data, mask=mask, variance=None)

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
