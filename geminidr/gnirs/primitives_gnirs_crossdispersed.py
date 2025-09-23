#
#                                                                 gemini_python
#
#                                            primitives_gnirs_crossdispersed.py
# -----------------------------------------------------------------------------
import astrodata, gemini_instruments

from astropy.table import Table
import numpy as np

from gempy.gemini import gemini_tools as gt
from gempy.library import astrotools as at
from recipe_system.utils.decorators import (parameter_override,
                                            capture_provenance)
from geminidr.gemini.lookups import DQ_definitions as DQ

from .primitives_gnirs_spect import GNIRSSpect
from . import parameters_gnirs_crossdispersed
from geminidr.core.primitives_crossdispersed import CrossDispersed
from .lookups.MDF_XD import get_slit_info
from ..gemini.lookups.timestamp_keywords import timestamp_keys


# -----------------------------------------------------------------------------
@parameter_override
@capture_provenance
class GNIRSCrossDispersed(GNIRSSpect, CrossDispersed):
    """This class contains all of the preprocessing primitives for the
    GNIRSCrossDispersed level of the type hierarchy tree. It inherits all the
    primitives from the above level.
    """
    tagset = {"GEMINI", "GNIRS", "SPECT", "XD"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_gnirs_crossdispersed)

    # TODO: handle _fields_overlap()


    def addMDF(self, adinputs=None, suffix=None):
        """
        This GNIRS XD-specific implementation of addMDF() calls
        primitives_gemini._addMDF() on each astrodata object to attach the MDFs.
        It also attaches two columns, 'slitlength_arcsec' and 'slitlength_pixels',
        with the length of the slit in arcseconds and pixels, respectively.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        for ad in adinputs:
            mdf_key_parts = ('telescope', '_prism', 'decker',
                             '_grating', 'camera')
            mdf_key = "_".join(getattr(ad, desc)()
                               for desc in mdf_key_parts)
            x_ccd, y_ccd, length_pix = get_slit_info(mdf_key, ad.central_wavelength(), ad._grating_order())
            mdf_table = Table([range(1, len(x_ccd) + 1), x_ccd], names=['slit_id', 'x_ccd'])
            mdf_table['y_ccd'] = y_ccd
            mdf_table['specorder'] = mdf_table['slit_id'] + 2
            mdf_table['slitlength_asec'] = length_pix * ad.pixel_scale()
            mdf_table['slitlength_pixels'] = length_pix
            ad.MDF = mdf_table
            log.stdinfo(f"Added MDF table for {ad.filename}")

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)

        return adinputs

    def distortionCorrect(self, adinputs=None, **params):
        """
        Corrects optical distortion in science frames, using a distortion map
        (a Chebyshev2D model, usually from a processed arc) that has previously
        been attached to each input's WCS by attachWavelengthSolution.

        If the input image requires mosaicking, then this is done as part of
        the resampling, to ensure one, rather than two, interpolations.

        This GNIRS XD version of this primitive adds a further step where the
        real estate of the distortion-corrected image is trimmed down to the
        minimum required to contain the data.

        Parameters
        ----------
        suffix : str/None
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
        adinputs = super().distortionCorrect(adinputs=adinputs, **params)

        # CJS 20240814: distortion-corrected XD data can have a lot of
        # unnecessary real estate, so trim this down. By constructing the
        # distortion model well, we can ensure that all this real estate is
        # on the right/top of the frame.
        # TODO: This is probably not the correct place for this and it may
        # need to be moved as other instrument modes are supported. It can
        # probably go in the main primitive in Spect but let's wait and see.
        adoutputs = []
        for ad in adinputs:
            adout = astrodata.create(ad.phu)
            adout.filename = ad.filename
            adout.orig_filename = ad.orig_filename
            for ext in ad:
                # Code generically for the dispersion axis
                dispaxis = 2 - ext.dispersion_axis()  # python sense
                fully_masked = (ext.mask & (DQ.no_data | DQ.unilluminated)).astype(
                    bool).all(axis=dispaxis)
                first_masked = ext.shape[1-dispaxis] - fully_masked[::-1].argmin()
                if dispaxis == 0:
                    adout.append(ext.nddata[:, :first_masked])
                    log.debug(f"Cutting {ad.filename}:{ext.id} right of "
                              f"column {first_masked}")
                else:
                    adout.append(ext.nddata[:first_masked])
                    log.debug(f"Cutting {ad.filename}:{ext.id} above row"
                              f" {first_masked}")
            adoutputs.append(adout)

        return adoutputs

    def determinePinholeRectification(self, adinputs=None, **params):
        """
        This primitive exists to provide some mode-specific values to the
        determinePinholeRectification() in primitives_spect, since the modes in GNIRS
        cross-dispersed are different enough to warrant having different values.
        """

        camera = getattr(adinputs[0], 'camera')()

        # CJS: Replace with 'debug_avoidance' parameter
        # if 'Short' in camera:
        #     # In the short camera configuration there are four good pinholes
        #     # and one that's right on the edge of the slit and isn't consistently
        #     # picked up. This setting stops it from being used in the orders it
        #     # is found in since it produces a sketchy fit.
        #     if params['debug_max_trace_pos'] is None:
        #         params['debug_max_trace_pos'] = 4
        #         self.log.debug("Setting debug_max_trace_pos to 4 for Short "
        #                        "camera.")

        if 'Long' in camera:
            # In the long camera configuration the 5th and 6th slits run off the
            # side of the array, necessitating a start point much closer to the
            # bottom instead of the default middle-of-the-array.
            if params['start_pos'] is None:
                params['start_pos'] = 150
                self.log.fullinfo("Setting trace start location to row 150 for "
                                  "Long camera.")

        # Call the parent primitive with the new parameter values.
        return super().determinePinholeRectification(adinputs=adinputs, **params)


    def determineWavelengthSolution(self, adinputs=None, **params):
        """
        Determines the wavelength solution for an ARC and updates the wcs
        with this solution. In addition, the solution and pixel/wavelength
        matches are stored as an attached `WAVECAL` :class:`~astropy.table.Table`.

        2D input images are converted to 1D by collapsing a slice of the image
        along the dispersion direction, and peaks are identified. These are then
        matched to an arc line list, using piecewise-fitting of (usually)
        linear functions to match peaks to arc lines, using the
        :class:`~gempy.library.matching.KDTreeFitter`.

        The `.WAVECAL` table contains four columns:
            ["name", "coefficients", "peaks", "wavelengths"]

        The `name` and the `coefficients` columns contain information to
        re-create an Chebyshev1D object, plus additional information about
        the way the spectrum was collapsed. The `peaks` column contains the
        (1-indexed) position of the lines that were matched to the catalogue,
        and the `wavelengths` column contains the matched wavelengths.

        This GNIRS-specific primitive sets debug_min_lines, order, min_snr,
        num_lines and average values depending on the
        observing mode, as the default value for these parameters is None.
        It then calls the generic version of the primitive.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
             Mosaicked Arc data as 2D spectral images or 1D spectra.

        suffix : str/None
            Suffix to be added to output files

        order : int
            Order of Chebyshev fitting function.

        center : None or int
            Central row/column for 1D extraction (None => use middle).

        nsum : int, optional
            Number of rows/columns to average.

        min_snr : float
            Minimum S/N ratio in line peak to be used in fitting.

        weighting : {'natural', 'relative', 'none'}
            How to weight the detected peaks.

        fwidth : float/None
            Expected width of arc lines in pixels. It tells how far the
            KDTreeFitter should look for when matching detected peaks with
            reference arcs lines. If None, `fwidth` is determined using
            `peak_finding.estimate_peak_width`.

        min_sep : float
            Minimum separation (in pixels) for peaks to be considered distinct

        central_wavelength : float/None
            central wavelength in nm (if None, use the WCS or descriptor)

        dispersion : float/None
            dispersion in nm/pixel (if None, use the WCS or descriptor)

        linelist : str/None
            Name of file containing arc lines. If None, then a default look-up
            table will be used.

        alternative_centers : bool
            Identify alternative central wavelengths and try to fit them?

        nbright : int (or may not exist in certain class methods)
            Number of brightest lines to cull before fitting

        absorption : bool
            If feature type is absorption (default: "False")

        interactive : bool
            Use the interactive tool?

        debug : bool
            Enable plots for debugging.

        num_lines: int/None
            Number of lines with largest weigths (within a wvl bin) to be used for
            the generated line list.

        wv_band: {'20', '50', '80', '100', 'header'}
            Water vapour content (as percentile) to be used for ATRAN model
            selection. If "header", then the value from the header is used.

        resolution: int/None
            Resolution of the observation (as l/dl), to which ATRAN spectrum should be
            convolved. If None, the default value for the instrument/mode is used.

        debug_combiner: {"mean", "median", "none"}
            Method to use for combining rows/columns when extracting 1D-spectrum.
            Default: "mean".

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Updated objects with a `.WAVECAL` attribute and improved wcs for
            each slice

        See Also
        --------
        :class:`~geminidr.core.primitives_visualize.Visualize.mosaicDetectors`,
        :class:`~gempy.library.matching.KDTreeFitter`,
        """
        adoutputs = []
        for ad in adinputs:
            these_params = params.copy()
            min_snr_isNone = True if these_params["min_snr"] is None else False
            order_isNone = True if these_params["order"] is None else False

            grating = ad._grating(pretty=True, stripID=True)
            cenwave = ad.central_wavelength(asMicrometers=True)
            log = self.log
            if 'ARC' not in ad.tags:
                if these_params["absorption"]:
                    # The case of wavecal from absorption using ATRAN lines
                    self.generated_linelist = "atran"
                    if these_params["order"] is None:
                        these_params["order"] = 1
                    if these_params["min_snr"] is None:
                        these_params["min_snr"] = 1
                else:
                    # Airglow emission
                    self.generated_linelist = "airglow"
            else:
                if 'Long' in ad.camera() and grating == "111/mm":
                    if these_params["order"] is None:
                        these_params["order"] = 1
                    if these_params["min_snr"] is None:
                        these_params["min_snr"] = 10



            if these_params["min_snr"] is None:
                these_params["min_snr"] = 20
            if these_params["order"] is None:
                these_params["order"] = 3

            if min_snr_isNone:
                self.log.stdinfo(f'Parameter "min_snr" is set to None. '
                                 f'Using min_snr={these_params["min_snr"]} for {ad.filename}')
            if order_isNone:
                self.log.stdinfo(f'Parameter "order" is set to None. '
                                 f'Using order={these_params["order"]} for {ad.filename}')

            adoutputs.extend(super().determineWavelengthSolution([ad], **these_params))
        return adoutputs

    def maskBeyondRegions(self, adinputs=None, **params):
        """
        suffix
        regions3   to keep
        regions4
        regions5
        regions6
        regions7
        regions8
        aperture default 1
        """

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params['suffix']
        aperture = params['aperture']

        adoutputs = []
        for ad in adinputs:
            aperture_extindex = []
            for i, ext in enumerate(ad):
                if ext.hdr.get('APERTURE') == aperture:
                    aperture_extindex.append((i, ext.hdr.get('SPECORDR')))

            for index, order in aperture_extindex:
                regions = at.parse_user_regions(params[f'regions{order}'], dtype=float)
                if regions is None:
                    log.warning(f"No regions provided for order {order}, skipping")
                    continue
                try:
                    ext = ad[index]
                except IndexError:
                    log.warning(f"Order {order} not found in {ad.filename}, skipping")
                    continue

                waves = ext.wcs(np.arange(ext.data.size))
                mask = at.create_mask_from_regions(waves, regions=regions)
                # regions defines what to keep.
                # mask is False for pixels to keep. True for pixels to mask.
                ext.mask[mask] |= DQ.no_data

                if params[f'regions{order}'] is not None:
                    log.stdinfo(f"Masked pixels outside '{params[f'regions{order}']}' nm "
                                f"in order {order} of aperture {aperture} of {ad.filename}")
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)
            adoutputs.append(ad)

        return adoutputs

    def thresholdFlatfield(self, adinputs=None, **params):
        """
        This primitive sets the DQ '64' bit (unilluminated) for any pixels
        which have a value <lower or >upper in the SCI plane.
        it also sets the science plane pixel value to 1.0 for pixels which are bad
        and very close to zero, to avoid divide by zero issues and inf values
        in the flat-fielded science data.

        The GNIRS XD version of this primitive additionally tidies the mask by
        masking small islands of good rows and extending large masked regions
        all the way to the edge of the spectrum. This avoids strange situations
        where low-brightness regions in the flatfield can leave illuminated
        regions at either end of the spectrum where primitives like
        determineWavelengthSolution or determinePinholeRectification will try
        to identify features that will result in a poor fit.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        lower: float
            value below which DQ pixels should be set to unilluminated
        upper: float
            value above which DQ pixels should be set to unilluminated
        debug_min_unmasked: int
            contiguous groups of unmasked rows smaller than this will be masked
        debug_min_masked: int
            contiguous groups of masked rows of this size or larger will be
            extended to the end of the spectrum
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        adinputs = super().thresholdFlatfield(adinputs=adinputs, **params)

        min_unmasked = params.pop('debug_min_unmasked', 0)
        min_masked = params.pop('debug_min_masked', np.inf)

        for ad in adinputs:
            for ext in ad:
                if ext.mask is None:
                    log.debug("Skipping {}:{} with no DQ plane".format(ad.filename, ext.id))
                    continue

                dispaxis = 2 - ext.dispersion_axis()  # python sense
                illum_mask = np.bitwise_and.reduce(ext.mask & DQ.unilluminated,
                                                   axis=1-dispaxis)
                tmp_array = np.ma.masked_array(illum_mask,
                                               mask=illum_mask.astype(bool))
                unmasked_slices = np.ma.clump_unmasked(tmp_array)
                for _slice in unmasked_slices:
                    if _slice.stop - _slice.start < min_unmasked:
                        # Small island of good rows, mask them
                        illum_mask[_slice] |= DQ.unilluminated
                        log.debug(f"Masking rows {list(range(_slice.start, _slice.stop))} "
                                  f"in {ad.filename}:{ext.id} as they are a small island")

                # tmp_array has been updated because its mask is a reference
                # to illum_mask
                masked_slices = np.ma.clump_masked(tmp_array)
                masked_slices = [s for s in masked_slices if s.stop - s.start >= min_masked]
                halfway = illum_mask.size // 2
                for _slice in masked_slices:
                    if _slice.stop < halfway:
                        illum_mask[:_slice.stop] |= DQ.unilluminated
                        log.debug("Masking to row {} in {}:{}".format(
                            _slice.stop, ad.filename, ext.id))
                    elif _slice.start >= halfway:
                        illum_mask[_slice.start:] |= DQ.unilluminated
                        log.debug("Masking from row {} in {}:{}".format(
                            _slice.start, ad.filename, ext.id))

                if dispaxis == 0:
                    ext.mask |= illum_mask[:, np.newaxis]
                else:
                    ext.mask |= illum_mask

                # Reset newly-masked pixels to 1.0
                ext.data[(ext.mask & DQ.unilluminated) > 0] = 1.0

        # Everything was timestamped and filename-updated in the parent primitive
        return adinputs