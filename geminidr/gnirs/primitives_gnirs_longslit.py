#
#                                                                 gemini_python
#
#                                                     primitives_gnirs_spect.py
# -----------------------------------------------------------------------------


from astropy.modeling import models
from astropy.table import Table
import numpy as np

from gempy.gemini import gemini_tools as gt
from recipe_system.utils.decorators import (parameter_override,
                                            capture_provenance)
from geminidr.gemini.lookups import DQ_definitions as DQ
from geminidr.gnirs.lookups.maskdb import bl_filter_range_dict

from .primitives_gnirs_spect import GNIRSSpect
from geminidr.core.primitives_longslit import Longslit
from . import parameters_gnirs_longslit
from .lookups.MDF_LS import slit_info

# -----------------------------------------------------------------------------
@parameter_override
@capture_provenance
class GNIRSLongslit(GNIRSSpect, Longslit):
    """
    This class contains all of the preprocessing primitives for the
    GNIRSLongslit level of the type hierarchy tree. It inherits all the
    primitives from the above level.
    """
    tagset = {"GEMINI", "GNIRS", "SPECT", "LS"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_gnirs_longslit)

    def _fields_overlap(self, ad1, ad2, frac_FOV=1.0,
                        max_perpendicular_offset=None):
        slit_length = ad1.MDF['slitlength_arcsec'][0]
        slit_width = ad1.slit_width()
        return super()._fields_overlap(
            ad1, ad2, frac_FOV=frac_FOV,
            slit_length=slit_length,
            slit_width=slit_width,
            max_perpendicular_offset=max_perpendicular_offset)

    def addIllumMaskToDQ(self, adinputs=None, suffix=None, illum_mask=None,
                         keep_second_order=False):
        """
        Adds an illumination mask to each AD object. The default illumination mask
        masks off extra orders and/or unilluminated areas outside the order blocking filter range.

        Parameters
        ----------
        suffix : str
            suffix to be added to output files
        illum_mask : str/None
            name of illumination mask mask (None -> use default)
        keep_second_order : bool
            don't apply second order mask? (default is False)

        """

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        for ad, illum in zip(*gt.make_lists(adinputs, illum_mask, force_ad=True)):
            if ad.phu.get(timestamp_key):
                log.warning('No changes will be made to {}, since it has '
                    'already been processed by addIllumMaskToDQ'.
                            format(ad.filename))
                continue
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

            elif keep_second_order is False:
                dispaxis = 2 - ad[0].dispersion_axis()
                dispaxis_center = ad[0].shape[dispaxis] // 2
                cenwave = ad.central_wavelength(asMicrometers=True)
                dispersion = ad.dispersion(asMicrometers=True)[0]
                filter = ad.filter_name(keepID=True)
                try:
                    filter_cuton_wvl = bl_filter_range_dict[filter][0]
                    filter_cutoff_wvl = bl_filter_range_dict[filter][1]
                except KeyError:
                    log.warning("Unknown illumination mask for the filter {} for {}".
                                format(filter, ad.filename))
                    break
                else:
                    filter_cuton_pix = min(int(dispaxis_center - (cenwave - filter_cuton_wvl) / dispersion), ad[0].shape[dispaxis] - 1)
                    filter_cutoff_pix = max(int(dispaxis_center + (filter_cutoff_wvl - cenwave) / dispersion), 0)

                for ext in ad:
                    ext.mask[:filter_cutoff_pix] |= DQ.unilluminated
                    ext.mask[filter_cuton_pix:] |= DQ.unilluminated
                if filter_cutoff_pix > 0:
                    log.stdinfo(f"Masking rows 1 to {filter_cutoff_pix+1}")
                if filter_cuton_pix < (ad[0].shape[dispaxis] - 1):
                    log.stdinfo(f"Masking rows {filter_cuton_pix+1} to {(ad[0].shape[dispaxis])}")
                # Mask out vignetting in the lower-left corner found in GNIRS
                # on Gemini-North. It's only really visible in LongRed camera
                # data, but no harm adding it to all data for correctness.
                if 'North' in ad.telescope():
                    log.fullinfo("Masking vignetting")
                    width = ext.data.shape[0 - dispaxis]
                    height = ext.data.shape[1 - dispaxis]
                    x, y = np.mgrid[0:width, 0:height]
                    # Numbers taken from model of on-detector edge in vignetted
                    # data, since the vignetting happens close enough to the
                    # side of the detector to not really be traceable. It's
                    # perhaps not pixel-perfect, but it looks reasonable and
                    # should err on the side of caution.
                    model = models.Chebyshev1D(1, c0=-1.09277804, c1=-7.2085752,
                                               domain=(0, 1023))
                    vignette_mask = y < model(x)
                    ext.mask |= vignette_mask * DQ.unilluminated

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs


    def addMDF(self, adinputs=None, suffix=None):
        """
        This GNIRS-specific implementation of addMDF() corrects for various
        instances of the GNIRS MDFs not corresponding to reality. It calls
        primitives_gemini._addMDF() on each astrodata object to attach the MDFs,
        then performs corrections depending on the data. It also attaches two
        columns, 'slitlength_arcsec' and 'slitlength_pixels' with the length of
        the slit in arcseconds and pixels, respectively.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        for ad in adinputs:
            telescope = ad.telescope().split('-')[1] # 'North' or 'South'
            grating = ad._grating(pretty=True)
            camera = ad.camera(pretty=True)
            mdf_key = "_".join((telescope, grating, camera))
            x_ccd, length_pix = slit_info[mdf_key]

            mdf_table = Table([[x_ccd], [511.5],
                               [length_pix*ad.pixel_scale()], [length_pix]],
                              names=('x_ccd', 'y_ccd',
                                     'slitlength_arcsec', 'slitlength_pixels'))
            ad.MDF = mdf_table
            log.fullinfo(f"Added MDF table for {ad.filename}")

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)

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

        This GNIRS-specific primitive sets default spectral order in case it's None
        (since there are only few lines available in H and K-bands in high-res mode, which
        requires setting order to 1), and minimum length of traced feature to be considered
        as a useful line for each pixel scale.
        It then calls the generic version of the primitive.


        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
            Arc data as 2D spectral images with the distortion and wavelength
            solutions encoded in the WCS.

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
            Reject lines with suspiciously high SNR (e.g. bad columns)? (Default: True)

        debug: bool
            plot arc line traces on image display window?

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            The same input list is used as output but each object now has the
            appropriate `nddata.wcs` defined for each of its extensions. This
            provides details of the 2D Chebyshev fit which maps the distortion.
        """
        adoutputs = []
        for ad in adinputs:
            these_params = params.copy()
            disp = ad.disperser(pretty=True)
            cam = ad.camera(pretty=True)
            cenwave = ad.central_wavelength(asMicrometers=True)
            if these_params["spectral_order"] is None:
                if 'ARC' in ad.tags:
                    if disp.startswith('111') and cam.startswith('Long') and \
                            cenwave >= 1.65:
                            these_params["spectral_order"] = 1
                    else:
                        these_params["spectral_order"] = 2
                else:
                # sky line case
                    these_params["spectral_order"] = 3
                self.log.stdinfo(f'Parameter "spectral_order" is set to None. '
                                 f'Using spectral_order={these_params["spectral_order"]} for {ad.filename}')

            if these_params["min_line_length"] is None:
                if cam.startswith('Long'):
                    these_params["min_line_length"] = 0.8
                else:
                    these_params["min_line_length"] = 0.6
                self.log.stdinfo(f'Parameter "min_line_length" is set to None. '
                 f'Using min_line_length={these_params["min_line_length"]} for {ad.filename}')

            if these_params["max_missed"] is None:
                if "ARC" in ad.tags:
                    # In arcs with few lines tracing strong horizontal noise pattern can
                    # affect distortion model.Using a lower max_missed value helps to
                    # filter out horizontal noise.
                    these_params["max_missed"] = 2
                else:
                    # In science frames we want this parameter be set to a higher value, since
                    # otherwise the line might be abandoned when crossing a bright object spectrum.
                    these_params["max_missed"] = 5
                self.log.stdinfo(f'Parameter "max_missed" is set to None. '
                 f'Using max_missed={these_params["max_missed"]} for {ad.filename}')
            adoutputs.extend(super().determineDistortion([ad], **these_params))
        return adoutputs


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
            combine_method_isNone = True if these_params["combine_method"] == "optimal" else False

            disp = ad.disperser(pretty=True)
            filt = ad.filter_name(pretty=True)
            cam = ad.camera(pretty=True)
            cenwave = ad.central_wavelength(asMicrometers=True)
            log = self.log

            if 'ARC' in ad.tags:
                if these_params["min_snr"] is None:
                    these_params["min_snr"] = 20
                if these_params["debug_min_lines"] is None:
                    these_params["debug_min_lines"] = 100000

                if these_params["order"] is None:
                    if ((filt == "H" and cenwave >= 1.75) or (filt == "K" and cenwave >= 2.2)) \
                            and ((cam.startswith('Long') and disp.startswith('32')) or
                                 (cam.startswith('Short') and disp.startswith('111'))):
                            these_params["order"] = 1
                    elif disp.startswith('111') and cam.startswith('Long'):
                            these_params["order"] = 1
                    else:
                        these_params["order"] = 3

            elif these_params["absorption"] or ad.central_wavelength(asMicrometers=True) >= 2.8:
                # The case of wavecal from absorption, or wavecal from telluric
                # emission in L- and M-bands, both done using ATRAN lines
                self.generated_linelist = "atran"
                # sigma=2 works better with ATRAN line lists
                these_params["lsigma"] = 2
                these_params["hsigma"] = 2

                if these_params["absorption"]:
                    # Telluric absorption case
                    if these_params["order"] is None:
                        these_params["order"] = 1

                    if these_params["min_snr"] is None:
                        these_params["min_snr"] = 1

                    if these_params["center"] is None:
                        try:
                            aptable = ad[0].APERTURE
                            these_params["center"] = int(aptable['c0'].data[0])
                        except (AttributeError, KeyError):
                            log.error("Could not find aperture locations in "
                                        f"{ad.filename} - continuing")
                            continue

                else:
                    # Telluric emission in L and M-bands
                    if these_params["order"] is None:
                        these_params["order"] = 3

                    if these_params["center"] is None:
                        these_params["center"] = 650

                    if these_params["min_snr"] is None:
                        if filt.startswith('L'):
                            # Use a lower min_snr for the regions with large illumination gradient,
                            # and for the region of "comb"-like lines beyond 3.8 um
                            if (disp.startswith('111') and 3.50 <= cenwave) or \
                                (disp.startswith('111') and cam.startswith('Short') and 3.80 <= cenwave) or \
                                (disp.startswith('32') and cam.startswith('Long') and 3.65 <= cenwave):
                                these_params["min_snr"] = 1
                            else:
                                these_params["min_snr"] = 10
                        else:
                            these_params["min_snr"] = 10

                    if these_params["num_lines"] is None:
                        if filt.startswith('M'):
                            these_params["num_lines"] = 150
                        elif filt.startswith('L'):
                            these_params["num_lines"] = 100
                            if ((disp.startswith('111') and cam.startswith('Short')) or
                                (disp.startswith('32') and cam.startswith('Long'))) and \
                                    3.80 <= cenwave:
                                these_params["num_lines"] = 300

                    if these_params["combine_method"] == "optimal":
                        # this is to reduce the impact of hot pixels
                        if filt.startswith('L') and cenwave >= 3.8:
                            these_params["combine_method"] = "median"
                        else:
                            these_params["combine_method"] = "mean"
            else:
                # OH emission
                self.generated_linelist = "airglow"
                if these_params["min_snr"] is None:
                    these_params["min_snr"] = 10
                if these_params["order"] is None:
                    these_params["order"] = 3
                if these_params["center"] is None:
                    these_params["center"] = 650

            if these_params["debug_min_lines"] is None:
                these_params["debug_min_lines"] = 15
            if these_params["num_lines"] is None:
                these_params["num_lines"] = 50
            if these_params["combine_method"] == "optimal":
                these_params["combine_method"] = "mean"

            if min_snr_isNone:
                self.log.stdinfo(f'Parameter "min_snr" is set to None. '
                                 f'Using min_snr={these_params["min_snr"]} for {ad.filename}')
            if order_isNone:
                self.log.stdinfo(f'Parameter "order" is set to None. '
                                 f'Using order={these_params["order"]} for {ad.filename}')
            if combine_method_isNone:
                self.log.stdinfo(f'Parameter "combine_method" is set to "optimal"".'
                                 f' Using "combine_method"={these_params["combine_method"]} for {ad.filename}')
            adoutputs.extend(super().determineWavelengthSolution([ad], **these_params))
        return adoutputs
