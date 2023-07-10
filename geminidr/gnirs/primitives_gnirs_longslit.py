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

from .primitives_gnirs_spect import GNIRSSpect
from geminidr.core.primitives_longslit import Longslit
from . import parameters_gnirs_longslit
from .lookups.MDF_LS_GNIRS import slit_info

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

    def addIllumMaskToDQ(self, adinputs=None, suffix=None, illum_mask=None):
        """
        Adds an illumination mask to each AD object. The default illumination mask
        masks off extra orders and/or unilluminated areas outside the order blocking filter range.

        Parameters
        ----------
        suffix : str
            suffix to be added to output files
        illum_mask : str/None
            name of illumination mask mask (None -> use default)

        """
        # Cut-on and cut-off wavelengths (um) of GNIRS order-blocking filters, based on conservative transmissivity (1%),
        # or inter-order minima.
        bl_filter_range_dict = {'X': (1.01, 1.19),
                                'J': (1.15, 1.385),
                                'H': (1.46, 1.84),
                                'K': (1.89, 2.54),
                                'L': (2.77, 4.44),
                                'M': (4.2, 6.0)}
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

            else:
                dispaxis = 2 - ad[0].dispersion_axis()
                dispaxis_center = ad[0].shape[dispaxis] // 2
                cenwave = ad.central_wavelength(asMicrometers=True)
                dispersion = ad.dispersion(asMicrometers=True)[0]
                filter = ad.filter_name(pretty=True)
                filter_cuton_wvl = bl_filter_range_dict[filter][0]
                filter_cutoff_wvl = bl_filter_range_dict[filter][1]
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


    def addMDF(self, adinputs=None, suffix=None, mdf=None):
        """
        This GNIRS-specific implementation of addMDF() corrects for various
        instances of the GNIRS MDFs not corresponding to reality. It calls
        primitives_gemini._addMDF() on each astrodata object to attach the MDFs,
        then performs corrections depending on the data. It also attaches two
        columns, 'slitlength_arcsec' and 'slitlength_pixels' with the length of
        the slit in arcseconds and pixels, respectively.

        Any parameters given will be passed to primitives_gemini._addMDF().

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        mdf: str/None
            name of MDF to add (None => use default)
        """

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        mdf_list = mdf or self.caldb.get_calibrations(adinputs,
                                                      caltype="mask").files

        # This is the conversion factor from arcseconds to millimeters of
        # slit width for f/16 on an 8m telescope.
        # arcsec_to_mm = 1.61144

        for ad, mdf in zip(*gt.make_lists(adinputs, mdf_list, force_ad=True)):

            # GNIRS LS doesn't use mask definition files, so this won't add
            # anything, but it will check if the file already has an MDF table.
            self._addMDF(ad, suffix, mdf)

            if hasattr(ad, 'MDF'):
                log.fullinfo(f"{ad.filename} already has an MDF table.")
                continue
            else:
                telescope = ad.telescope().split('-')[1] # 'North' or 'South'
                grating = ad._grating(pretty=True)
                camera = ad.camera(pretty=True)
                mdf_key = "_".join((telescope, grating, camera))

                mdf_table = Table(np.array(slit_info[mdf_key]),
                                  names=('x_ccd', 'slitlength_arcsec',
                                         'slitlength_pixels'))
                ad.MDF = mdf_table
                log.stdinfo(f"Added MDF table for {ad.filename}")

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
        for ad in adinputs:
            disp = ad.disperser(pretty=True)
            cam = ad.camera(pretty=True)
            cenwave = ad.central_wavelength(asMicrometers=True)
            if params["spectral_order"] is None:
                if disp.startswith('111') and cam.startswith('Long') and \
                        cenwave >= 1.65:
                        params["spectral_order"] = 1
                else:
                    params["spectral_order"] = 2
                self.log.stdinfo(f'Parameter "spectral_order" is set to None. '
                                 f'Using spectral_order={params["spectral_order"]}')

            if params["min_line_length"] is None:
                if cam.startswith('Long'):
                    params["min_line_length"] = 0.8
                else:
                    params["min_line_length"] = 0.6
                self.log.stdinfo(f'Parameter "min_line_length" is set to None. '
                 f'Using min_line_length={params["min_line_length"]}')
        adinputs = super().determineDistortion(adinputs, **params)
        return adinputs
