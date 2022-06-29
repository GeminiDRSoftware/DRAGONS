#
#                                                                 gemini_python
#
#                                                     primitives_gnirs_spect.py
# -----------------------------------------------------------------------------


from astropy.table import Table, hstack
import numpy as np

from gempy.gemini import gemini_tools as gt
from recipe_system.utils.decorators import (parameter_override,
                                            capture_provenance)

from .primitives_gnirs_spect import GNIRSSpect
from . import parameters_gnirs_longslit

# -----------------------------------------------------------------------------
@parameter_override
@capture_provenance
class GNIRSLongslit(GNIRSSpect):
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
        slit_length = ad1.MDF['slitlength_arcsec']
        slit_width = float(ad1.slit()[:4])
        return super()._fields_overlap(
            ad1, ad2, frac_FOV=frac_FOV,
            slit_length=slit_length,
            slit_width=slit_width,
            max_perpendicular_offset=max_perpendicular_offset)

    def addIllumMaskToDQ(self, adinputs=None, **params):

        pass

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

        # "Slit" indicates a change in the expected length of the slit, "x"
        # denotes a shift in the expected midpoint of the illuminated region.
        corrections = {'slit_short_south': 103,  # arcsec
                       'x_32/mm_south': 406,  # pixels
                       'slit_long_north': 49,  # arcsec
                       'x_longred_north': 482,  # pixels
                       'x_longblue_north': 542,  # pixels
                       }

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        mdf_list = mdf or self.caldb.get_calibrations(adinputs,
                                                      caltype="mask").files

        # This is the conversion factor from arcseconds to millimeters of
        # slit width for f/16 on an 8m telescope.
        arcsec_to_mm = 1.61144

        for ad, mdf in zip(*gt.make_lists(adinputs, mdf_list, force_ad=True)):

            self._addMDF(ad, suffix, mdf)

            try:
                mdf = ad.MDF
            except AttributeError:
                log.warning(f"MDF not found for {ad.filename}, continuing.")
                continue

            # TODO: fix GNIRS WCS handling
            # At some point, when the WCS for GNIRS is fixed, this code
            # block can be removed. This is an empirically-determined
            # correction factor for the fact that the pixel scale is a few
            # percent different from the nominal value.
            if 'Short' in ad.camera():
                slit_correction_factor = 0.96
            elif 'Long' in ad.camera():
                slit_correction_factor = 0.97

            # The MDFs for GNIRS are sometimes incorrect, so apply the various
            # corrections given above as appropriate.
            if (ad.telescope() == 'Gemini-South'):
                if ('Short' in ad.camera()):
                    mdf['slitsize_mx'][0] = corrections['slit_short_south']

                if ('Long' in ad.camera()) and ('32/mm' in ad.disperser()):
                    mdf['x_ccd'][0] = corrections['x_32/mm_south']

            elif (ad.telescope() == 'Gemini-North'):

                if ('LongRed' in ad.camera()):
                    mdf['x_ccd'][0] = corrections['x_longred_north']
                    if ('111/mm' in ad.disperser()):
                        mdf['slitsize_mx'][0] = corrections['slit_long_north']

                if ('LongBlue' in ad.camera()):
                    mdf['x_ccd'][0] = corrections['x_longblue_north']


            # For GNIRS, the 'slitsize_mx' column is in arcsec, so grab it:
            arcsec = mdf['slitsize_mx'][0] * slit_correction_factor
            pixels = arcsec / ad.pixel_scale()

            # Only the 'slitsize_mx' value needs the width correction; the
            # 'slitsize_my' isn't actually used, but we convert it for
            # consistency.
            mdf['slitsize_mx'][0] *= slit_correction_factor / arcsec_to_mm
            mdf['slitsize_my'][0] /= arcsec_to_mm

            extra_cols = Table(np.array([arcsec, pixels]),
                               names=('slitlength_arcsec',
                                      'slitlength_pixels'))
            ad.MDF = hstack([mdf, extra_cols], join_type='inner')

            log.stdinfo('Converted slit sizes from arcseconds to millimeters '
                        f'in {ad.filename}.')
