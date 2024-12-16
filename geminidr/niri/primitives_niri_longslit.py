#
#                                                                 gemini_python
#
#                                                      primitives_niri_spect.py
# -----------------------------------------------------------------------------

from astropy.table import Table
import numpy as np

from gempy.gemini import gemini_tools as gt
from recipe_system.utils.decorators import (parameter_override,
                                            capture_provenance)

from .primitives_niri_spect import NIRISpect
from geminidr.core.primitives_longslit import Longslit
from . import parameters_niri_longslit
from .lookups.MDF_LS import slit_info

# -----------------------------------------------------------------------------

@parameter_override
@capture_provenance
class NIRILongslit(NIRISpect, Longslit):
    """
    This class contains all the preprocessing primitives for the NIRILongslit
    level of the type hierarchy tree. It inherits all the primitives from the
    above level.
    """
    tagset = {"GEMINI", "NIRI", "SPECT", "LS"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_niri_longslit)

    def _fields_overlap(self, ad1, ad2, frac_FOV=1.0,
                        max_perpendicular_offset=None):
        slit_length = ad1.MDF['slitlength_arcsec'][0]
        slit_width = ad1.slit_width()
        return super()._fields_overlap(
            ad1, ad2, frac_FOV=frac_FOV,
            slit_length=slit_length,
            slit_width=slit_width,
            max_perpendicular_offset=max_perpendicular_offset)

    def addMDF(self, adinputs=None, suffix=None, mdf=None):
        """
        This NIRI-specific version adds an MDF table containing the expected
        midpoint (in pixels) of the slit, and the length of the slit in arcsec
        and pixels.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        mdf: str/None
            name of MDF to add (None => use default)
        """

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        for ad in adinputs:
            log.stdinfo(f"Creating MDF table for NIRI file {ad.filename}.")

            maskname = ad.focal_plane_mask(pretty=True)
            fratio = ad.phu.get('BEAMSPLT')
            if fratio == 'f32' and maskname == 'f6-2pix':
                # The f/32-10 pixel configuation actually uses the same
                # slit as f6-2pix but with the f/32 camera, so just modify
                # the maskname here.
                maskname = 'f32-10pix'

            y_ccd, length_pix = slit_info[maskname]
            mdf_table = Table([[511.5], [y_ccd], [length_pix * ad.pixel_scale()], [length_pix]],
                              names=('x_ccd', 'y_ccd', 'slitlength_arcsec', 'slitlength_pixels'))
            ad.MDF = mdf_table

            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs
