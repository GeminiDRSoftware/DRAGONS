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
from . import parameters_niri_longslit

# -----------------------------------------------------------------------------

# The NIRI webpage lists "f/32 4-pixel" and "f/32 7-pixel" options, but the
# archive only has options for "f32-6pix" and "f32-9pix". The component numbers
# make it clear that these are the same as the 4 and 7 pixels configurations
# listed.
niri_slit_info = {'f6-2pix': (501, 51.4, 439.),
                  'f6-2pixBl': (499, 51.6, 442),
                  'f6-4pix': (498, 112.5, 962),
                  'f6-4pixBl': (494, 51.5, 440),
                  'f6-6pix': (505, 110, 965.81),
                  'f6-6pixBl': (496.5, 51.5, 439.),
                  # The nominal slit length for the f/32 camera is 22", but
                  # the illuminated region goes off the edge and is longer
                  # than the nominal 1,000 pixels in each case, so 22.2" is a
                  # minimum guess, assuming the pixel scale is correct.
                  'f32-6pix': (506, 22.2, 1015),
                  'f32-9pix': (508, 22.2, 1015),
                  'f32-10pix': (507, 22.2, 1015)}

@parameter_override
@capture_provenance
class NIRILongslit(NIRISpect):
    """
    This class contains all the preprocessing primitives for the NIRILongslit
    level of the type hierarchy tree. It inherits all the primitives from the
    above level.
    """
    tagset = {"GEMINI", "NIRI", "SPECT", "LS"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_niri_longslit)

    def addMDF(self, adinputs=None, suffix=None, mdf=None):
        """


        Parameters
        ----------
        adinputs : TYPE, optional
            DESCRIPTION. The default is None.
        suffix : TYPE, optional
            DESCRIPTION. The default is None.
        mdf : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        mdf_list = mdf or self.caldb.get_calibrations(adinputs,
                                                      caltype="mask").files

        for ad, mdf in zip(*gt.make_lists(adinputs, mdf_list, force_ad=True)):

            self._addMDF(ad, suffix, mdf)

            if hasattr(ad, 'MDF'):
                continue
            else:
                log.stdinfo(f"Creating MDF table for NIRI file {ad.filename}.")

                maskname = ad.focal_plane_mask(pretty=True)
                fratio = ad.phu.get('BEAMSPLT')

                if fratio == 'f32' and maskname == 'f6-2pix':
                    # This configuation actually uses the same slit as f6-2pix
                    # but with the f/32 camera, so just modify the maskname.
                    maskname = 'f32-10pix'

                mdf_table = Table(np.array(niri_slit_info[maskname]),
                                  names=('y_ccd', 'slitlength_arcsec',
                                  'slitlength_pixels'))
                ad.MDF = mdf_table
