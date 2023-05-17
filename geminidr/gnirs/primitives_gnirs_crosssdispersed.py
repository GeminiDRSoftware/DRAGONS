#
#                                                                 gemini_python
#
#                                            primitives_gnirs_crossdispersed.py
# -----------------------------------------------------------------------------

from copy import deepcopy

from astropy.table import Table, hstack
import numpy as np

from gempy.gemini import gemini_tools as gt
from recipe_system.utils.decorators import (parameter_override,
                                            capture_provenance)
from geminidr.gemini.lookups import DQ_definitions as DQ

from .primitives_gnirs_spect import GNIRSSpect
from . import parameters_gnirs_crossdispersed
from .lookups.MDF.xd_MDF_table import mdf_table

# -----------------------------------------------------------------------------
@parameter_override
@capture_provenance
class GNIRSCrossDispersed(GNIRSSpect):
    """This class contains all of the preprocessing primitives for the
    GNIRSCrossDispersed level of the type hierarchy tree. It inherits all the
    primitives from the above level.
    """
    tagset = {"GEMINI", "GNIRS", "SPECT", "XD"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_gnirs_crossdispersed)

    # TODO: handle _fields_overlap()


    def addMDF(self, adinputs=None, suffix=None, mdf=None):
        """
        This GNIRS XD-specific implementation of addMDF() calls
        primitives_gemini._addMDF() on each astrodata object to attach the MDFs.
        It also attaches two columns, 'slitlength_arcsec' and 'slitlength_pixels',
        with the length of the slit in arcseconds and pixels, respectively.

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

        for ad, mdf in zip(*gt.make_lists(adinputs, mdf_list, force_ad=True)):

            # GNIRS XD doesn't use mask definition files, so this won't add
            # anything, but it will check if the file already has an MDF table.
            self._addMDF(ad, suffix, mdf)

            if hasattr(ad, 'MDF'):
                continue
            else:
                mdf_key_parts = ('telescope', '_prism', 'decker',
                                 '_grating', 'camera')
                mdf_key = "_".join(getattr(ad, desc)() for desc in mdf_key_parts)

                ad.MDF = mdf_table[mdf_key]
                log.stdinfo(f"Added MDF table for {ad.filename}")

        return adinputs
