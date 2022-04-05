#
#                                                                 gemini_python
#
#                                                     primitives_gnirs_spect.py
# -----------------------------------------------------------------------------


import astrodata
import gemini_instruments
from gempy.gemini import gemini_tools as gt

from geminidr.gemini.lookups import DQ_definitions as DQ

from .primitives_gnirs import GNIRS
from ..core import NearIR, Spect
from . import parameters_gnirs_longslit

from recipe_system.utils.decorators import (parameter_override,
                                            capture_provenance)

# -----------------------------------------------------------------------------
@parameter_override
@capture_provenance
class GNIRSLongslit(GNIRS, Spect, NearIR):
    """
    This class contains all of the preprocessing primitives for the
    GNIRSLongslit level of the type hierarchy tree. It inherits all the
    primitives from the above level.
    """
    tagset = {'GEMINI', 'GNIRS', 'SPECT'}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_gnirs_longslit)

    def addIllumMaskToDQ(self, adinputs=None, **params):

        pass

    def addMDF(self, adinputs=None, suffix=None, mdf=None):
        """This GNIRS-specific implementation of addMDF() corrects for the fact
        that GNIRS MDFs have dimensions in arcseconds, instead of in millimeters
        like other instruments. It calls primitives_gemini.addMDF() to attach
        the MDFs, then performs two calculations on the "slitsize_mx" and
        "slitsize_my" fields. First it multiplies by 0.96 ("slitsize_mx" only)
        to correct for the fact that the GNIRS slit width given in the MDFs is
        slightly too large, and then multiplies by 1.61144 to convert from
        arcsec to millimeters (for f/16 on an 8m telescope).

        Any parameters given will be passed to primitives_gemini.addMDF().

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

        if not isinstance(adinputs, list):
            adinputs = list(adinputs)

        # Delegate up to primitives_gemini.addMDF() to attach the MDFs.
        adinputs = super().addMDF(adinputs=adinputs, suffix=suffix, mdf=mdf)

        for ad in adinputs:

            try:
                mdf = ad.MDF
            except AttributeError:
                log.warning(f"MDF not found for {ad.filename}, continuing.")
                continue

            # This is an empirically-determined correction factor for the fact
            # that the slitsize in the MDFS appears to be slightly larger than
            # in reality.
            if 'Short' in ad.camera():
                slit_correction_factor = 0.96
            elif 'Long' in ad.camera():
                slit_correction_factor = 0.97

            # This is the conversion factor from arcseconds to millimeters of
            # slit width for f/16 on an 8m telescope.
            arcsec_to_mm = 1.61144

            # The MDFs for Gemini-South data report a slitwidth of 100",
            # but from measuring slitwidths in observations it appears that the
            # 103" used in Gemini-North MDFs is actually the correct value, so
            # add 3" here.
            if (ad.telescope() == 'Gemini-South') and ('Short' in ad.camera()):
                ad.MDF['slitsize_mx'][0] += 3

            # Flats from the LongBlue and LongRed cameras appear to have
            # physically different slit widths, despite both drawing from the
            # same MDF. LongRed data appears to be 3" narrower than LongBlue,
            # so subtract that here.
            if ('LongRed' in ad.camera()):
                ad.MDF['slitsize_mx'][0] -= 3

            # Only the 'slitsize_mx' value needs the width correction; the
            # 'slitsize_my' isn't actually used, but we convert it for
            # consistency.
            mdf['slitsize_mx'][0] *= slit_correction_factor / arcsec_to_mm
            mdf['slitsize_my'][0] *= arcsec_to_mm

            log.stdinfo('Converted slit sizes from arcseconds to millimeters '
                        f'in {ad.filename}.')
