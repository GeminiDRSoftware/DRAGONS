#
#                                                                  gemini_python
#
#                                                      primitives_gnirs_image.py
# ------------------------------------------------------------------------------
import os

import astrodata, gemini_instruments
from geminidr.gemini.lookups import DQ_definitions as DQ
from gempy.gemini import gemini_tools as gt

from .primitives_gnirs import GNIRS
from ..core import Image, Photometry
from . import parameters_gnirs_image
from .lookups import FOV

from recipe_system.utils.decorators import parameter_override, capture_provenance


# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class GNIRSImage(GNIRS, Image, Photometry):
    """
    This is the class containing all of the preprocessing primitives
    for the GNIRSImage level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = {"GEMINI", "GNIRS", "IMAGE"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_gnirs_image)

    def addIllumMaskToDQ(self, adinputs=None, **params):
        """
        This primitive combines the illumination mask from the lookup directory
        into the DQ plane

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        # Since this primitive needs a reference, it must no-op without any
        if not adinputs:
            return adinputs

        # Get list of input and identify a suitable reference frame.
        # In most case it will be the first image, but for lamp-on, lamp-off
        # flats, one wants the reference frame to be a lamp-on since there's
        # next to no signal in the lamp-off.
        #
        # BEWARE: See note on the only GCAL_IR_OFF case below this block.
        lampons = self.selectFromInputs(adinputs, 'GCAL_IR_ON')
        reference = lampons[0] if lampons else adinputs[0]

        # When only a GCAL_IR_OFF is available:
        # To cover the one-at-a-time mode check for a compatible list
        # if list found, try to find a lamp-on in there to use as
        # reference for the mask and the shifts.
        # The mask's name and the shifts should stored in the headers
        # of the reference to simplify this and speed things up.
        #
        # NOT NEEDED NOW because we calling addIllumMask after the
        # lamp-offs have been subtracted.  But kept the idea here
        # in case we needed.

        # Fetching a corrected illumination mask with a keyhole that aligns
        # with the science data
        illum_mask = FOV._create_illum_mask(reference, log, xshift=params["xshift"],
                                            yshift=params["yshift"])
        if illum_mask is None:
            log.warning(f"No illumination mask found for {reference.filename},"
                        " no mask can be added to the DQ planes of the inputs")
            return adinputs

        illum_mask = illum_mask.astype(DQ.datatype) * DQ.unilluminated
        for ad in adinputs:
            # binary_OR the illumination mask or create a DQ plane from it.
            if ad[0].mask is None:
                ad[0].mask = illum_mask
            else:
                ad[0].mask |= illum_mask

            # Update the filename
            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs
