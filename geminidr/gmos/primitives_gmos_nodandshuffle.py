#
#                                                                  gemini_python
#
#                                                primtives_gmos_nodandshuffle.py
#
# NB This is a pure mixin and should not be instantiated as a primitives class!
# ------------------------------------------------------------------------------
from copy import deepcopy

from gempy.gemini import gemini_tools as gt

from .primitives_gmos import GMOS
from . import parameters_gmos_nodandshuffle

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class GMOSNodAndShuffle(GMOS):
    """
    This is the class containing all of the preprocessing primitives
    for the GMOSImage level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set()

    def __init__(self, adinputs, **kwargs):
        super().__init__(adinputs, **kwargs)
        self._param_update(parameters_gmos_nodandshuffle)

    def skyCorrectNodAndShuffle(self, adinputs=None, suffix=None):
        """
        Perform sky correction on GMOS N&S images bytaking each image and
        subtracting from it a shifted version of the same image.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        for ad in adinputs:
            # Check whether the myScienceStep primitive has been run previously
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by skyCorrectNodShuffle".
                            format(ad.filename))
                continue

            # Determine N&S offset in (binned) pixels
            shuffle = ad.shuffle_pixels() // ad.detector_y_bin()
            a_nod_count, b_nod_count = ad.nod_count()

            ad_nodded = deepcopy(ad)

            # Shuffle B position data up for all extensions (SCI, DQ, VAR)
            for ext, ext_nodded in zip(ad, ad_nodded):
                #TODO: Add DQ=16 to top and bottom?
                # Set image initially to zero
                ext_nodded.multiply(0)
                # Then replace with the upward-shifted data
                for attr in ('data', 'mask', 'variance'):
                    getattr(ext_nodded, attr)[shuffle:] = getattr(ext,
                                                        attr)[:-shuffle]

            # Normalize if the A and B nod counts differ
            if a_nod_count != b_nod_count:
                log.stdinfo("{} A and B nod counts differ...normalizing".
                            format(ad.filename))
                ad.multiply(0.5 * (a_nod_count + b_nod_count) / a_nod_count)
                ad_nodded.multiply(0.5 * (a_nod_count + b_nod_count) / b_nod_count)

            # Subtract nodded image from image to complete the process
            ad.subtract(ad_nodded)

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs
