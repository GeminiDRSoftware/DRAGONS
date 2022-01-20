#
#                                                                       DRAGONS
#
#                                                         primitives_igrins.py
# ------------------------------------------------------------------------------

from gempy.gemini import gemini_tools as gt

from geminidr.gemini.primitives_gemini import Gemini

from . import parameters_igrins

from .lookups import timestamp_keywords as igrins_stamps

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------

@parameter_override
class Igrins(Gemini):
    """
    This class inherits from the level above.  Any primitives specific
    to IGRINS can go here.
    """

    tagset = {"GEMINI", "IGRINS"}

    def __init__(self, adinputs, **kwargs):
        super(Igrins, self).__init__(adinputs, **kwargs)
        self._param_update(parameters_igrins)
        # Add IGRINS specific timestamp keywords
        self.timestamp_keys.update(igrins_stamps.timestamp_keys)

    def someStuff(self, adinputs=None, **params):
        """
        Write message to screen.  Test primitive.

        Parameters
        ----------
        adinputs
        params

        Returns
        -------

        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        for ad in adinputs:
            log.status('I see '+ad.filename)

            gt.mark_history(ad, primname=self.myself(), keyword="TEST")
            ad.update_filename(suffix=params['suffix'], strip=True)

        return adinputs


    @staticmethod
    def _has_valid_extensions(ad):
        """ Check that the AD has a valid number of extensions. """

        # this needs to be updated at appropriate.
        return len(ad) in [1]

