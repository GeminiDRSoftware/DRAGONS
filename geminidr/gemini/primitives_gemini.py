#
#                                                                  gemini_python
#
#                                                           primitives_gemini.py
# ------------------------------------------------------------------------------
from gempy.gemini import gemini_tools as gt

from geminidr.core import Bookkeeping, CalibDB, Preprocess
from geminidr.core import Visualize, Standardize, Stack

from .primitives_qa import QA
from .parameters_gemini import ParametersGemini

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class Gemini(Standardize, Bookkeeping, Preprocess, Visualize, Stack, QA,
             CalibDB):
    """
    This is the class containing the generic Gemini primitives.

    """
    tagset = set(["GEMINI"])

    def __init__(self, adinputs, **kwargs):
        super(Gemini, self).__init__(adinputs, **kwargs)
        self.inst_lookups = 'geminidr.gemini.lookups'
        self.parameters = ParametersGemini

    def standardizeObservatoryHeaders(self, adinputs=None, **params):
        """
        This primitive is used to make the changes and additions to the
        keywords in the headers of Gemini data.

        Parameters
        ----------
        suffix: str
            suffix to be added to output files
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]

        for ad in adinputs:
            if ad.phu.get(timestamp_key):
                log.warning("No changes will be made to {}, since it has "
                            "already been processed by standardize"
                            "ObservatoryHeaders".format(ad.filename))
                continue

            # Update various header keywords
            log.status("Updating keywords that are common to all Gemini data")
            ad.phu.set('NSCIEXT', len(ad), self.keyword_comments['NSCIEXT'])
            ad.hdr.set('BUNIT', 'adu', self.keyword_comments['BUNIT'])

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=params["suffix"], strip=True)
        return adinputs
