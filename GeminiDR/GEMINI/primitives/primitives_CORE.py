# Demo prototype primitive sets.

from inspect import stack

from astrodata.utils import logutils
from gempy.gemini import gemini_tools as gt

# new system imports - 10-06-2016 kra
from GEMINI.lookups import calurl_dict
from GEMINI.lookups import keyword_comments
from GEMINI.lookups import timestamp_keywords
from GEMINI.lookups.source_detection import sextractor_default_dict

from GEMINI.parameters.parameters_CORE import ParametersCORE

# ------------------------------------------------------------------------------
DENT = 3
# ------------------------------------------------------------------------------
class PrimitivesCORE(object):
    """
    This is the class containing all of the primitives for the GENERAL level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'PrimitiveSet'.
    """
    astrotype = "CORE"
    
    def __init__(self, adinputs):
        self.timestamp_keys   = timestamp_keywords.timestamp_keys
        self.keyword_comments = keyword_comments.keyword_comments
        self.sx_default_dict  = sextractor_default_dict.sextractor_default_dict
        self.calurl_dict      = calurl_dict.calurl_dict
        self.parameters       = ParametersCORE
        self.adinputs         = adinputs
        self.adoutputs        = None
        # This lambda will return the name of the current caller.
        self.myself           = lambda: stack()[1][3]
    
    def _override_pars(self, params):
        if params:
            for key, val in params.items():
                self.parameters[key] = val

        return

    def _primitive_exec(self, pname, parset=None, indent=0):
        """ 
        Reporting and logging for all.

        """
        pindent = DENT + indent
        log = logutils.get_logger(__name__)
        pmsg = "{}:{}".format("PRIMITIVE:", pname)
        sfx = ''
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        log.stdinfo("Parameters:\n{}".format(parset))
        if parset:
            try:
                sfx = parset["suffix"]
            except KeyError:
                pass

        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        return
