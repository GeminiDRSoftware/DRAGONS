"""
The geminidr package provides the base classes for all parameter and primitive
classes in the geminidr package.

E.g.,
>>> from geminidr import ParametersBASE
>>> from geminidr import PrimitivesBASE

"""
class ParametersBASE(object):
    """
    Base class for all Gemini package parameter sets.

    Most other parameter classes will be separate from their
    matching primitives class. Here, we incorporate the parameter
    class, ParametersBASE, into the mod.

    """
    pass
# ------------------------------------------------------------------------------

from inspect import stack
import os
import warnings
from astropy.io.fits.verify import VerifyWarning

from gempy.utils import logutils
# new system imports - 10-06-2016 kra
# NOTE: imports of these and other tables will be moving around ...
from .gemini.lookups import keyword_comments
from .gemini.lookups import timestamp_keywords
from .gemini.lookups.source_detection import sextractor_dict

from recipe_system.cal_service import calurl_dict
from recipe_system.cal_service import caches
from recipe_system.cal_service import Calibrations

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class PrimitivesBASE(object):
    """
    This is the base class for all of primitives classes for the geminidr 
    primitive sets. __init__.py provides, or should provide, all attributes
    needed by subclasses.

    Three parameters are required on the initializer:

    adinputs: a list of astrodata objects
        <list>

    context: the context for recipe selection, etc.
        <str>

    upmetrics: upload the QA metrics produced by the QAP.
        <bool>

    """
    tagset = None

    def __init__(self, adinputs, context=['qa'], upmetrics=False, upcalibs=False,
                 ucals=None, uparms=None):
        self.streams          = {'main': adinputs}
        self.context          = context
        self.parameters       = ParametersBASE
        self.log              = logutils.get_logger(__name__)
        self.upload_metrics   = upmetrics
        self.upload_calibs    = upcalibs
        self.user_params      = uparms if uparms else {}
        self.usercals         = ucals if ucals else {}
        self.calurl_dict      = calurl_dict.calurl_dict
        self.timestamp_keys   = timestamp_keywords.timestamp_keys
        self.keyword_comments = keyword_comments.keyword_comments
        self.sx_dict          = sextractor_dict.sx_dict
        # Prepend paths to SExtractor input files now
        self.sx_dict.update({k:
                os.path.join(os.path.dirname(sextractor_dict.__file__), v)
                for k,v in self.sx_dict.items()})

        self.cachedict        = caches.set_caches()
        self.calibrations     = Calibrations(caches.calindfile)
        self.stacks           = caches.load_cache(caches.stkindfile)

        # This lambda will return the name of the current caller.
        self.myself           = lambda: stack()[1][3]

        warnings.simplefilter('ignore', category=VerifyWarning)
