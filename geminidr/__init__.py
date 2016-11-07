
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

from gempy.utils import logutils
from gempy.gemini import gemini_tools as gt


# new system imports - 10-06-2016 kra
# NOTE: imports of these and other tables will be moving around ...
from gemini.lookups import calurl_dict
from gemini.lookups import keyword_comments
from gemini.lookups import timestamp_keywords
from gemini.lookups.source_detection import sextractor_default_dict

from recipe_system.utils.decorators import parameter_override

# ------------------------------------------------------------------------------
@parameter_override
class PrimitivesBASE(object):
    """
    This is the base class for all of primitives classes for the geminidr 
    primitive sets. __init__.py provides, or should provide, all attributes
    needed by subclasses. 

    """
    tagset = None

    def __init__(self, adinputs, context, ucals=None, uparms=None):
        self.adinputs         = adinputs
        self.adoutputs        = None
        self.context          = context
        self.parameters       = ParametersBASE
        self.log              = logutils.get_logger(__name__)
        self.user_params      = uparms if uparms else {}
        self.usercals         = ucals if ucals else {}
        self.calurl_dict      = calurl_dict.calurl_dict
        self.timestamp_keys   = timestamp_keywords.timestamp_keys
        self.keyword_comments = keyword_comments.keyword_comments
        self.sx_default_dict  = sextractor_default_dict.sextractor_default_dict

        # This lambda will return the name of the current caller.
        self.myself           = lambda: stack()[1][3]
