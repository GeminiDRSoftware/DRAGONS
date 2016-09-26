from astrodata.utils import logutils

from .parameters_display import ParametersDisplay
from .primitives_CORE import PrimitivesCORE

# ------------------------------------------------------------------------------
from pkg_utilities.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class Display(PrimitivesCORE):
    """
    This is the class containing all of the display primitives
    for the GEMINI level of the type hierarchy tree. It inherits
    all the primitives from the level above, 'GENERALPrimitives'.
    """
    tagset = set(["GEMINI"])

    def __init__(self, adinputs, context, ucals=None, uparms=None):
        super(Display, self).__init__(adinputs, context, 
                                               ucals=ucals, uparms=uparms)
        self.parameters = ParametersDisplay
    
    def display(self, adinputs=None, stream='main', **params):
        """
        Demo prototype.

        """
        log = logutils.get_logger(__name__)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(self.parameters.display))

        return
