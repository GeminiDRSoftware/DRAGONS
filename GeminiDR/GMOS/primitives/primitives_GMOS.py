import os

# Demo prototype primitive sets.
from astrodata.utils import logutils

from gempy.gemini import gemini_tools as gt
# new system imports - 10-06-2016 kra

from GMOS.lookups import GMOSArrayGaps
from GMOS.parameters.parameters_GMOS import ParametersGMOS
from GEMINI.primitives.primitives_GEMINI import PrimitivesGemini

# ------------------------------------------------------------------------------
class PrimitivesGMOS(PrimitivesGemini):
    """
    This is the class containing all of the primitives for the GMOS level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'GEMINIPrimitives'.
    """
    tag = "GMOS"

    def __init__(self, adinputs):
        super(PrimitivesGMOS, self).__init__(adinputs)
        self.parameters = ParametersGMOS

    def biasCorrect(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo. 
        Emulates subrecipe biasCorrect.

        """
        self._primitive_exec(self.myself(), indent=3)
        self.getProcessedBias()
        self.subtractBias()
        return

    
    def mosaicDetectors(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        try:
            parset = getattr(self.parameters, self.myself())
        except AttributeError:
            parset = None
        self._primitive_exec(self.myself(), parset=parset, indent=3)
        return


    def overscanCorrect(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        try:
            parset = getattr(self.parameters, self.myself())
        except AttributeError:
            parset = None
        self._primitive_exec(self.myself(), parset=parset, indent=3)
        self.subtractOverscan(params)
        self.trimOverscan(params)
        return


    def standardizeInstrumentHeaders(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        try:
            parset = getattr(self.parameters, self.myself())
        except AttributeError:
            parset = None
        self._primitive_exec(self.myself(), parset=parset, indent=3)
        return


    def standardizeStructure(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        try:
            parset = getattr(self.parameters, self.myself())
        except AttributeError:
            parset = None
        self._primitive_exec(self.myself(), parset=parset, indent=3)
        return


    def subtractBias(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        try:
            parset = getattr(self.parameters, self.myself())
        except AttributeError:
            parset = None
        self._primitive_exec(self.myself(), parset=parset, indent=3)
        return


    def subtractOverscan(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        try:
            parset = getattr(self.parameters, self.myself())
        except AttributeError:
            parset = None
        self._primitive_exec(self.myself(), parset=parset, indent=3)
        return


    def tileArrays(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.
        """
        try:
            parset = getattr(self.parameters, self.myself())
        except AttributeError:
            parset = None
        self._primitive_exec(self.myself(), parset=parset, indent=3)
        return


    def trimOverscan(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        try:
            parset = getattr(self.parameters, self.myself())
        except AttributeError:
            parset = None
        self._primitive_exec(self.myself(), parset=parset, indent=3)
        return


    def validateData(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        try:
            parset = getattr(self.parameters, self.myself())
        except AttributeError:
            parset = None
        self._primitive_exec(self.myself(), parset=parset, indent=3)
        return
