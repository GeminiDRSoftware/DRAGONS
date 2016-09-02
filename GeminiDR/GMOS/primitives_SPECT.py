from GMOS.primitives_GMOS import PrimitivesGMOS
from GMOS.parameters_SPECT import ParametersSPECT

# ------------------------------------------------------------------------------
class PrimitivesSPECT(PrimitivesGMOS):
    """
    This is the class containing all of the primitives for the GMOS_SPECT 
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'GMOSPrimitives'.
    """
    tagset = set(["GMOS", "SPECT"])
    
    def __init__(self, adinputs):
        super(self.__class__, self).__init__(adinputs)
        self.parameters = ParametersSPECT

    def determineWavelengthSolution(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        try:
            parset = getattr(self.parameters, self.myself())
        except AttributeError:
            parset = None
        self._primitive_exec(self.myself(), parset=parset, indent=3)
        return

    def extract1DSpectra(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """        
        try:
            parset = getattr(self.parameters, self.myself())
        except AttributeError:
            parset = None
        self._primitive_exec(self.myself(), parset=parset, indent=3)
        return
        
    def findAcquisitionSlits(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """        
        try:
            parset = getattr(self.parameters, self.myself())
        except AttributeError:
            parset = None
        self._primitive_exec(self.myself(), parset=parset, indent=3)
        return

    def makeFlat(self,adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """        
        try:
            parset = getattr(self.parameters, self.myself())
        except AttributeError:
            parset = None
        self._primitive_exec(self.myself(), parset=parset, indent=3)
        return

    def rejectCosmicRays(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """        
        try:
            parset = getattr(self.parameters, self.myself())
        except AttributeError:
            parset = None
        self._primitive_exec(self.myself(), parset=parset, indent=3)
        return

    def resampleToLinearCoords(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """        
        try:
            parset = getattr(self.parameters, self.myself())
        except AttributeError:
            parset = None
        self._primitive_exec(self.myself(), parset=parset, indent=3)
        return

    def skyCorrectFromSlit(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """        
        try:
            parset = getattr(self.parameters, self.myself())
        except AttributeError:
            parset = None
        self._primitive_exec(self.myself(), parset=parset, indent=3)
        return

    def skyCorrectNodShuffle(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """        
        try:
            parset = getattr(self.parameters, self.myself())
        except AttributeError:
            parset = None
        self._primitive_exec(self.myself(), parset=parset, indent=3)
        return

