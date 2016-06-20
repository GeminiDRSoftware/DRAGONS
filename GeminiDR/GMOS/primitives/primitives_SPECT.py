import numpy as np
from copy import deepcopy 

from astrodata import AstroData
from astrodata.utils import Errors
from astrodata.utils import logutils
from astrodata.utils.gemutil import pyrafLoader
from astrodata.interface.slices import pixel_exts
from astrodata.utils.gemconstants import SCI, DQ

from gempy.gemini import gemini_tools as gt
from gempy.gemini import eti

from primitives_GMOS import PrimitivesGMOS
from GMOS.parameters.parameters_SPECT import ParametersSPECT
# ------------------------------------------------------------------------------
class PrimitivesSPECT(PrimitivesGMOS):
    """
    This is the class containing all of the primitives for the GMOS_SPECT 
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'GMOSPrimitives'.
    """
    adtags = ("GMOS", "SPECT")
    
    def __init__(self, adinputs):
        super(PrimitivesSPECT, self).__init__(adinputs)
        self.parameters = ParametersSPECT

    def determineWavelengthSolution(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """        
        self._primitive_exec(self.myself(), self.parameters.determineWavelengthSolution, indent=3)
        return

    def extract1DSpectra(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """        
        self._primitive_exec(self.myself(), self.parameters.extract1DSpectra, indent=3)
        return
        
    def findAcquisitionSlits(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """        
        self._primitive_exec(self.myself(), self.parameters.findAcquisitionSlits, indent=3)
        return

    def makeFlat(self,adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """        
        self._primitive_exec(self.myself(), self.parameters.makeFlat, indent=3)
        return

    def rejectCosmicRays(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """        
        self._primitive_exec(self.myself(), self.parameters.rejectCosmicRays, indent=3)
        return

    def resampleToLinearCoords(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """        
        self._primitive_exec(self.myself(), self.parameters.resampleToLinearCoords, indent=3)
        return

    def skyCorrectFromSlit(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """        
        self._primitive_exec(self.myself(), self.parameters.skyCorrectFromSlit, indent=3)
        return

    def skyCorrectNodShuffle(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """        
        self._primitive_exec(self.myself(), self.parameters.skyCorrectNodShuffle, indent=3)
        return



        

