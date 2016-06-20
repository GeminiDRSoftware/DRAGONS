import os
import numpy as np
from copy import deepcopy

import pifgemini.gmos_image as gmi

from astrodata import AstroData
from astrodata.utils import Errors
from astrodata.utils import Lookups
from astrodata.utils import logutils
from astrodata.utils.gemutil import pyrafLoader

from gempy.gemini import gemini_tools as gt

from primitives_GMOS import PrimitivesGMOS
from GMOS.parameters.parameters_IMAGE import ParametersIMAGE

# ------------------------------------------------------------------------------
class PrimitivesIMAGE(PrimitivesGMOS):
    """
    This is the class containing all of the primitives for the GMOS_IMAGE
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'GMOSPrimitives'.
    """
    tag = ("GMOS", "IMAGE")
    
    def __init__(self, adinputs):
        super(PrimitivesIMAGE, self).__init__(adinputs)
        self.parameters = ParametersIMAGE

    def fringeCorrect(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """        
        self._primitive_exec(self.myself(), indent=3)
        return
    
    def makeFringe(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """        
        self._primitive_exec(self.myself(), self.parameters.makeFringe, indent=3)
        return

    def makeFringeFrame(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """        
        self._primitive_exec(self.myself(), self.parameters.makeFringeFrame, indent=3)
        return

    def normalizeFlat(self, adinputs=None, stream='main', **params):
        """
        This primitive will calculate a normalization factor from statistics
        on CCD2, then divide by this factor and propagate variance accordingly.
        CCD2 is used because of the dome-like shape of the GMOS detector
        response: CCDs 1 and 3 have lower average illumination than CCD2, 
        and that needs to be corrected for by the flat.
        
        This primitive is prototype demo.

        """        
        self._primitive_exec(self.myself(), self.parameters.normalizeFlat, indent=3)
        return
    
    def scaleByIntensity(self, adinputs=None, stream='main', **params):
        """
        This primitive scales input images to the mean value of the first
        image.  It is intended to be used to scale flats to the same
        level before stacking.
       
        This primitive is prototype demo.

        """        
        self._primitive_exec(self.myself(), self.parameters.scaleByIntensity, indent=3)
        return

    def scaleFringeToScience(self, adinputs=None, stream='main', **params):
        """
        This primitive will scale the fringes to their matching science data
        The fringes should be in the stream this primitive is called on,
        and the reference science frames should be loaded into the RC,
        as, eg. rc["science"] = adinput.
        
        There are two ways to find the value to scale fringes by:
        1. If stats_scale is set to True, the equation:
        (letting science data = b (or B), and fringe = a (or A))
    
        arrayB = where({where[SCIb < (SCIb.median+2.5*SCIb.std)]} 
                          > [SCIb.median-3*SCIb.std])
        scale = arrayB.std / SCIa.std
    
        The section of the SCI arrays to use for calculating these statistics
        is the CCD2 SCI data excluding the outer 5% pixels on all 4 sides.
        Future enhancement: allow user to choose section
    
        2. If stats_scale=False, then scale will be calculated using:
        exposure time of science / exposure time of fringe

        :param stats_scale: Use statistics to calculate the scale values,
                            rather than exposure time
        :type stats_scale: Python boolean (True/False)

        This primitive is prototype demo.

        """        
        self._primitive_exec(self.myself(), self.parameters.scaleFringeToScience, indent=3)
        return
        
    
    def stackFlats(self, adinputs=None, stream='main', **params):
        """
        This primitive will combine the input flats with rejection
        parameters set appropriately for GMOS imaging twilight flats.
 
        This primitive is prototype demo.

        """        
        self._primitive_exec(self.myself(), self.parameters.stackFlats, indent=3)
        return

    def subtractFringe(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """        
        self._primitive_exec(self.myself(), self.parameters.subtractFringe, indent=3)
        return
    
