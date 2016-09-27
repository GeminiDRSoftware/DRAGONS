from GMOS.primitives_GMOS import PrimitivesGMOS
from GMOS.parameters_IMAGE import ParametersIMAGE

from rpms.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class PrimitivesIMAGE(PrimitivesGMOS):
    """
    This primitive class provides all primitives for GMOS IMAGE datasets, as 
    indicated by a dataset's tag set. It inherits all primitives from the
    level above, 'GMOSPrimitives'.
    """
    tagset = set(["GMOS", "IMAGE"])

    def __init__(self, adinputs, context, ucals=None, uparms=None):
        """
        Receive a list of astrodata objects and a dictionary of user-supplied 
        parameters. Typically, adinputs is of length 1, a single dataset.

        The 'uparms' dictionary will be checked by each primitive method when
        called to determine whether the user parameter(s) apply, either
        generally, or if the caller has specified the primitive explicitly.

        E.g.,

        uparms = {'par1': 'val1', 'makeFringeFrame:reject_method': 'jilt'}

        Non-explicit parameters are checked against a primitive's own 
        parameter set and determine if the user parameter overrides apply.

        :parameter adinputs: list of AstroData instances
        :type adinputs: <list>, [ad1, ... ]

        :parameter uparms: User supplied parameters, usually via a 'reduce'
                           command line option, -p
        :type uparms: <dict>, a dictionary of key:val pairs.

        """
        super(self.__class__, self).__init__(adinputs, context, 
                                             ucals=ucals, uparms=uparms)
        self.parameters = ParametersIMAGE
        self.primitive_parset = None
        if self.user_params:
            print "PrimitivesIMAGE recieved user parameters:"
            print str(self.user_params)
        else:
            print "No user parameters specified."


    def fringeCorrect(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        self._primitive_exec(self.myself(), parset=self.primitive_parset, indent=3)
        self.getProcessedFringe()
        return

    def makeFringe(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        self._primitive_exec(self.myself(), parset=self.primitive_parset, indent=3)
        self.makeFringeFrame()
        self.storeProcessedFringe()
        return

    def makeFringeFrame(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        self._primitive_exec(self.myself(), parset=self.primitive_parset, indent=3)
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
        self._primitive_exec(self.myself(), parset=self.primitive_parset, indent=3)
        return
    
    def scaleByIntensity(self, adinputs=None, stream='main', **params):
        """
        This primitive scales input images to the mean value of the first
        image.  It is intended to be used to scale flats to the same
        level before stacking.
       
        This primitive is prototype demo.

        """        
        self._primitive_exec(self.myself(), parset=self.primitive_parset, indent=3)
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
        self._primitive_exec(self.myself(), parset=self.primitive_parset, indent=3)
        return
        
    
    def stackFlats(self, adinputs=None, stream='main', **params):
        """
        This primitive will combine the input flats with rejection
        parameters set appropriately for GMOS imaging twilight flats.
 
        This primitive is prototype demo.

        """        
        self._primitive_exec(self.myself(), parset=self.primitive_parset, indent=3)
        return

    def subtractFringe(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        self._primitive_exec(self.myself(), parset=self.primitive_parset, indent=3)
        return
