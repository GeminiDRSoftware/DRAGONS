from primitives_SPECT import PrimitivesSPECT

class PrimitivesNODANDSHUFFLE(PrimitivesSPECT):
    """
    This is the class containing all of the primitives for the GMOS_SPECT 
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'GMOSPrimitives'.
    """
    adtags = ("GMOS", "NODANDSHUFFLE")
    
    def init(self, rc):
        PrimitivesSPECT.init(self, rc)
        return rc
