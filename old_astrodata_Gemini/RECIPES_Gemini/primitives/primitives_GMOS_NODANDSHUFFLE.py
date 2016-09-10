from primitives_GMOS_SPECT import GMOS_SPECTPrimitives

class GMOS_NODANDSHUFFLEPrimitives(GMOS_SPECTPrimitives):
    """
    This is the class containing all of the primitives for the GMOS_SPECT 
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'GMOSPrimitives'.
    """
    astrotype = "GMOS_NODANDSHUFFLE"
    
    def init(self, rc):
        GMOS_SPECTPrimitives.init(self, rc)
        return rc
