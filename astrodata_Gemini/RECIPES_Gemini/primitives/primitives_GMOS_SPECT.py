from primitives_GMOS import GMOSPrimitives

class GMOS_SPECTPrimitives(GMOSPrimitives):
    """
    This is the class containing all of the primitives for the F2_IMAGE
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'GMOSPrimitives'.
    """
    astrotype = "GMOS_SPECT"
    
    def init(self, rc):
        GMOSPrimitives.init(self, rc)
        return rc
