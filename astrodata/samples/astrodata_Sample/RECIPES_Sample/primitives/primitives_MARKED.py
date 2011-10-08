from primitives_OBSERVED import OBSERVEDPrimitives

class MARKEDPrimitives(OBSERVEDPrimitives):
    astrotype = "MARKED"
    
    def init(self, rc):
        print "MARKEDPrimitives.init(rc)"
        return
        
    def typeSpecificPrimitive(self, rc):
        print "MARKEDPrimitives::typeSpecificPrimitive"
        yield rc
    
