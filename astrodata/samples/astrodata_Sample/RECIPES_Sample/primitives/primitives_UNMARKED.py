from primitives_OBSERVED import OBSERVEDPrimitives

class UNMARKEDPrimitives(OBSERVEDPrimitives):
    astrotype = "UNMARKED"
    
    def init(self, rc):
        print "UNMARKEDPrimitives.init(rc)"
        return
        
    def typeSpecificPrimitive(self, rc):
        print "UNMARKEDPrimitives::typeSpecificPrimitive"
        yield rc
    
