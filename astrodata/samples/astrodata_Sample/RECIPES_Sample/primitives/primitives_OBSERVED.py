from astrodata.ReductionObjects import PrimitiveSet

class OBSERVEDPrimitives(PrimitiveSet):
    astrotype = "OBSERVED"
    
    def init(self, rc):
        print "OBSERVEDPrimitives.init(rc)"
        return
    def typeSpecificPrimitive(self, rc):
        print "OBSERVEDPrimitives::typeSpecificPrimitive()"
        
    def mark(self, rc):
        for ad in rc.get_inputs_as_astrodata():
            if ad.is_type("MARKED"):
                print "OBSERVEDPrimitives::mark(%s) already marked" % ad.filename
            else:
                ad.phu_set_key_value("S_MARKED", "TRUE")
                rc.report_output(ad)
        yield rc 
        
    def unmark(self, rc):
        for ad in rc.get_inputs_as_astrodata():
            if ad.is_type("UNMARKED"):
                print "OBSERVEDPrimitives::unmark(%s) not marked" % ad.filename
            else:
                ad.phu_set_key_value("S_MARKED", None)
                rc.report_output(ad)
        yield rc
