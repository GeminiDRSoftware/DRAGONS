class DescriptorDescriptor:
    name = None
    
    thunkfuncbuff = """
    def %(name)s(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "%(name)s")
            if not hasattr( self.descriptorCalculator, "%(name)s"):
                key = "key_"+"%(name)s"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.%(name)s(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    """
    def __init__(self, name = None):
        self.name = name
        
    def funcbody(self):
        ret = self.thunkfuncbuff % { "name":self.name}
        return ret
        
DD = DescriptorDescriptor
        
descriptors =   [   DD("airmass"),
                    DD("azimuth"),
                    DD("camera"),
                    DD("crpa"),
                    DD("cwave"),
                    DD("datalab"),
                    DD("datasec"),
                    DD("dec"),
                    DD("detsec"),
                    DD("disperser"),
                    DD("elevation"),
                    DD("exptime"),
                    DD("filterid"),
                    DD("filtername"),
                    DD("fpmask"),
                    DD("gain"),
                    DD("instrument"),
                    DD("mdfrow"),
                    DD("nonlinear"),
                    DD("nsciext"),
                    DD("object"),
                    DD("obsclass"),
                    DD("obsepoch"),
                    DD("observer"),
                    DD("obsid"),
                    DD("obsmode"),
                    DD("obstype"),
                    DD("pixscale"),
                    DD("progid"),
                    DD("pupilmask"),
                    DD("ra"),
                    DD("rawiq"),
                    DD("rawcc"),
                    DD("rawwv"),
                    DD("rawbg"),
                    DD("rawpireq"),
                    DD("rawgemqa"),
                    DD("rdnoise"),
                    DD("satlevel"),
                    DD("ssa"),
                    DD("telescope"),
                    DD("utdate"),
                    DD("uttime"),
                    DD("wdelta"),
                    DD("wrefpix"),
                    DD("xccdbin"),
                    DD("xoffset"),
                    DD("yccdbin"),
                    DD("amproa"),
                    DD("ccdroa"),
                    DD("readspeedmode"),
                    DD("gainmode"),
                ]

wholeout = """
import sys
import StandardDescriptorKeyDict as SDKD
from astrodata import Descriptors

class CalculatorInterface:

    descriptorCalculator = None
%(descriptors)s
# UTILITY FUNCTIONS, above are descriptor thunks            
    def _lazyloadCalculator(self, **args):
        '''Function to put at top of all descriptor members
        to ensure the descriptor is loaded.  This way we avoid
        loading it if it is not needed.'''
        if self.descriptorCalculator == None:
            self.descriptorCalculator = Descriptors.getCalculator(self, **args)
"""

out = ""
for dd in descriptors:
    out += dd.funcbody()
    
finalout = wholeout % {"descriptors": out}

print finalout
