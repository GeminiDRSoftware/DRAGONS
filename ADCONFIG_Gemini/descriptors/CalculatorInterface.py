import sys
import StandardDescriptorKeyDict as SDKD
from astrodata import Descriptors

class CalculatorInterface:

    descriptorCalculator = None

    def airmass(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "airmass")
            if not hasattr( self.descriptorCalculator, "airmass"):
                key = "key_"+"airmass"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.airmass(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def azimuth(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "azimuth")
            if not hasattr( self.descriptorCalculator, "azimuth"):
                key = "key_"+"azimuth"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.azimuth(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def camera(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "camera")
            if not hasattr( self.descriptorCalculator, "camera"):
                key = "key_"+"camera"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.camera(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def crpa(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "crpa")
            if not hasattr( self.descriptorCalculator, "crpa"):
                key = "key_"+"crpa"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.crpa(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def cwave(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "cwave")
            if not hasattr( self.descriptorCalculator, "cwave"):
                key = "key_"+"cwave"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.cwave(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def datalab(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "datalab")
            if not hasattr( self.descriptorCalculator, "datalab"):
                key = "key_"+"datalab"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.datalab(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def datasec(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "datasec")
            if not hasattr( self.descriptorCalculator, "datasec"):
                key = "key_"+"datasec"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.datasec(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def dec(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "dec")
            if not hasattr( self.descriptorCalculator, "dec"):
                key = "key_"+"dec"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.dec(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def detsec(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "detsec")
            if not hasattr( self.descriptorCalculator, "detsec"):
                key = "key_"+"detsec"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.detsec(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def disperser(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "disperser")
            if not hasattr( self.descriptorCalculator, "disperser"):
                key = "key_"+"disperser"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.disperser(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def elevation(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "elevation")
            if not hasattr( self.descriptorCalculator, "elevation"):
                key = "key_"+"elevation"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.elevation(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def exptime(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "exptime")
            if not hasattr( self.descriptorCalculator, "exptime"):
                key = "key_"+"exptime"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.exptime(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def filterid(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "filterid")
            if not hasattr( self.descriptorCalculator, "filterid"):
                key = "key_"+"filterid"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.filterid(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def filtername(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "filtername")
            if not hasattr( self.descriptorCalculator, "filtername"):
                key = "key_"+"filtername"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.filtername(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def fpmask(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "fpmask")
            if not hasattr( self.descriptorCalculator, "fpmask"):
                key = "key_"+"fpmask"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.fpmask(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def gain(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "gain")
            if not hasattr( self.descriptorCalculator, "gain"):
                key = "key_"+"gain"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.gain(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def instrument(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "instrument")
            if not hasattr( self.descriptorCalculator, "instrument"):
                key = "key_"+"instrument"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.instrument(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def mdfrow(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "mdfrow")
            if not hasattr( self.descriptorCalculator, "mdfrow"):
                key = "key_"+"mdfrow"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.mdfrow(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def nonlinear(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "nonlinear")
            if not hasattr( self.descriptorCalculator, "nonlinear"):
                key = "key_"+"nonlinear"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.nonlinear(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def nsciext(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "nsciext")
            if not hasattr( self.descriptorCalculator, "nsciext"):
                key = "key_"+"nsciext"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.nsciext(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def object(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "object")
            if not hasattr( self.descriptorCalculator, "object"):
                key = "key_"+"object"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.object(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def obsclass(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "obsclass")
            if not hasattr( self.descriptorCalculator, "obsclass"):
                key = "key_"+"obsclass"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.obsclass(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def obsepoch(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "obsepoch")
            if not hasattr( self.descriptorCalculator, "obsepoch"):
                key = "key_"+"obsepoch"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.obsepoch(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def observer(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "observer")
            if not hasattr( self.descriptorCalculator, "observer"):
                key = "key_"+"observer"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.observer(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def obsid(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "obsid")
            if not hasattr( self.descriptorCalculator, "obsid"):
                key = "key_"+"obsid"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.obsid(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def obsmode(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "obsmode")
            if not hasattr( self.descriptorCalculator, "obsmode"):
                key = "key_"+"obsmode"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.obsmode(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def obstype(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "obstype")
            if not hasattr( self.descriptorCalculator, "obstype"):
                key = "key_"+"obstype"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.obstype(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def pixscale(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "pixscale")
            if not hasattr( self.descriptorCalculator, "pixscale"):
                key = "key_"+"pixscale"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.pixscale(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def progid(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "progid")
            if not hasattr( self.descriptorCalculator, "progid"):
                key = "key_"+"progid"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.progid(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def pupilmask(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "pupilmask")
            if not hasattr( self.descriptorCalculator, "pupilmask"):
                key = "key_"+"pupilmask"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.pupilmask(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def ra(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "ra")
            if not hasattr( self.descriptorCalculator, "ra"):
                key = "key_"+"ra"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.ra(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def rawiq(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "rawiq")
            if not hasattr( self.descriptorCalculator, "rawiq"):
                key = "key_"+"rawiq"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.rawiq(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def rawcc(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "rawcc")
            if not hasattr( self.descriptorCalculator, "rawcc"):
                key = "key_"+"rawcc"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.rawcc(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def rawwv(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "rawwv")
            if not hasattr( self.descriptorCalculator, "rawwv"):
                key = "key_"+"rawwv"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.rawwv(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def rawbg(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "rawbg")
            if not hasattr( self.descriptorCalculator, "rawbg"):
                key = "key_"+"rawbg"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.rawbg(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def rawpireq(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "rawpireq")
            if not hasattr( self.descriptorCalculator, "rawpireq"):
                key = "key_"+"rawpireq"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.rawpireq(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def rawgemqa(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "rawgemqa")
            if not hasattr( self.descriptorCalculator, "rawgemqa"):
                key = "key_"+"rawgemqa"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.rawgemqa(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def rdnoise(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "rdnoise")
            if not hasattr( self.descriptorCalculator, "rdnoise"):
                key = "key_"+"rdnoise"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.rdnoise(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def satlevel(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "satlevel")
            if not hasattr( self.descriptorCalculator, "satlevel"):
                key = "key_"+"satlevel"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.satlevel(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def ssa(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "ssa")
            if not hasattr( self.descriptorCalculator, "ssa"):
                key = "key_"+"ssa"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.ssa(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def telescope(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "telescope")
            if not hasattr( self.descriptorCalculator, "telescope"):
                key = "key_"+"telescope"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.telescope(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def utdate(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "utdate")
            if not hasattr( self.descriptorCalculator, "utdate"):
                key = "key_"+"utdate"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.utdate(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def uttime(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "uttime")
            if not hasattr( self.descriptorCalculator, "uttime"):
                key = "key_"+"uttime"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.uttime(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def wdelta(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "wdelta")
            if not hasattr( self.descriptorCalculator, "wdelta"):
                key = "key_"+"wdelta"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.wdelta(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def wrefpix(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "wrefpix")
            if not hasattr( self.descriptorCalculator, "wrefpix"):
                key = "key_"+"wrefpix"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.wrefpix(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def xccdbin(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "xccdbin")
            if not hasattr( self.descriptorCalculator, "xccdbin"):
                key = "key_"+"xccdbin"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.xccdbin(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def xoffset(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "xoffset")
            if not hasattr( self.descriptorCalculator, "xoffset"):
                key = "key_"+"xoffset"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.xoffset(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def yccdbin(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "yccdbin")
            if not hasattr( self.descriptorCalculator, "yccdbin"):
                key = "key_"+"yccdbin"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.yccdbin(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
    def yoffset(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "yoffset")
            if not hasattr( self.descriptorCalculator, "yoffset"):
                key = "key_"+"yoffset"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.yoffset(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None

    def readout_dwelltime(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "readout_dwelltime")
            if not hasattr( self.descriptorCalculator, "readout_dwelltime"):
                key = "key_"+"readout_dwelltime"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
                else:
                    return None
            return self.descriptorCalculator.readout_dwelltime(self, **args)
        except:
            #print "NONE BY EXCEPTION"
            self.noneMsg = str(sys.exc_info()[1])
            return None
    
# UTILITY FUNCTIONS, above are descriptor thunks            
    def _lazyloadCalculator(self, **args):
        '''Function to put at top of all descriptor members
        to ensure the descriptor is loaded.  This way we avoid
        loading it if it is not needed.'''
        if self.descriptorCalculator == None:
            self.descriptorCalculator = Descriptors.getCalculator(self, **args)

