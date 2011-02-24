
import sys
import StandardDescriptorKeyDict as SDKD
from astrodata import Descriptors

class CalculatorInterface:

    descriptorCalculator = None

    def airmass(self, **args):
        """
        description of what it does
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "airmass")
            if not hasattr( self.descriptorCalculator, "airmass"):
                key = "key_"+"airmass"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.airmass(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def amp_read_area(self, **args):
        """
        Return the amp_read_area value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "amp_read_area")
            if not hasattr( self.descriptorCalculator, "amp_read_area"):
                key = "key_"+"amp_read_area"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.amp_read_area(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def azimuth(self, **args):
        """
        Return the azimuth value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "azimuth")
            if not hasattr( self.descriptorCalculator, "azimuth"):
                key = "key_"+"azimuth"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.azimuth(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def camera(self, **args):
        """
        Return the camera value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "camera")
            if not hasattr( self.descriptorCalculator, "camera"):
                key = "key_"+"camera"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.camera(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def cass_rotator_pa(self, **args):
        """
        Return the cass_rotator_pa value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "cass_rotator_pa")
            if not hasattr( self.descriptorCalculator, "cass_rotator_pa"):
                key = "key_"+"cass_rotator_pa"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.cass_rotator_pa(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def central_wavelength(self, **args):
        """
        Return the central_wavelength value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "central_wavelength")
            if not hasattr( self.descriptorCalculator, "central_wavelength"):
                key = "key_"+"central_wavelength"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.central_wavelength(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def coadds(self, **args):
        """
        Return the coadds value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "coadds")
            if not hasattr( self.descriptorCalculator, "coadds"):
                key = "key_"+"coadds"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.coadds(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def data_label(self, **args):
        """
        Return the data_label value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "data_label")
            if not hasattr( self.descriptorCalculator, "data_label"):
                key = "key_"+"data_label"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.data_label(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def data_section(self, **args):
        """
        Return the data_section value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "data_section")
            if not hasattr( self.descriptorCalculator, "data_section"):
                key = "key_"+"data_section"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.data_section(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def dec(self, **args):
        """
        Return the dec value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "dec")
            if not hasattr( self.descriptorCalculator, "dec"):
                key = "key_"+"dec"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.dec(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def decker(self, **args):
        """
        Return the decker value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "decker")
            if not hasattr( self.descriptorCalculator, "decker"):
                key = "key_"+"decker"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.decker(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def detector_section(self, **args):
        """
        Return the detector_section value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "detector_section")
            if not hasattr( self.descriptorCalculator, "detector_section"):
                key = "key_"+"detector_section"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.detector_section(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def detector_x_bin(self, **args):
        """
        Return the detector_x_bin value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "detector_x_bin")
            if not hasattr( self.descriptorCalculator, "detector_x_bin"):
                key = "key_"+"detector_x_bin"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.detector_x_bin(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def detector_y_bin(self, **args):
        """
        Return the detector_y_bin value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "detector_y_bin")
            if not hasattr( self.descriptorCalculator, "detector_y_bin"):
                key = "key_"+"detector_y_bin"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.detector_y_bin(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def disperser(self, **args):
        """
        Return the disperser value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "disperser")
            if not hasattr( self.descriptorCalculator, "disperser"):
                key = "key_"+"disperser"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.disperser(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def dispersion(self, **args):
        """
        Return the dispersion value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "dispersion")
            if not hasattr( self.descriptorCalculator, "dispersion"):
                key = "key_"+"dispersion"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.dispersion(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def dispersion_axis(self, **args):
        """
        Return the dispersion_axis value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "dispersion_axis")
            if not hasattr( self.descriptorCalculator, "dispersion_axis"):
                key = "key_"+"dispersion_axis"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.dispersion_axis(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def elevation(self, **args):
        """
        Return the elevation value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "elevation")
            if not hasattr( self.descriptorCalculator, "elevation"):
                key = "key_"+"elevation"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.elevation(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def exposure_time(self, **args):
        """
        Return the exposure_time value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "exposure_time")
            if not hasattr( self.descriptorCalculator, "exposure_time"):
                key = "key_"+"exposure_time"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.exposure_time(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def filter_name(self, **args):
        """
        Return the filter_name value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "filter_name")
            if not hasattr( self.descriptorCalculator, "filter_name"):
                key = "key_"+"filter_name"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.filter_name(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def focal_plane_mask(self, **args):
        """
        Return the focal_plane_mask value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "focal_plane_mask")
            if not hasattr( self.descriptorCalculator, "focal_plane_mask"):
                key = "key_"+"focal_plane_mask"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.focal_plane_mask(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def gain(self, **args):
        """
        Return the gain value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "gain")
            if not hasattr( self.descriptorCalculator, "gain"):
                key = "key_"+"gain"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.gain(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def grating(self, **args):
        """
        Return the grating value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "grating")
            if not hasattr( self.descriptorCalculator, "grating"):
                key = "key_"+"grating"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.grating(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def gain_setting(self, **args):
        """
        Return the gain_setting value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "gain_setting")
            if not hasattr( self.descriptorCalculator, "gain_setting"):
                key = "key_"+"gain_setting"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.gain_setting(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def instrument(self, **args):
        """
        Return the instrument value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "instrument")
            if not hasattr( self.descriptorCalculator, "instrument"):
                key = "key_"+"instrument"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.instrument(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def local_time(self, **args):
        """
        Return the local_time value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "local_time")
            if not hasattr( self.descriptorCalculator, "local_time"):
                key = "key_"+"local_time"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.local_time(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def mdf_row_id(self, **args):
        """
        Return the mdf_row_id value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "mdf_row_id")
            if not hasattr( self.descriptorCalculator, "mdf_row_id"):
                key = "key_"+"mdf_row_id"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.mdf_row_id(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def nod_count(self, **args):
        """
        Return the nod_count value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "nod_count")
            if not hasattr( self.descriptorCalculator, "nod_count"):
                key = "key_"+"nod_count"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.nod_count(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def nod_pixels(self, **args):
        """
        Return the nod_pixels value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "nod_pixels")
            if not hasattr( self.descriptorCalculator, "nod_pixels"):
                key = "key_"+"nod_pixels"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.nod_pixels(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def non_linear_level(self, **args):
        """
        Return the non_linear_level value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "non_linear_level")
            if not hasattr( self.descriptorCalculator, "non_linear_level"):
                key = "key_"+"non_linear_level"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.non_linear_level(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def object(self, **args):
        """
        Return the object value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "object")
            if not hasattr( self.descriptorCalculator, "object"):
                key = "key_"+"object"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.object(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def observation_class(self, **args):
        """
        Return the observation_class value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "observation_class")
            if not hasattr( self.descriptorCalculator, "observation_class"):
                key = "key_"+"observation_class"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.observation_class(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def observation_epoch(self, **args):
        """
        Return the observation_epoch value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "observation_epoch")
            if not hasattr( self.descriptorCalculator, "observation_epoch"):
                key = "key_"+"observation_epoch"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.observation_epoch(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def observation_id(self, **args):
        """
        Return the observation_id value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "observation_id")
            if not hasattr( self.descriptorCalculator, "observation_id"):
                key = "key_"+"observation_id"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.observation_id(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def observation_type(self, **args):
        """
        Return the observation_type value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "observation_type")
            if not hasattr( self.descriptorCalculator, "observation_type"):
                key = "key_"+"observation_type"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.observation_type(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def pixel_scale(self, **args):
        """
        Return the pixel_scale value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "pixel_scale")
            if not hasattr( self.descriptorCalculator, "pixel_scale"):
                key = "key_"+"pixel_scale"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.pixel_scale(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def prism(self, **args):
        """
        Return the prism value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "prism")
            if not hasattr( self.descriptorCalculator, "prism"):
                key = "key_"+"prism"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.prism(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def program_id(self, **args):
        """
        Return the program_id value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "program_id")
            if not hasattr( self.descriptorCalculator, "program_id"):
                key = "key_"+"program_id"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.program_id(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def pupil_mask(self, **args):
        """
        Return the pupil_mask value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "pupil_mask")
            if not hasattr( self.descriptorCalculator, "pupil_mask"):
                key = "key_"+"pupil_mask"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.pupil_mask(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def qa_state(self, **args):
        """
        Return the qa_state value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "qa_state")
            if not hasattr( self.descriptorCalculator, "qa_state"):
                key = "key_"+"qa_state"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.qa_state(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def ra(self, **args):
        """
        Return the ra value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "ra")
            if not hasattr( self.descriptorCalculator, "ra"):
                key = "key_"+"ra"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.ra(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def raw_bg(self, **args):
        """
        Return the raw_bg value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "raw_bg")
            if not hasattr( self.descriptorCalculator, "raw_bg"):
                key = "key_"+"raw_bg"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.raw_bg(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def raw_cc(self, **args):
        """
        Return the raw_cc value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "raw_cc")
            if not hasattr( self.descriptorCalculator, "raw_cc"):
                key = "key_"+"raw_cc"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.raw_cc(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def raw_iq(self, **args):
        """
        Return the raw_iq value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "raw_iq")
            if not hasattr( self.descriptorCalculator, "raw_iq"):
                key = "key_"+"raw_iq"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.raw_iq(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def raw_wv(self, **args):
        """
        Return the raw_wv value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "raw_wv")
            if not hasattr( self.descriptorCalculator, "raw_wv"):
                key = "key_"+"raw_wv"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.raw_wv(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def read_mode(self, **args):
        """
        Return the read_mode value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "read_mode")
            if not hasattr( self.descriptorCalculator, "read_mode"):
                key = "key_"+"read_mode"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.read_mode(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def read_noise(self, **args):
        """
        Return the read_noise value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "read_noise")
            if not hasattr( self.descriptorCalculator, "read_noise"):
                key = "key_"+"read_noise"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.read_noise(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def read_speed_setting(self, **args):
        """
        Return the read_speed_setting value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "read_speed_setting")
            if not hasattr( self.descriptorCalculator, "read_speed_setting"):
                key = "key_"+"read_speed_setting"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.read_speed_setting(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def saturation_level(self, **args):
        """
        Return the saturation_level value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "saturation_level")
            if not hasattr( self.descriptorCalculator, "saturation_level"):
                key = "key_"+"saturation_level"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.saturation_level(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def slit(self, **args):
        """
        Return the slit value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "slit")
            if not hasattr( self.descriptorCalculator, "slit"):
                key = "key_"+"slit"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.slit(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def telescope(self, **args):
        """
        Return the telescope value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "telescope")
            if not hasattr( self.descriptorCalculator, "telescope"):
                key = "key_"+"telescope"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.telescope(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def ut_date(self, **args):
        """
        Return the ut_date value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "ut_date")
            if not hasattr( self.descriptorCalculator, "ut_date"):
                key = "key_"+"ut_date"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.ut_date(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def ut_datetime(self, **args):
        """
        Return the ut_datetime value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "ut_datetime")
            if not hasattr( self.descriptorCalculator, "ut_datetime"):
                key = "key_"+"ut_datetime"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.ut_datetime(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def ut_time(self, **args):
        """
        Return the ut_time value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "ut_time")
            if not hasattr( self.descriptorCalculator, "ut_time"):
                key = "key_"+"ut_time"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.ut_time(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def wavefront_sensor(self, **args):
        """
        Return the wavefront_sensor value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "wavefront_sensor")
            if not hasattr( self.descriptorCalculator, "wavefront_sensor"):
                key = "key_"+"wavefront_sensor"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.wavefront_sensor(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def wavelength_reference_pixel(self, **args):
        """
        Return the wavelength_reference_pixel value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "wavelength_reference_pixel")
            if not hasattr( self.descriptorCalculator, "wavelength_reference_pixel"):
                key = "key_"+"wavelength_reference_pixel"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.wavelength_reference_pixel(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def well_depth_setting(self, **args):
        """
        Return the well_depth_setting value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "well_depth_setting")
            if not hasattr( self.descriptorCalculator, "well_depth_setting"):
                key = "key_"+"well_depth_setting"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.well_depth_setting(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def x_offset(self, **args):
        """
        Return the x_offset value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "x_offset")
            if not hasattr( self.descriptorCalculator, "x_offset"):
                key = "key_"+"x_offset"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.x_offset(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def y_offset(self, **args):
        """
        Return the y_offset value for generic data
        """
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "y_offset")
            if not hasattr( self.descriptorCalculator, "y_offset"):
                key = "key_"+"y_offset"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            retval = self.descriptorCalculator.y_offset(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(a, datetime):                    
                    retval = stdDateString(a)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
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

