
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
            return self.descriptorCalculator.airmass(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def amp_read_area(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "amp_read_area")
            if not hasattr( self.descriptorCalculator, "amp_read_area"):
                key = "key_"+"amp_read_area"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.amp_read_area(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
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
            return self.descriptorCalculator.azimuth(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
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
            return self.descriptorCalculator.camera(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def cass_rotator_pa(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "cass_rotator_pa")
            if not hasattr( self.descriptorCalculator, "cass_rotator_pa"):
                key = "key_"+"cass_rotator_pa"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.cass_rotator_pa(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def central_wavelength(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "central_wavelength")
            if not hasattr( self.descriptorCalculator, "central_wavelength"):
                key = "key_"+"central_wavelength"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.central_wavelength(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def coadds(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "coadds")
            if not hasattr( self.descriptorCalculator, "coadds"):
                key = "key_"+"coadds"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.coadds(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def data_label(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "data_label")
            if not hasattr( self.descriptorCalculator, "data_label"):
                key = "key_"+"data_label"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.data_label(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def data_section(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "data_section")
            if not hasattr( self.descriptorCalculator, "data_section"):
                key = "key_"+"data_section"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.data_section(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
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
            return self.descriptorCalculator.dec(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def detector_section(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "detector_section")
            if not hasattr( self.descriptorCalculator, "detector_section"):
                key = "key_"+"detector_section"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.detector_section(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def detector_x_bin(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "detector_x_bin")
            if not hasattr( self.descriptorCalculator, "detector_x_bin"):
                key = "key_"+"detector_x_bin"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.detector_x_bin(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def detector_y_bin(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "detector_y_bin")
            if not hasattr( self.descriptorCalculator, "detector_y_bin"):
                key = "key_"+"detector_y_bin"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.detector_y_bin(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
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
            return self.descriptorCalculator.disperser(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def dispersion(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "dispersion")
            if not hasattr( self.descriptorCalculator, "dispersion"):
                key = "key_"+"dispersion"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.dispersion(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def dispersion_axis(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "dispersion_axis")
            if not hasattr( self.descriptorCalculator, "dispersion_axis"):
                key = "key_"+"dispersion_axis"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.dispersion_axis(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
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
            return self.descriptorCalculator.elevation(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def exposure_time(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "exposure_time")
            if not hasattr( self.descriptorCalculator, "exposure_time"):
                key = "key_"+"exposure_time"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.exposure_time(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def filter_id(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "filter_id")
            if not hasattr( self.descriptorCalculator, "filter_id"):
                key = "key_"+"filter_id"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.filter_id(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def filter_name(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "filter_name")
            if not hasattr( self.descriptorCalculator, "filter_name"):
                key = "key_"+"filter_name"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.filter_name(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def focal_plane_mask(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "focal_plane_mask")
            if not hasattr( self.descriptorCalculator, "focal_plane_mask"):
                key = "key_"+"focal_plane_mask"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.focal_plane_mask(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
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
            return self.descriptorCalculator.gain(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def gain_mode(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "gain_mode")
            if not hasattr( self.descriptorCalculator, "gain_mode"):
                key = "key_"+"gain_mode"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.gain_mode(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
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
            return self.descriptorCalculator.instrument(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def local_time(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "local_time")
            if not hasattr( self.descriptorCalculator, "local_time"):
                key = "key_"+"local_time"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.local_time(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def mdf_row_id(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "mdf_row_id")
            if not hasattr( self.descriptorCalculator, "mdf_row_id"):
                key = "key_"+"mdf_row_id"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.mdf_row_id(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def non_linear_level(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "non_linear_level")
            if not hasattr( self.descriptorCalculator, "non_linear_level"):
                key = "key_"+"non_linear_level"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.non_linear_level(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
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
            return self.descriptorCalculator.object(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def observation_class(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "observation_class")
            if not hasattr( self.descriptorCalculator, "observation_class"):
                key = "key_"+"observation_class"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.observation_class(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def observation_epoch(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "observation_epoch")
            if not hasattr( self.descriptorCalculator, "observation_epoch"):
                key = "key_"+"observation_epoch"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.observation_epoch(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def observation_id(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "observation_id")
            if not hasattr( self.descriptorCalculator, "observation_id"):
                key = "key_"+"observation_id"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.observation_id(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def observation_mode(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "observation_mode")
            if not hasattr( self.descriptorCalculator, "observation_mode"):
                key = "key_"+"observation_mode"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.observation_mode(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def observation_type(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "observation_type")
            if not hasattr( self.descriptorCalculator, "observation_type"):
                key = "key_"+"observation_type"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.observation_type(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
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
            return self.descriptorCalculator.observer(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def pixel_scale(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "pixel_scale")
            if not hasattr( self.descriptorCalculator, "pixel_scale"):
                key = "key_"+"pixel_scale"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.pixel_scale(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def program_id(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "program_id")
            if not hasattr( self.descriptorCalculator, "program_id"):
                key = "key_"+"program_id"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.program_id(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def pupil_mask(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "pupil_mask")
            if not hasattr( self.descriptorCalculator, "pupil_mask"):
                key = "key_"+"pupil_mask"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.pupil_mask(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
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
            return self.descriptorCalculator.ra(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def raw_bg(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "raw_bg")
            if not hasattr( self.descriptorCalculator, "raw_bg"):
                key = "key_"+"raw_bg"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.raw_bg(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def raw_cc(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "raw_cc")
            if not hasattr( self.descriptorCalculator, "raw_cc"):
                key = "key_"+"raw_cc"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.raw_cc(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def raw_gemini_qa(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "raw_gemini_qa")
            if not hasattr( self.descriptorCalculator, "raw_gemini_qa"):
                key = "key_"+"raw_gemini_qa"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.raw_gemini_qa(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def raw_iq(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "raw_iq")
            if not hasattr( self.descriptorCalculator, "raw_iq"):
                key = "key_"+"raw_iq"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.raw_iq(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def raw_pi_requirement(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "raw_pi_requirement")
            if not hasattr( self.descriptorCalculator, "raw_pi_requirement"):
                key = "key_"+"raw_pi_requirement"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.raw_pi_requirement(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def raw_wv(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "raw_wv")
            if not hasattr( self.descriptorCalculator, "raw_wv"):
                key = "key_"+"raw_wv"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.raw_wv(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def read_mode(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "read_mode")
            if not hasattr( self.descriptorCalculator, "read_mode"):
                key = "key_"+"read_mode"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.read_mode(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def read_noise(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "read_noise")
            if not hasattr( self.descriptorCalculator, "read_noise"):
                key = "key_"+"read_noise"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.read_noise(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def read_speed_mode(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "read_speed_mode")
            if not hasattr( self.descriptorCalculator, "read_speed_mode"):
                key = "key_"+"read_speed_mode"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.read_speed_mode(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def saturation_level(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "saturation_level")
            if not hasattr( self.descriptorCalculator, "saturation_level"):
                key = "key_"+"saturation_level"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.saturation_level(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
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
            return self.descriptorCalculator.ssa(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
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
            return self.descriptorCalculator.telescope(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def ut_date(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "ut_date")
            if not hasattr( self.descriptorCalculator, "ut_date"):
                key = "key_"+"ut_date"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.ut_date(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def ut_time(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "ut_time")
            if not hasattr( self.descriptorCalculator, "ut_time"):
                key = "key_"+"ut_time"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.ut_time(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def wavefront_sensor(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "wavefront_sensor")
            if not hasattr( self.descriptorCalculator, "wavefront_sensor"):
                key = "key_"+"wavefront_sensor"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.wavefront_sensor(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def wavelength_reference_pixel(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "wavelength_reference_pixel")
            if not hasattr( self.descriptorCalculator, "wavelength_reference_pixel"):
                key = "key_"+"wavelength_reference_pixel"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.wavelength_reference_pixel(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def well_depth_mode(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "well_depth_mode")
            if not hasattr( self.descriptorCalculator, "well_depth_mode"):
                key = "key_"+"well_depth_mode"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.well_depth_mode(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def x_offset(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "x_offset")
            if not hasattr( self.descriptorCalculator, "x_offset"):
                key = "key_"+"x_offset"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.x_offset(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def y_offset(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "y_offset")
            if not hasattr( self.descriptorCalculator, "y_offset"):
                key = "key_"+"y_offset"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.y_offset(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def release_date(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "release_date")
            if not hasattr( self.descriptorCalculator, "release_date"):
                key = "key_"+"release_date"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.release_date(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.noneMsg = str(sys.exc_info()[1])
                return None
    
    def qa_state(self, **args):
        try:
            self._lazyloadCalculator()
            #print hasattr( self.descriptorCalculator, "qa_state")
            if not hasattr( self.descriptorCalculator, "qa_state"):
                key = "key_"+"qa_state"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    return self.phuHeader(SDKD.globalStdkeyDict[key])
            return self.descriptorCalculator.qa_state(self, **args)
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

