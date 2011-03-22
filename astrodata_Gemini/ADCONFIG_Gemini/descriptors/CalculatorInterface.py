import sys
import StandardDescriptorKeyDict as SDKD
from astrodata import Descriptors
from astrodata import Errors

class CalculatorInterface:

    descriptorCalculator = None

    def airmass(self, **args):
        """
        Return the airmass value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: float
        :return: the mean airmass of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "airmass")
            if not hasattr(self.descriptorCalculator, "airmass"):
                key = "key_"+"airmass"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.airmass(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def amp_read_area(self, **args):
        """
        Return the amp_read_area value
        :param dataset: the data set
        :type dataset: AstroData
        :param asDict: set to True to return a dictionary, where the number of 
                       dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type asDict: Python boolean
        :rtype: dictionary containing one or more string(s)
        :return: the composite string containing the name of the detector
                 amplifier (ampname) and the readout area of the CCD (detsec) 
                 used for the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "amp_read_area")
            if not hasattr(self.descriptorCalculator, "amp_read_area"):
                key = "key_"+"amp_read_area"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.amp_read_area(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def azimuth(self, **args):
        """
        Return the azimuth value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: float
        :return: the azimuth (in degrees between 0 and 360) of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "azimuth")
            if not hasattr(self.descriptorCalculator, "azimuth"):
                key = "key_"+"azimuth"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.azimuth(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def camera(self, **args):
        """
        Return the camera value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the camera used for the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "camera")
            if not hasattr(self.descriptorCalculator, "camera"):
                key = "key_"+"camera"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.camera(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def cass_rotator_pa(self, **args):
        """
        Return the cass_rotator_pa value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: float
        :return: the cassegrain rotator position angle (in degrees between -360
                 and 360) of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "cass_rotator_pa")
            if not hasattr(self.descriptorCalculator, "cass_rotator_pa"):
                key = "key_"+"cass_rotator_pa"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.cass_rotator_pa(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def central_wavelength(self, **args):
        """
        Return the central_wavelength value
        :param dataset: the data set
        :type dataset: AstroData
        :param asMicrometers: set to True to return the central_wavelength 
                              value in units of Micrometers
        :type asMicrometers: Python boolean
        :param asNanometers: set to True to return the central_wavelength 
                             value in units of Nanometers
        :type asNanometers: Python boolean
        :param asAngstroms: set to True to return the central_wavelength 
                            value in units of Angstroms
        :type asAngstroms: Python boolean
        :param asDict: set to True to return a dictionary, where the number of 
                       dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type asDict: Python boolean
        :rtype: dictionary containing one or more float(s)
        :return: the central wavelength (in meters by default) of the 
                 observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "central_wavelength")
            if not hasattr(self.descriptorCalculator, "central_wavelength"):
                key = "key_"+"central_wavelength"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.central_wavelength(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def coadds(self, **args):
        """
        Return the coadds value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: integer
        :return: the number of coadds used for the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "coadds")
            if not hasattr(self.descriptorCalculator, "coadds"):
                key = "key_"+"coadds"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.coadds(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def data_label(self, **args):
        """
        Return the data_label value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the data label of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "data_label")
            if not hasattr(self.descriptorCalculator, "data_label"):
                key = "key_"+"data_label"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.data_label(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def data_section(self, **args):
        """
        Return the data_section value
        :param dataset: the data set
        :type dataset: AstroData
        :param pretty: set to True to return a human meaningful data_section 
                       value in the form [x1:x2,y1:y2] that uses 1-based 
                       indexing
        :type pretty: Python boolean
        :param asDict: set to True to return a dictionary, where the number of 
                       dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type asDict: Python boolean
        :rtype: dictionary containing one or more tuple(s) that use 0-based 
                indexing in the form (x1 - 1, x2 - 1, y1 - 1, y2 - 1) as 
                default, or one or more string(s) that use 1-based indexing in 
                the form [x1:x2,y1:y2] if pretty=True, where x1, x2, y1 and y2 
                are integers.
        :return: the section of the data of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "data_section")
            if not hasattr(self.descriptorCalculator, "data_section"):
                key = "key_"+"data_section"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.data_section(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def dec(self, **args):
        """
        Return the dec value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: float
        :return: the declination (in decimal degrees) of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "dec")
            if not hasattr(self.descriptorCalculator, "dec"):
                key = "key_"+"dec"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.dec(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def decker(self, **args):
        """
        Return the decker value
        :param dataset: the data set
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned decker value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       decker value
        :type pretty: Python boolean
        :rtype: string
        :return: the decker position used for the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "decker")
            if not hasattr(self.descriptorCalculator, "decker"):
                key = "key_"+"decker"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.decker(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def detector_section(self, **args):
        """
        Return the detector_section value
        :param dataset: the data set
        :type dataset: AstroData
        :param pretty: set to True to return a human meaningful 
                       detector_section value in the form [x1:x2,y1:y2] that 
                       uses 1-based indexing
        :type pretty: Python boolean
        :param asDict: set to True to return a dictionary, where the number of 
                       dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type asDict: Python boolean
        :rtype: dictionary containing one or more tuple(s) that use 0-based 
                indexing in the form (x1 - 1, x2 - 1, y1 - 1, y2 - 1) as 
                default, or one or more string(s) that use 1-based indexing in 
                the form [x1:x2,y1:y2] if pretty=True, where x1, x2, y1 and y2 
                are integers.
        :return: the detector section of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "detector_section")
            if not hasattr(self.descriptorCalculator, "detector_section"):
                key = "key_"+"detector_section"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.detector_section(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def detector_x_bin(self, **args):
        """
        Return the detector_x_bin value
        :param dataset: the data set
        :type dataset: AstroData
        :param asDict: set to True to return a dictionary, where the number of 
                       dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type asDict: Python boolean
        :rtype: dictionary containing one or more integer(s)
        :return: the binning of the x-axis of the detector used for the 
                 observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "detector_x_bin")
            if not hasattr(self.descriptorCalculator, "detector_x_bin"):
                key = "key_"+"detector_x_bin"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.detector_x_bin(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def detector_y_bin(self, **args):
        """
        Return the detector_y_bin value
        :param dataset: the data set
        :type dataset: AstroData
        :param asDict: set to True to return a dictionary, where the number of 
                       dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type asDict: Python boolean
        :rtype: dictionary containing one or more integer(s)
        :return: the binning of the y-axis of the detector used for the 
                 observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "detector_y_bin")
            if not hasattr(self.descriptorCalculator, "detector_y_bin"):
                key = "key_"+"detector_y_bin"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.detector_y_bin(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def disperser(self, **args):
        """
        Return the disperser value
        :param dataset: the data set
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned disperser value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       disperser value
        :type pretty: Python boolean
        :rtype: string
        :return: the disperser used for the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "disperser")
            if not hasattr(self.descriptorCalculator, "disperser"):
                key = "key_"+"disperser"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.disperser(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def dispersion(self, **args):
        """
        Return the dispersion value
        :param dataset: the data set
        :type dataset: AstroData
        :param asMicrometers: set to True to return the dispersion 
                              value in units of Micrometers
        :type asMicrometers: Python boolean
        :param asNanometers: set to True to return the dispersion 
                             value in units of Nanometers
        :type asNanometers: Python boolean
        :param asAngstroms: set to True to return the dispersion 
                            value in units of Angstroms
        :type asAngstroms: Python boolean
        :param asDict: set to True to return a dictionary, where the number of 
                       dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type asDict: Python boolean
        :rtype: dictionary containing one or more float(s)
        :return: the dispersion (in meters per pixel by default) of the 
                 observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "dispersion")
            if not hasattr(self.descriptorCalculator, "dispersion"):
                key = "key_"+"dispersion"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.dispersion(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def dispersion_axis(self, **args):
        """
        Return the dispersion_axis value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: integer
        :return: the dispersion axis (x = 1; y = 2; z = 3) of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "dispersion_axis")
            if not hasattr(self.descriptorCalculator, "dispersion_axis"):
                key = "key_"+"dispersion_axis"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.dispersion_axis(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def elevation(self, **args):
        """
        Return the elevation value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: float
        :return: the elevation (in degrees) of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "elevation")
            if not hasattr(self.descriptorCalculator, "elevation"):
                key = "key_"+"elevation"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.elevation(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def exposure_time(self, **args):
        """
        Return the exposure_time value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: float
        :return: the total exposure time (in seconds) of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "exposure_time")
            if not hasattr(self.descriptorCalculator, "exposure_time"):
                key = "key_"+"exposure_time"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.exposure_time(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def filter_name(self, **args):
        """
        Return the filter_name value
        :param dataset: the data set
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned filter_name value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       filter_name value
        :type pretty: Python boolean
        :rtype: string
        :return: the unique, sorted filter name idenifier string used for the 
                 observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "filter_name")
            if not hasattr(self.descriptorCalculator, "filter_name"):
                key = "key_"+"filter_name"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.filter_name(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def focal_plane_mask(self, **args):
        """
        Return the focal_plane_mask value
        :param dataset: the data set
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned focal_plane_mask value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       focal_plane_mask value
        :type pretty: Python boolean
        :rtype: string
        :return: the focal plane mask used for the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "focal_plane_mask")
            if not hasattr(self.descriptorCalculator, "focal_plane_mask"):
                key = "key_"+"focal_plane_mask"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.focal_plane_mask(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def gain(self, **args):
        """
        Return the gain value
        :param dataset: the data set
        :type dataset: AstroData
        :param asDict: set to True to return a dictionary, where the number of 
                       dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type asDict: Python boolean
        :rtype: dictionary containing one or more float(s)
        :return: the gain (in electrons per ADU) of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "gain")
            if not hasattr(self.descriptorCalculator, "gain"):
                key = "key_"+"gain"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.gain(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def grating(self, **args):
        """
        Return the grating value
        :param dataset: the data set
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned grating value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       grating value
        :type pretty: Python boolean
        :rtype: string
        :return: the grating used for the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "grating")
            if not hasattr(self.descriptorCalculator, "grating"):
                key = "key_"+"grating"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.grating(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def gain_setting(self, **args):
        """
        Return the gain_setting value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the gain setting of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "gain_setting")
            if not hasattr(self.descriptorCalculator, "gain_setting"):
                key = "key_"+"gain_setting"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.gain_setting(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def instrument(self, **args):
        """
        Return the instrument value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the instrument used for the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "instrument")
            if not hasattr(self.descriptorCalculator, "instrument"):
                key = "key_"+"instrument"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.instrument(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def local_time(self, **args):
        """
        Return the local_time value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the local time (in HH:MM:SS.S) at the start of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "local_time")
            if not hasattr(self.descriptorCalculator, "local_time"):
                key = "key_"+"local_time"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.local_time(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def mdf_row_id(self, **args):
        """
        Return the mdf_row_id value
        :param dataset: the data set
        :type dataset: AstroData
        :param asDict: set to True to return a dictionary, where the number of 
                       dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type asDict: Python boolean
        :rtype: dictionary containing one or more integer(s)
        :return: the corresponding reference row in the MDF
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "mdf_row_id")
            if not hasattr(self.descriptorCalculator, "mdf_row_id"):
                key = "key_"+"mdf_row_id"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.mdf_row_id(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def nod_count(self, **args):
        """
        Return the nod_count value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: integer
        :return: the number of nod and shuffle cycles in the nod and shuffle 
                 observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "nod_count")
            if not hasattr(self.descriptorCalculator, "nod_count"):
                key = "key_"+"nod_count"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.nod_count(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def nod_pixels(self, **args):
        """
        Return the nod_pixels value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: integer
        :return: the number of pixel rows the charge is shuffled by in the nod 
                 and shuffle observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "nod_pixels")
            if not hasattr(self.descriptorCalculator, "nod_pixels"):
                key = "key_"+"nod_pixels"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.nod_pixels(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def non_linear_level(self, **args):
        """
        Return the non_linear_level value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: integer
        :return: the non linear level in the raw images (in ADU) of the 
                 observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "non_linear_level")
            if not hasattr(self.descriptorCalculator, "non_linear_level"):
                key = "key_"+"non_linear_level"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.non_linear_level(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def object(self, **args):
        """
        Return the object value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the name of the target object observed
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "object")
            if not hasattr(self.descriptorCalculator, "object"):
                key = "key_"+"object"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.object(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def observation_class(self, **args):
        """
        Return the observation_class value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the class (either 'science', 'progCal', 'partnerCal', 'acq', 
                 'acqCal' or 'dayCal') of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "observation_class")
            if not hasattr(self.descriptorCalculator, "observation_class"):
                key = "key_"+"observation_class"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.observation_class(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def observation_epoch(self, **args):
        """
        Return the observation_epoch value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the epoch (in years) at the start of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "observation_epoch")
            if not hasattr(self.descriptorCalculator, "observation_epoch"):
                key = "key_"+"observation_epoch"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.observation_epoch(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def observation_id(self, **args):
        """
        Return the observation_id value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the ID (e.g., GN-2011A-Q-123-45) of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "observation_id")
            if not hasattr(self.descriptorCalculator, "observation_id"):
                key = "key_"+"observation_id"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.observation_id(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def observation_type(self, **args):
        """
        Return the observation_type value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the type (either 'OBJECT', 'DARK', 'FLAT', 'ARC', 'BIAS' or 
                 'MASK') of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "observation_type")
            if not hasattr(self.descriptorCalculator, "observation_type"):
                key = "key_"+"observation_type"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.observation_type(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def pixel_scale(self, **args):
        """
        Return the pixel_scale value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: float
        :return: the pixel scale (in arcsec per pixel) of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "pixel_scale")
            if not hasattr(self.descriptorCalculator, "pixel_scale"):
                key = "key_"+"pixel_scale"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.pixel_scale(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def prism(self, **args):
        """
        Return the prism value
        :param dataset: the data set
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned prism value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       prism value
        :type pretty: Python boolean
        :rtype: string
        :return: the prism used for the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "prism")
            if not hasattr(self.descriptorCalculator, "prism"):
                key = "key_"+"prism"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.prism(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def program_id(self, **args):
        """
        Return the program_id value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the Gemini program ID (e.g., GN-2011A-Q-123) of the 
                 observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "program_id")
            if not hasattr(self.descriptorCalculator, "program_id"):
                key = "key_"+"program_id"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.program_id(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def pupil_mask(self, **args):
        """
        Return the pupil_mask value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the pupil mask used for the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "pupil_mask")
            if not hasattr(self.descriptorCalculator, "pupil_mask"):
                key = "key_"+"pupil_mask"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.pupil_mask(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def qa_state(self, **args):
        """
        Return the qa_state value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the quality assessment state (either 'Undefined', 'Pass', 
                 'Usable', 'Fail' or 'CHECK') of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "qa_state")
            if not hasattr(self.descriptorCalculator, "qa_state"):
                key = "key_"+"qa_state"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.qa_state(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def ra(self, **args):
        """
        Return the ra value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: float
        :return: the Right Ascension (in decimal degrees) of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "ra")
            if not hasattr(self.descriptorCalculator, "ra"):
                key = "key_"+"ra"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.ra(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def raw_bg(self, **args):
        """
        Return the raw_bg value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the raw background (either '20-percentile', '50-percentile', 
                 '80-percentile' or 'Any') of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "raw_bg")
            if not hasattr(self.descriptorCalculator, "raw_bg"):
                key = "key_"+"raw_bg"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.raw_bg(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def raw_cc(self, **args):
        """
        Return the raw_cc value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the raw cloud cover (either '50-percentile', '70-percentile', 
                 '80-percentile', '90-percentile' or 'Any') of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "raw_cc")
            if not hasattr(self.descriptorCalculator, "raw_cc"):
                key = "key_"+"raw_cc"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.raw_cc(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def raw_iq(self, **args):
        """
        Return the raw_iq value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the raw image quality (either '20-percentile', 
                 '70-percentile', '85-percentile' or 'Any') of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "raw_iq")
            if not hasattr(self.descriptorCalculator, "raw_iq"):
                key = "key_"+"raw_iq"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.raw_iq(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def raw_wv(self, **args):
        """
        Return the raw_wv value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the raw water vapour (either '20-percentile', 
                 '50-percentile', '80-percentile' or 'Any') of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "raw_wv")
            if not hasattr(self.descriptorCalculator, "raw_wv"):
                key = "key_"+"raw_wv"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.raw_wv(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def read_mode(self, **args):
        """
        Return the read_mode value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the read mode (either 'Very Faint Objects', 
                 'Faint Object(s)', 'Medium Object', 'Bright Object(s)', 
                 'Very Bright Object', 'Low Background', 'Medium Background', 
                 'High Background' or 'Invalid') of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "read_mode")
            if not hasattr(self.descriptorCalculator, "read_mode"):
                key = "key_"+"read_mode"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.read_mode(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def read_noise(self, **args):
        """
        Return the read_noise value
        :param dataset: the data set
        :type dataset: AstroData
        :param asDict: set to True to return a dictionary, where the number of 
                       dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type asDict: Python boolean
        :rtype: dictionary containing one or more float(s)
        :return: the estimated readout noise (in electrons) of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "read_noise")
            if not hasattr(self.descriptorCalculator, "read_noise"):
                key = "key_"+"read_noise"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.read_noise(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def read_speed_setting(self, **args):
        """
        Return the read_speed_setting value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the read speed setting (either 'fast' or 'slow') of the 
                 observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "read_speed_setting")
            if not hasattr(self.descriptorCalculator, "read_speed_setting"):
                key = "key_"+"read_speed_setting"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.read_speed_setting(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def saturation_level(self, **args):
        """
        Return the saturation_level value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: integer
        :return: the saturation level in the raw images (in ADU) of the 
                 observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "saturation_level")
            if not hasattr(self.descriptorCalculator, "saturation_level"):
                key = "key_"+"saturation_level"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.saturation_level(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def slit(self, **args):
        """
        Return the slit value
        :param dataset: the data set
        :type dataset: AstroData
        :param stripID: set to True to remove the component ID from the 
                        returned slit value
        :type stripID: Python boolean
        :param pretty: set to True to return a human meaningful 
                       slit value
        :type pretty: Python boolean
        :rtype: string
        :return: the slit used for the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "slit")
            if not hasattr(self.descriptorCalculator, "slit"):
                key = "key_"+"slit"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.slit(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def telescope(self, **args):
        """
        Return the telescope value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the telescope used for the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "telescope")
            if not hasattr(self.descriptorCalculator, "telescope"):
                key = "key_"+"telescope"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.telescope(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def ut_date(self, **args):
        """
        Return the ut_date value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: datatime.date
        :return: the UT date at the start of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "ut_date")
            if not hasattr(self.descriptorCalculator, "ut_date"):
                key = "key_"+"ut_date"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.ut_date(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def ut_datetime(self, **args):
        """
        Return the ut_datetime value
        This descriptor attempts to figure out the datetime even when the
        headers are malformed or not present. It tries just about every header
        combination that could allow it to determine an appropriate datetime
        for the file in question. This makes it somewhat specific to Gemini
        data, in that the headers it looks at, and the assumptions it makes in
        trying to parse their values, are those known to occur in Gemini data.
        Note that some of the early gemini data, and that taken from lower
        level engineering interfaces, lack standard headers. Also the format
        and occurence of various headers has changed over time, even on the
        same instrument. If strict is set to True, the date or time are
        determined from valid FITS keywords. If it cannot be determined, None
        is returned. If dateonly or timeonly are set to True, then a
        datetime.date object or datetime.time object, respectively, is
        returned, containing only the date or time, respectively. These two
        interplay with strict in the sense that if strict is set to True and a
        date can be determined but not a time, then this function will return
        None unless the dateonly flag is set, in which case it will return the
        valid date. The dateonly and timeonly flags are intended for use by
        the ut_date and ut_time descriptors.
        :param dataset: the data set
        :type dataset: AstroData
        :param strict: set to True to not try to guess the date or time
        :type strict: Python boolean
        :param dateonly: set to True to return a datetime.date
        :type dateonly: Python boolean
        :param timeonly: set to True to return a datetime.time
        :param timeonly: Python boolean
        :rtype: datetime.datetime (dateonly=False and timeonly=False)
        :rtype: datetime.time (timeonly=True)
        :rtype: datetime.date (dateonly=True)
        :return: the UT date and time at the start of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "ut_datetime")
            if not hasattr(self.descriptorCalculator, "ut_datetime"):
                key = "key_"+"ut_datetime"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.ut_datetime(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def ut_time(self, **args):
        """
        Return the ut_time value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: datatime.time
        :return: the UT time at the start of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "ut_time")
            if not hasattr(self.descriptorCalculator, "ut_time"):
                key = "key_"+"ut_time"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.ut_time(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def wavefront_sensor(self, **args):
        """
        Return the wavefront_sensor value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the wavefront sensor (either 'AOWFS', 'OIWFS', 'PWFS1', 
                 'PWFS2', some combination in alphebetic order separated with 
                 an ampersand or None) used for the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "wavefront_sensor")
            if not hasattr(self.descriptorCalculator, "wavefront_sensor"):
                key = "key_"+"wavefront_sensor"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.wavefront_sensor(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def wavelength_reference_pixel(self, **args):
        """
        Return the wavelength_reference_pixel value
        :param dataset: the data set
        :type dataset: AstroData
        :param asDict: set to True to return a dictionary, where the number of 
                       dictionary elements equals the number of pixel data 
                       extensions in the image. The key of the dictionary is 
                       an (EXTNAME, EXTVER) tuple, if available. Otherwise, 
                       the key is the integer index of the extension.
        :type asDict: Python boolean
        :rtype: dictionary containing one or more float(s)
        :return: the reference pixel of the central wavelength of the 
                 observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "wavelength_reference_pixel")
            if not hasattr(self.descriptorCalculator, "wavelength_reference_pixel"):
                key = "key_"+"wavelength_reference_pixel"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.wavelength_reference_pixel(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def well_depth_setting(self, **args):
        """
        Return the well_depth_setting value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: string
        :return: the well depth setting (either 'Shallow', 'Deep' or 
                 'Invalid') of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "well_depth_setting")
            if not hasattr(self.descriptorCalculator, "well_depth_setting"):
                key = "key_"+"well_depth_setting"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.well_depth_setting(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def x_offset(self, **args):
        """
        Return the x_offset value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: float
        :return: the x offset of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "x_offset")
            if not hasattr(self.descriptorCalculator, "x_offset"):
                key = "key_"+"x_offset"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.x_offset(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
    def y_offset(self, **args):
        """
        Return the y_offset value
        :param dataset: the data set
        :type dataset: AstroData
        :rtype: float
        :return: the y offset of the observation
        """
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "y_offset")
            if not hasattr(self.descriptorCalculator, "y_offset"):
                key = "key_"+"y_offset"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.y_offset(self, **args)
            if "asString" in args and args["asString"]==True:
                from datetime import datetime
                from astrodata.adutils.gemutil import stdDateString
                if isinstance(retval, datetime):
                    retval = stdDateString(retval)
                else:
                    retval = str(retval)
            return retval
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
            #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()
                return None
    
# UTILITY FUNCTIONS, above are descriptor thunks            
    def _lazyloadCalculator(self, **args):
        '''Function to put at top of all descriptor members
        to ensure the descriptor is loaded.  This way we avoid
        loading it if it is not needed.'''
        if self.descriptorCalculator == None:
            self.descriptorCalculator = Descriptors.getCalculator(self, **args)

