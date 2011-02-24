from astrodata import Lookups
from astrodata import Descriptors
import re

from astrodata.Calculator import Calculator

import GemCalcUtil 

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardMICHELLEKeyDict import stdkeyDictMICHELLE
from GEMINI_Descriptor import GEMINI_DescriptorCalc

class MICHELLE_DescriptorCalc(GEMINI_DescriptorCalc):
    # Updating the global key dict with the local dict of this descriptor class
    globalStdkeyDict.update(stdkeyDictMICHELLE)
    
    def __init__(self):
        pass
    
    def central_wavelength(self, dataset, **args):
        """
        Return the central_wavelength value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the central wavelength (nanometers)
        """
        hdu = dataset.hdulist
        central_wavelength = \
            hdu[0].header[stdkeyDictMICHELLE['key_central_wavelength']]
        
        ret_central_wavelength = float(central_wavelength)
        if(ret_central_wavelength < 0.0):
            ret_central_wavelength = None
        
        return ret_central_wavelength
    
    def exposure_time(self, dataset, **args):
        """
        Return the exposure_time value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the total exposure time of the observation (seconds)
        """
        hdu = dataset.hdulist
        exposure = float(hdu[0].header[stdkeyDictMICHELLE['key_exposure']])
        numexpos = float(hdu[0].header[stdkeyDictMICHELLE['key_coadds']])
        numext = float(hdu[0].header[stdkeyDictMICHELLE['key_numext']])
        
        ret_exposure_time = float(exposure * numexpos * numext)
        
        return ret_exposure_time
    
    def filter_name(self, dataset, stripID=False, pretty=False, **args):
        """
        Return the filter_name value for MICHELLE
        @param dataset: the data set
        @type dataset: AstroData
        @param stripID: set to True to remove the component ID from the
        returned filter name
        @param pretty: set to True to return a meaningful filter name
        @rtype: string
        @return: the unique filter identifier string
        """
        # The Michelle filters don't have ID strings, so we just ignore the
        # stripID and pretty options
        hdu = dataset.hdulist
        filter = hdu[0].header[stdkeyDictMICHELLE['key_filter']]
        
        if filter == 'NBlock':
            ret_filter_name = 'blank'
        else:
            ret_filter_name = str(filter)
        
        return ret_filter_name
