from astrodata import Lookups
from astrodata import Descriptors

from astrodata.Calculator import Calculator

import GemCalcUtil

import re

from StandardDescriptorKeyDict import globalStdkeyDict
from StandardGEMINIKeyDict import stdkeyDictGEMINI
from Generic_Descriptor import Generic_DescriptorCalc

class GEMINI_DescriptorCalc(Generic_DescriptorCalc):
    
    def airmass(self, dataset, **args):
        """
        Return the airmass value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the mean airmass for the observation
        """
        try:
            hdu = dataset.hdulist
            ret_airmass = hdu[0].header[globalStdkeyDict['key_airmass']]
        
        except KeyError:
            return None
        
        try:
            ret_airmass = float(ret_airmass)
        except ValueError:
            return None

        if(ret_airmass < 0.0):
            return None

        return ret_airmass
    
    def data_label(self, dataset, **args):
        """
        Return the data_label value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the DHS data label of the observation
        """
        try:
            hdu = dataset.hdulist
            ret_data_label = hdu[0].header[globalStdkeyDict['key_data_label']]
        
        except KeyError:
            return None
        
        return str(ret_data_label)
    
    def observation_id(self, dataset, **args):
        """
        Return the observation_id for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the observation ID / data label of the observation
        """
        try:
            hdu = dataset.hdulist
            ret_observation_id = \
                hdu[0].header[globalStdkeyDict['key_observation_id']]
        
        except KeyError:
            return None
        
        return str(ret_observation_id)
    
    def program_id(self, dataset, **args):
        """
        Return the program_id value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the Gemini science program ID of the observation
        """
        try:
            hdu = dataset.hdulist
            ret_program_id = hdu[0].header[globalStdkeyDict['key_program_id']]
        
        except KeyError:
            return None
        
        return str(ret_program_id)
    
    def ut_time(self, dataset, **args):
        """
        Return the ut_time value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the UT at the start of the observation (HH:MM:SS.S)
        """
        try:
            hdu = dataset.hdulist
            ret_ut_time = hdu[0].header[globalStdkeyDict['key_ut_time']]
        
        except KeyError:
            return None

        # Validate the result.
        # The assumption is that the standard mandates HH:MM:SS[.S] 
        # We don't enforce the number of decimal places
        # These are somewhat basic checks, it's not completely rigorous
        # Note that seconds can be > 59 when leap seconds occurs

        if(re.match('^([012]\d)(:)([012345]\d)(:)(\d\d\.?\d*)$', ret_ut_time)):
            return ret_ut_time
        else:
            return None

    def local_time(self, dataset, **args):
        """
        Return the local_time value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the LT at the start of the observation (HH:MM:SS.S)
        """
        try:
            hdu = dataset.hdulist
            ret_local_time = hdu[0].header[globalStdkeyDict['key_local_time']]

        except KeyError:
            return None

        # Validate the result.
        # The assumption is that the standard mandates HH:MM:SS[.S] 
        # We don't enforce the number of decimal places
        # These are somewhat basic checks, it's not completely rigorous
        # Note that seconds can be > 59 when leap seconds occurs

        if(re.match('^([012]\d)(:)([012345]\d)(:)(\d\d\.?\d*)$', ret_local_time)):
            return ret_local_time
        else:
            return None

