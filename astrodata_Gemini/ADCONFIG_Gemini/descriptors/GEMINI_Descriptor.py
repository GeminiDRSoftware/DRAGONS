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
        hdu = dataset.hdulist
        airmass = hdu[0].header[globalStdkeyDict['key_airmass']]
        
        if airmass < 0.0:
            ret_airmass = None
        else:
            ret_airmass = float(airmass)
        
        return ret_airmass
    
    def cass_rotator_pa(self, dataset, **args):
        """
        Return the cassegrain rotator position angle value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: float
        @return: the crpa for the observation
        """
        hdu = dataset.hdulist
        crpa = hdu[0].header[globalStdkeyDict['key_cass_rotator_pa']]

        if (crpa < -999.0) or (crpa > 999.0):
            ret_crpa = None
        else:
            ret_crpa = float(crpa)

        return ret_crpa



    def data_label(self, dataset, **args):
        """
        Return the data_label value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the DHS data label of the observation
        """
        hdu = dataset.hdulist
        data_label = hdu[0].header[globalStdkeyDict['key_data_label']]
        
        ret_data_label = str(data_label)
        
        return ret_data_label
    
    def detector_x_bin(self, dataset, **args):
        """
        Return the detector_x_bin value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the binning of the detector x-axis
        """
        ret_detector_x_bin = int(1)
        
        return ret_detector_x_bin
    
    def detector_y_bin(self, dataset, **args):
        """
        Return the detector_y_bin value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: integer
        @return: the binning of the detector y-axis
        """
        ret_detector_y_bin = int(1)
        
        return ret_detector_y_bin
    
    def local_time(self, dataset, **args):
        """
        Return the local_time value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the local time at the start of the observation (HH:MM:SS.S)
        """
        hdu = dataset.hdulist
        local_time = hdu[0].header[globalStdkeyDict['key_local_time']]
        
        # Validate the result.
        # The assumption is that the standard mandates HH:MM:SS[.S] 
        # We don't enforce the number of decimal places
        # These are somewhat basic checks, it's not completely rigorous
        # Note that seconds can be > 59 when leap seconds occurs
        
        if re.match('^([012]\d)(:)([012345]\d)(:)(\d\d\.?\d*)$', local_time):
            ret_local_time = str(local_time)
        else:
            ret_local_time = None
        
        return ret_local_time
    
    def observation_id(self, dataset, **args):
        """
        Return the observation_id for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the observation ID / data label of the observation
        """
        hdu = dataset.hdulist
        observation_id = hdu[0].header[globalStdkeyDict['key_observation_id']]
        
        ret_observation_id = str(observation_id)
        
        return ret_observation_id
    
    def program_id(self, dataset, **args):
        """
        Return the program_id value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @return: the Gemini science program ID of the observation
        """
        hdu = dataset.hdulist
        program_id = hdu[0].header[globalStdkeyDict['key_program_id']]
        
        ret_program_id = str(program_id)
        
        return ret_program_id
    
    def qa_state(self, dataset, **args):
        """
        Return the qa_state for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the QA state for the observation
        """
        hdu = dataset.hdulist
        rawpireq = hdu[0].header[stdkeyDictGEMINI['key_raw_pi_requirements_met']]
        rawgemqa = hdu[0].header[stdkeyDictGEMINI['key_raw_gemini_qa']]
        
        # Calculate the derived QA state
        qa_state = "%s:%s" % (rawpireq, rawgemqa)
        if rawpireq == 'UNKNOWN' and rawgemqa == 'UNKNOWN':
            qa_state = 'Undefined'
        if rawpireq == 'YES' and rawgemqa == 'USABLE':
            qa_state = 'Pass'
        if rawpireq == 'NO' and rawgemqa == 'USABLE':
            qa_state = 'Usable'
        if rawpireq == 'NO' and rawgemqa == 'BAD':
            qa_state = 'Fail'
        if rawpireq == 'CHECK' and rawgemqa == 'CHECK':
            qa_state = 'CHECK'
        
        return qa_state
    
    def ut_time(self, dataset, **args):
        """
        Return the ut_time value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the UT at the start of the observation (HH:MM:SS.S)
        """
        hdu = dataset.hdulist
        ut_time = hdu[0].header[globalStdkeyDict['key_ut_time']]
        
        # Validate the result.
        # The assumption is that the standard mandates HH:MM:SS[.S] 
        # We don't enforce the number of decimal places
        # These are somewhat basic checks, it's not completely rigorous
        # Note that seconds can be > 59 when leap seconds occurs
        
        if re.match('^([012]\d)(:)([012345]\d)(:)(\d\d\.?\d*)$', ut_time):
            ret_ut_time = str(ut_time)
        else:
            ret_ut_time = None
        
        return ret_ut_time
    
    def wavefront_sensor(self, dataset, **args):
        """
        Return the wavefront_sensor value for GEMINI data
        @param dataset: the data set
        @type dataset: AstroData
        @rtype: string
        @returns: the wavefront sensor used for the observation
        """
        hdu = dataset.hdulist
        aowfs = hdu[0].header[stdkeyDictGEMINI['key_aowfs']]
        oiwfs = hdu[0].header[stdkeyDictGEMINI['key_oiwfs']]
        pwfs1 = hdu[0].header[stdkeyDictGEMINI['key_pwfs1']]
        pwfs2 = hdu[0].header[stdkeyDictGEMINI['key_pwfs2']]
        
        wavefront_sensors = []
        if aowfs == 'guiding':
            wavefront_sensors.append('AOWFS')
        if oiwfs == 'guiding':
            wavefront_sensors.append('OIWFS')
        if pwfs1 == 'guiding':
            wavefront_sensors.append('PWFS1')
        if pwfs2 == 'guiding':
            wavefront_sensors.append('PWFS2')
        
        if len(wavefront_sensors) == 0:
            ret_wavefront_sensor = None
        else:
            ret_wavefront_sensor = str('&'.join(wavefront_sensors))
        
        return ret_wavefront_sensor

