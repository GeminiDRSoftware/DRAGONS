# The gemini_data_calculations module contains functions that calculate values
# from Gemini data

from datetime import datetime
import numpy as np
from time import strptime

from astrodata import Lookups
from astrodata.gemconstants import SCI
from gempy.gemini import gemini_metadata_utils as gmu

def get_bias_level(adinput=None, estimate=True):
    if estimate:
        ret_bias_level = _get_bias_level_estimate(adinput=adinput)
    else:
        ret_bias_level = _get_bias_level(adinput=adinput)
    
    return ret_bias_level
    
def _get_bias_level(adinput=None):
    """
    Determine the bias level value from GMOS data. The bias level is equal to
    the median of the overscan region
    """
    # Since this function accesses keywords in the headers of the pixel data
    # extensions, always construct a dictionary where the key of the dictionary
    # is an (EXTNAME, EXTVER) tuple 
    ret_bias_level = {}
    
    # Get the overscan section value of the science extensions using the
    # appropriate descriptor. Use as_dict() to return the value as a dictionary
    # rather than an object.
    overscan_section_dict = adinput[SCI].overscan_section().as_dict()
    
    if overscan_section_dict is not None:
        
        # The type of CCD determines the number of contaminated columns in the
        # overscan region. Get the pretty detector name value using the
        # appropriate descriptor.
        detector_name = adinput.detector_name(pretty=True)
        
        if detector_name == "EEV":
            nbiascontam = 4
        elif detector_name == "e2vDD":
            nbiascontam = 5
        elif detector_name == "Hamamastu":
            nbiascontam = 4
        else:
            nbiascontam = 4
        
        os_dict = overscan_section_dict.iteritems()
        for ext_name_ver, overscan_section in os_dict:
            
            # Don't include columns at edges
            if overscan_section[0] == 0:
                # Overscan region is on the left
                overscan_section[1] -= nbiascontam
                overscan_section[0] += 1
            else:
                # Overscan region is on the right
                overscan_section[0] += nbiascontam
                overscan_section[1] -= 1
            
            # Extract overscan data. In numpy arrays, y indices come first.
            overdata = adinput[ext_name_ver].data[
              overscan_section[2]:overscan_section[3],
              overscan_section[0]:overscan_section[1]]
            bias_level = np.median(overdata)
            
            # Update the dictionary with the bias level value
            ret_bias_level.update({ext_name_ver: bias_level})
        
        unique_values = set(ret_bias_level.values())
        if len(unique_values) == 1 and None in unique_values:
            # The bias level was not found for any of the pixel data extensions
            # (all the values in the dictionary are equal to None)
            ret_bias_level = None
    else:
        ret_bias_level = _get_bias_level_estimate(adinput=adinput)
    
    return ret_bias_level

def _get_bias_level_estimate(adinput=None):
    """
    Determine an estiamte of the bias level value from GMOS data.
    """
    # Since this function accesses keywords in the headers of the pixel data
    # extensions, always return a dictionary where the key of the dictionary is
    # an (EXTNAME, EXTVER) tuple 
    ret_bias_level = {}
    
    # Get the overscan value and the raw bias level from the header of each
    # pixel data extension as a dictionary where the key of the dictionary is
    # an ("*", EXTVER) tuple
    overscan_value_dict = gmu.get_key_value_dict(adinput, "OVERSCAN")
    raw_bias_level_dict = gmu.get_key_value_dict(adinput, "RAWBIAS")
    
    if overscan_value_dict is None:
        
        # If there is no overscan value for any extensions, use the raw bias
        # level value as the value for the bias level
        if raw_bias_level_dict is None:
            
            # If there is no raw bias level value for any extensions, use the
            # static bias levels from the lookup table as the value for the
            # bias level
            ret_bias_level = _get_static_bias_level(adinput=adinput)
        else:
            # Use the raw bias level value as the value for the bias level
            rbl_dict = raw_bias_level_dict.iteritems()
            for ext_name_ver, raw_bias_level in rbl_dict:
                
                if raw_bias_level is None:
                    # If the raw bias level does not exist for a given
                    # extension, use the static bias levels from the lookup
                    # table as the value for the bias level
                    bias_level = _get_static_bias_level_for_ext(
                      adinput=adinput[ext_name_ver])
                else:
                    bias_level = raw_bias_level
                
                # Update the dictionary with the bias level value
                ret_bias_level.update({ext_name_ver: bias_level})
    else:
        for ext_name_ver, overscan_value in overscan_value_dict.iteritems():
            if overscan_value is None:
                
                # If the overscan value does not exist for a given extension,
                # use the raw bias level value from the header as the value for
                # the bias level
                if raw_bias_level_dict is None:
                    
                    # If there is no raw bias level value for any extensions,
                    # use the static bias levels from the lookup table as the
                    # value for the bias level 
                    bias_level = _get_static_bias_level_for_ext(
                      adinput=adinput, ext_name_ver=ext_name_ver)
                else:
                    raw_bias_level = raw_bias_level_dict[ext_name_ver]
                    if raw_bias_level is None:
                        
                        # If the raw bias level does not exist for a given 
                        # extension, use the static bias levels from the lookup
                        # table as the value for the bias level 
                        bias_level = _get_static_bias_level_for_ext(
                          adinput=adinput, ext_name_ver=ext_name_ver)
                    else:
                        bias_level = raw_bias_level
            else:
                bias_level = overscan_value
            
            # Update the dictionary with the bias level value
            ret_bias_level.update({ext_name_ver: bias_level})
    
    unique_values = set(ret_bias_level.values())
    if len(unique_values) == 1 and None in unique_values:
        # The bias level was not found for any of the pixel data extensions
        # (all the values in the dictionary are equal to None)
        ret_bias_level = None
        
    return ret_bias_level

def _get_static_bias_level(adinput=None):
    """
    Determine the static bias level value from GMOS data.
    """
    # Since this function accesses keywords in the headers of the pixel data
    # extensions, always return a dictionary where the key of the dictionary is
    # an (EXTNAME, EXTVER) tuple 
    static_bias_level = {}
    
    # Get the static bias level lookup table
    gmosampsBias, gmosampsBiasBefore20060831 = Lookups.get_lookup_table(
        "Gemini/GMOS/GMOSAmpTables", "gmosampsBias",
        "gmosampsBiasBefore20060831")
    
    # Get the UT date, read speed setting and gain setting values using the
    # appropriate descriptors. Use as_pytype() and as_dict() to return the
    # values as the default python type and a dictionary, respectively, rather
    # than an object.
    ut_date = str(adinput.ut_date())
    read_speed_setting = adinput.read_speed_setting().as_pytype()
    gain_setting_dict = adinput.gain_setting().as_dict()
    
    # Get the name of the detector amplifier from the header of each pixel data
    # extension as a dictionary
    ampname_dict = gmu.get_key_value_dict(adinput, "AMPNAME")
    
    if (ut_date is not None and read_speed_setting is not None and
        gain_setting_dict is not None and ampname_dict is not None):
        
        obs_ut_date = datetime(*strptime(ut_date, "%Y-%m-%d")[0:6])
        old_ut_date = datetime(2006, 8, 31, 0, 0)
        
        for ext_name_ver, gain_setting in gain_setting_dict.iteritems():
            ampname = ampname_dict[ext_name_ver]
            
            bias_key = (read_speed_setting, gain_setting, ampname)
            
            bias_level = None
            if obs_ut_date > old_ut_date:
                 if bias_key in gmosampsBias:
                    bias_level = gmosampsBias[bias_key]
            else:
                if bias_key in gmosampsBiasBefore20060831:
                    bias_level = gmosampsBiasBefore20060831[bias_key]
            
            # Update the dictionary with the bias level value
            static_bias_level.update({ext_name_ver: bias_level})
    
    if len(static_bias_level) == 1:
        # Only one value will be returned
        ret_static_bias_level = static_bias_level.values()[0]
    else:
        unique_values = set(static_bias_level.values())
        if len(unique_values) == 1 and None in unique_values:
            
            # The bias level was not found for any of the pixel data extensions
            # (all the values in the dictionary are equal to None)
            ret_static_bias_level = None
        else:
            ret_static_bias_level = static_bias_level
    
    return ret_static_bias_level

def _get_static_bias_level_for_ext(adinput=None):
    """
    Determine the static bias level value from GMOS data.
    """
    # Get the static bias level lookup table
    gmosampsBias, gmosampsBiasBefore20060831 = Lookups.get_lookup_table(
        "Gemini/GMOS/GMOSAmpTables", "gmosampsBias",
        "gmosampsBiasBefore20060831")
    
    # Get the UT date, read speed setting and gain setting values using the
    # appropriate descriptors. Use as_pytype() to return the values as the
    # default python type rather than an object.
    ut_date = str(adinput.ut_date())
    read_speed_setting = adinput.read_speed_setting().as_pytype()
    gain_setting = adinput.gain_setting().as_pytype()
    
    # Get the name of the detector amplifier from the header of each pixel data
    # extension as a dictionary
    ampname = adinput.get_key_value("AMPNAME")
    
    ret_static_bias_level = None
    if (ut_date is not None and read_speed_setting is not None and
        gain_setting is not None and ampname is not None):
        
        obs_ut_date = datetime(*strptime(ut_date, "%Y-%m-%d")[0:6])
        old_ut_date = datetime(2006, 8, 31, 0, 0)
        
        bias_key = (read_speed_setting, gain_setting, ampname)
        
        ret_static_bias_level = None
        if obs_ut_date > old_ut_date:
            if bias_key in gmosampsBias:
                ret_static_bias_level = gmosampsBias[bias_key]
        else:
            if bias_key in gmosampsBiasBefore20060831:
                ret_static_bias_level = gmosampsBiasBefore20060831[bias_key]
            
    return ret_static_bias_level
