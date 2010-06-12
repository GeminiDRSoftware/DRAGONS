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
            return self.descriptorCalculator.%(name)s(self, **args)
        except:
            if self.descriptorCalculator.throwExceptions == True:
                raise
            else:
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
                    DD("amp_read_area"),
                    DD("azimuth"),
                    DD("camera"),
                    DD("cass_rotator_pa"),
                    DD("central_wavelength"),
                    DD("coadds"),
                    DD("data_label"),
                    DD("data_section"),
                    DD("dec"),
                    DD("detector_section"),
                    DD("detector_x_bin"),
                    DD("detector_y_bin"),
                    DD("disperser"),
                    DD("dispersion"),
                    DD("dispersion_axis"),
                    DD("elevation"),
                    DD("exposure_time"),
                    DD("filter_id"),
                    DD("filter_name"),
                    DD("focal_plane_mask"),
                    DD("gain"),
                    DD("gain_mode"),
                    DD("instrument"),
                    DD("local_time"),
                    DD("mdf_row_id"),
                    DD("non_linear_level"),
                    DD("object"),
                    DD("observation_class"),
                    DD("observation_epoch"),
                    DD("observation_id"),
                    DD("observation_mode"),
                    DD("observation_type"),
                    DD("observer"),
                    DD("pixel_scale"),
                    DD("program_id"),
                    DD("pupil_mask"),
                    DD("ra"),
                    DD("raw_bg"),
                    DD("raw_cc"),
                    DD("raw_gemini_qa"),
                    DD("raw_iq"),
                    DD("raw_pi_requirement"),
                    DD("raw_wv"),
                    DD("read_mode"),
                    DD("read_noise"),
                    DD("read_speed_mode"),
                    DD("saturation_level"),
                    DD("ssa"),
                    DD("telescope"),
                    DD("ut_date"),
                    DD("ut_time"),
                    DD("wavefront_sensor"),
                    DD("wavelength_reference_pixel"),
                    DD("well_depth_mode"),
                    DD("x_offset"),
                    DD("y_offset"),
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
