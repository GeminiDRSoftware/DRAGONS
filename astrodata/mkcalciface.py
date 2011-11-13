from datetime import datetime

from astrodata.ConfigSpace import CALCIFACEMARKER, DDLISTMARKER
CALCIFACECLASSMARKER = "CalculatorInterface"

import re
class DescriptorDescriptor:
    name = None
    description = ""
    pytype = None
    unit = None
    
    thunkfuncbuff = """
    def %(name)s(self, format=None, **args):
        \"\"\"
        %(description)s
        \"\"\"
        try:
            self._lazyloadCalculator()
            if not hasattr(self.descriptor_calculator, "%(name)s"):
                msg = "Unable to find an appropriate descriptor function "
                msg += "or a default keyword for %(name)s"
                raise KeyError(msg)
            else:
                retval = self.descriptor_calculator.%(name)s(self, **args)
            
            %(pytypeimport)s
            ret = DescriptorValue( retval, 
                                   format = format, 
                                   name = "%(name)s",
                                   ad = self,
                                   pytype = %(pytype)s )
            return ret
        except:
            if not hasattr(self, "exception_info"):
                setattr(self, "exception_info", sys.exc_info()[1])
                import traceback
                traceback.print_exc()
            if (self.descriptor_calculator is None 
                or self.descriptor_calculator.throwExceptions == True):
                raise
            else:
                #print "NONE BY EXCEPTION"
                self.exception_info = sys.exc_info()[1]
                return None
    """
    def __init__(self, name=None, pytype=None):
        self.name = name
        if pytype:
            self.pytype = pytype
            rtype = pytype.__name__
        
    def funcbody(self):
        if self.pytype:
            pytypestr = self.pytype.__name__
        else:
            pytypestr = "None"
        if pytypestr == "datetime":
            pti = "from datetime import datetime"
        else:
            pti = ""
        #print "mkC150:", pti
        ret = self.thunkfuncbuff % {'name':self.name,
                                    'pytypeimport': pti,
                                    'pytype': pytypestr,
                                    'description':self.description}
        return ret
        
DD = DescriptorDescriptor
        
descriptors =   [   DD("airmass", pytype=float),
                    DD("amp_read_area", pytype=str),
                    DD("array_section", pytype=list),
                    DD("azimuth", pytype=float),
                    DD("camera", pytype=str),
                    DD("cass_rotator_pa", pytype=float),
                    DD("central_wavelength", pytype=float),
                    DD("coadds", pytype=int),
                    DD("data_label", pytype=str),
                    DD("data_section", pytype=list),
                    DD("dec", pytype=float),
                    DD("decker", pytype=str),
                    DD("detector_section", pytype=list),
                    DD("detector_x_bin", pytype=int),
                    DD("detector_y_bin", pytype=int),
                    DD("disperser", pytype=str),
                    DD("dispersion", pytype=float),
                    DD("dispersion_axis", pytype=int),
                    DD("elevation", pytype=float),
                    DD("exposure_time", pytype=float),
                    DD("filter_name", pytype=str),
                    DD("focal_plane_mask", pytype=str),
                    DD("gain", pytype=float),
                    DD("grating", pytype=str),
                    DD("group_id", pytype=str),
                    DD("gain_setting", pytype=str),
                    DD("instrument", pytype=str),
                    DD("local_time", pytype=str),
                    DD("mdf_row_id", pytype=int),
                    DD("nod_count", pytype=int),
                    DD("nod_pixels", pytype=int),
                    DD("non_linear_level", pytype=int),
                    DD("object", pytype=str),
                    DD("observation_class", pytype=str),
                    DD("observation_epoch", pytype=str),
                    DD("observation_id", pytype=str),
                    DD("observation_type", pytype=str),
                    DD("overscan_section", pytype=list),
                    DD("pixel_scale", pytype=float),
                    DD("prism", pytype=str),
                    DD("program_id", pytype=str),
                    DD("pupil_mask", pytype=str),
                    DD("qa_state", pytype=str),
                    DD("ra", pytype=float),
                    DD("raw_bg", pytype=str),
                    DD("raw_cc", pytype=str),
                    DD("raw_iq", pytype=str),
                    DD("raw_wv", pytype=str),
                    DD("read_mode", pytype=str),
                    DD("read_noise", pytype=float),
                    DD("read_speed_setting", pytype=str),
                    DD("saturation_level", pytype=int),
                    DD("slit", pytype=str),
                    DD("telescope", pytype=str),
                    DD("ut_date", pytype=datetime),
                    DD("ut_datetime", pytype=datetime),
                    DD("ut_time", pytype=datetime),
                    DD("wavefront_sensor", pytype=str),
                    DD("wavelength_reference_pixel", pytype=float),
                    DD("well_depth_setting", pytype=str),
                    DD("x_offset", pytype=float),
                    DD("y_offset", pytype=float),
                ]

wholeout = """import sys
from astrodata import Descriptors
from astrodata.Descriptors import DescriptorValue
from astrodata import Errors

class CalculatorInterface:

    descriptor_calculator = None
%(descriptors)s
# UTILITY FUNCTIONS, above are descriptor thunks            
    def _lazyloadCalculator(self, **args):
        '''Function to put at top of all descriptor members
        to ensure the descriptor is loaded.  This way we avoid
        loading it if it is not needed.'''
        if self.descriptor_calculator is None:
            self.descriptor_calculator = Descriptors.get_calculator(self, **args)
"""

class ComplexCalculatorInterface():
    pass

def get_calculator_interface():
    """Combination of making and getting calc iface objects
    """
    from astrodata.ConfigSpace import ConfigSpace
    print "mci239:", repr(ConfigSpace.calc_iface_list)
    calcIfaces = []
    for cil_el in ConfigSpace.calc_iface_list:
        ifType = cil_el[0]
        ifFile = cil_el[1]
        if ifType == "CALCIFACE":
            print CALCIFACEMARKER
            cib = open(ifFile)
            d = globals()
            exec(cib, d)
            for key in d:
                print "key",key
                if re.match(CALCIFACECLASSMARKER, key):
                    print "ADDING a calc iface"
                    calcIfaces.append(d[key])
                    break; # just get the first one, only one per module
        if ifType == "DDLIST":
            cib = open(ifFile)
            cibsrc = cib.read()
            cib.close()
            d = {"DescriptorDescriptor":DD, 
                 "DD":DD
                }
            ddlist = eval(cibsrc, d)
            cisrc = mk_calc_iface_body(ddlist)
            exec(cisrc,d)
            for key in d:
                if re.match(CALCIFACECLASSMARKER, key):
                    calcIfaces.append(d[key])
            
    CalculatorInterfaceClass = ComplexCalculatorInterface
    for calcIface in calcIfaces:
        # print "mcif183:", repr(calcIface)
        CalculatorInterfaceClass.__bases__ += (calcIface, )
    
    return CalculatorInterfaceClass
                
            
def get_calc_iface_body():
    pass
    
def mk_calc_iface_body(ddlist):
    out = ""
    for dd in ddlist:
        out += dd.funcbody()

    finalout = wholeout % {"descriptors": out}

    return finalout
