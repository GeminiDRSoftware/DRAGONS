from datetime import datetime
from descriptorDescriptionDict import asDictArgDict
from descriptorDescriptionDict import descriptorDescDict
from descriptorDescriptionDict import detailedNameDict
from descriptorDescriptionDict import stripIDArgDict

class DescriptorDescriptor:
    name = None
    description = None
    pytype = None
    unit = None
    
    thunkfuncbuff = """
    def %(name)s(self, format=None, **args):
        \"\"\"
        %(description)s
        \"\"\"
        try:
            self._lazyloadCalculator()
            keydict = self.descriptor_calculator._specifickey_dict
            key = "key_"+"%(name)s"
            #print "mkCI22:",key, repr(keydict)
            #print "mkCI23:", key in keydict
            if key in keydict.keys():
                keyword = keydict[key]
            else:
                keyword = None
            #print hasattr(self.descriptor_calculator, "%(name)s")
            if not hasattr(self.descriptor_calculator, "%(name)s"):
                if keyword is not None:
                    retval = self.phu_get_key_value(keyword)
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info
                else:
                    msg = "Unable to find an appropriate descriptor function "
                    msg += "or a default keyword for %(name)s"
                    raise KeyError(msg)
            else:
                retval = self.descriptor_calculator.%(name)s(self, **args)
            
            %(pytypeimport)s
            ret = DescriptorValue( retval, 
                                   format = format, 
                                   name = "%(name)s",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = %(pytype)s )
            return ret
        except:
            if not hasattr(self, "exception_info"):
                setattr(self, "exception_info", sys.exc_info()[1])
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
        try:
            desc = descriptorDescDict[name]
        except:
            if rtype == 'str':
                rtype = 'string'
            if rtype == 'int':
                rtype = 'integer'
            try:
                dname = detailedNameDict[name]
            except:
                dname = name
            try:
                asDictArg = asDictArgDict[name]
            except:
                asDictArg = 'no'
            try:
                stripIDArg = stripIDArgDict[name]
            except:
                stripIDArg = 'no'

            if stripIDArg == 'yes':
                desc = 'Return the %(name)s value\n' % {'name':name} + \
                       '        :param dataset: the data set\n' + \
                       '        :type dataset: AstroData\n' + \
                       '        :param stripID: set to True to remove the ' + \
                       'component ID from the \n                        ' + \
                       'returned %(name)s value\n' % {'name':name} + \
                       '        :type stripID: Python boolean\n' + \
                       '        :param pretty: set to True to return a ' + \
                       'human meaningful \n' + \
                       '                       %(name)s ' % {'name':name} + \
                       'value\n' + \
                       '        :type pretty: Python boolean\n' + \
                       '        :rtype: %(rtype)s ' % {'rtype':rtype} + \
                       'as default (i.e., format=None)\n' + \
                       '        :return: the %(dname)s' \
                       % {'dname':dname}
            elif asDictArg == 'yes':
                desc = 'Return the %(name)s value\n' % {'name':name} + \
                       '        :param dataset: the data set\n' + \
                       '        :type dataset: AstroData\n' + \
                       '        :param format: the return format\n' + \
                       '                       set to as_dict to return a ' + \
                       'dictionary, where the number ' + \
                       '\n                       of dictionary elements ' + \
                       'equals the number of pixel data ' + \
                       '\n                       extensions in the image. ' + \
                       'The key of the dictionary is ' + \
                       '\n                       an (EXTNAME, EXTVER) ' + \
                       'tuple, if available. Otherwise, ' + \
                       '\n                       the key is the integer ' + \
                       'index of the extension.\n' + \
                       '        :type format: string\n' + \
                       '        :rtype: %(rtype)s ' % {'rtype':rtype} + \
                       'as default (i.e., format=None)\n' + \
                       '        :rtype: dictionary containing one or more ' + \
                       '%(rtype)s(s) ' % {'rtype':rtype} + \
                       '(format=as_dict)\n' + \
                       '        :return: the %(dname)s' \
                       % {'dname':dname}

            else:
                desc = 'Return the %(name)s value\n' % {'name':name} + \
                       '        :param dataset: the data set\n' + \
                       '        :type dataset: AstroData\n' + \
                       '        :param format: the return format\n' + \
                       '        :type format: string\n' + \
                       '        :rtype: %(rtype)s ' % {'rtype':rtype} + \
                       'as default (i.e., format=None)\n' + \
                       '        :return: the %(dname)s' \
                       % {'dname':dname}
                
        self.description = desc
        
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
                    DD("detector_name", pytype=str),
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
                    DD("nominal_atmospheric_extinction", pytype=float),
                    DD("nominal_photometric_zeropoint", pytype=float),
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
                    DD("requested_iq", pytype=str),
                    DD("requested_cc", pytype=str),
                    DD("requested_wv", pytype=str),
                    DD("requested_bg", pytype=str),
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
out = ""

for dd in descriptors:
    out += dd.funcbody()
    
finalout = wholeout % {"descriptors": out}

print finalout
