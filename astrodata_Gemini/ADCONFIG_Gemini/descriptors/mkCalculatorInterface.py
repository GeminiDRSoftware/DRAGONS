from descriptorDescriptionDict import descriptorDescDict, returnTypeDict
from descriptorDescriptionDict import detailedNameDict, asDictArgDict
from descriptorDescriptionDict import stripIDArgDict

class DescriptorDescriptor:
    name = None
    description = None
    
    thunkfuncbuff = """
    def %(name)s(self, **args):
        \"\"\"
        %(description)s
        \"\"\"
        try:
            self._lazyloadCalculator()
            #print hasattr(self.descriptorCalculator, "%(name)s")
            if not hasattr(self.descriptorCalculator, "%(name)s"):
                key = "key_"+"%(name)s"
                #print "mkCI10:",key, repr(SDKD.globalStdkeyDict)
                #print "mkCI12:", key in SDKD.globalStdkeyDict
                if key in SDKD.globalStdkeyDict.keys():
                    retval = self.phuHeader(SDKD.globalStdkeyDict[key])
                    if retval is None:
                        if hasattr(self, "exception_info"):
                            raise self.exception_info[1]
                    return retval
            retval = self.descriptorCalculator.%(name)s(self, **args)
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
    """
    def __init__(self, name = None):
        self.name = name
        
        try:
            desc = descriptorDescDict[name]
        except:
            try:
                rtype = returnTypeDict[name]
            except:
                rtype = 'string'
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
                       '        :rtype: %(rtype)s\n' % {'rtype':rtype} + \
                       '        :return: the %(dname)s' \
                       % {'dname':dname}
            elif asDictArg == 'yes':
                desc = 'Return the %(name)s value\n' % {'name':name} + \
                       '        :param dataset: the data set\n' + \
                       '        :type dataset: AstroData\n' + \
                       '        :param asDict: set to True to return a ' + \
                       'dictionary, where the number of ' + \
                       '\n                       dictionary elements ' + \
                       'equals the number of pixel data ' + \
                       '\n                       extensions in the image. ' + \
                       'The key of the dictionary is ' + \
                       '\n                       an (EXTNAME, EXTVER) ' + \
                       'tuple, if available. Otherwise, ' + \
                       '\n                       the key is the integer ' + \
                       'index of the extension.\n' + \
                       '        :type asDict: Python boolean\n' + \
                       '        :rtype: dictionary containing one or more ' + \
                       '%(rtype)s(s)\n' % {'rtype':rtype} + \
                       '        :return: the %(dname)s' \
                       % {'dname':dname}

            else:
                desc = 'Return the %(name)s value\n' % {'name':name} + \
                       '        :param dataset: the data set\n' + \
                       '        :type dataset: AstroData\n' + \
                       '        :rtype: %(rtype)s\n' % {'rtype':rtype} + \
                       '        :return: the %(dname)s' \
                       % {'dname':dname}
                
        self.description = desc
        
    def funcbody(self):
        ret = self.thunkfuncbuff % {"name":self.name, \
            'description':self.description}
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
                    DD("decker"),
                    DD("detector_section"),
                    DD("detector_x_bin"),
                    DD("detector_y_bin"),
                    DD("disperser"),
                    DD("dispersion"),
                    DD("dispersion_axis"),
                    DD("elevation"),
                    DD("exposure_time"),
                    DD("filter_name"),
                    DD("focal_plane_mask"),
                    DD("gain"),
                    DD("grating"),
                    DD("gain_setting"),
                    DD("instrument"),
                    DD("local_time"),
                    DD("mdf_row_id"),
                    DD("nod_count"),
                    DD("nod_pixels"),
                    DD("non_linear_level"),
                    DD("object"),
                    DD("observation_class"),
                    DD("observation_epoch"),
                    DD("observation_id"),
                    DD("observation_type"),
                    DD("pixel_scale"),
                    DD("prism"),
                    DD("program_id"),
                    DD("pupil_mask"),
                    DD("qa_state"),
                    DD("ra"),
                    DD("raw_bg"),
                    DD("raw_cc"),
                    DD("raw_iq"),
                    DD("raw_wv"),
                    DD("read_mode"),
                    DD("read_noise"),
                    DD("read_speed_setting"),
                    DD("saturation_level"),
                    DD("slit"),
                    DD("telescope"),
                    DD("ut_date"),
                    DD("ut_datetime"),
                    DD("ut_time"),
                    DD("wavefront_sensor"),
                    DD("wavelength_reference_pixel"),
                    DD("well_depth_setting"),
                    DD("x_offset"),
                    DD("y_offset"),
                ]

wholeout = """import sys
import StandardDescriptorKeyDict as SDKD
from astrodata import Descriptors
from astrodata import Errors

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
