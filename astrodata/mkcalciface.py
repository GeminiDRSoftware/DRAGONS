#
#                                                                  gemini_python
#
#                                                                      astrodata
#                                                                 mkcalciface.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
import re
from datetime import datetime
from astrodata import Errors
# ------------------------------------------------------------------------------
CALCIFACECLASSMARKER = "CalculatorInterface"
# ------------------------------------------------------------------------------
function_buffer = """
def %(name)s(self, format=None, **args):
    \"\"\"%(description)s\"\"\"
    try:
        self._lazyloadCalculator()
        keydict = self.descriptor_calculator._specifickey_dict
        key = \"key_%(name)s\"
        #print \"mkCI22:\",key, repr(keydict)
        #print \"mkCI23:\", key in keydict
        keyword = None
        if key in keydict.keys():
            keyword = keydict[key]
                
            #print hasattr(self.descriptor_calculator, \"%(name)s\")
        if not hasattr(self.descriptor_calculator, \"%(name)s\"):
            if keyword is not None:
                retval = self.phu_get_key_value(keyword)
                if retval is None:
                    if hasattr(self, \"exception_info\"):
                        raise Errors.DescriptorError(self.exception_info)
            else:
                msg = (\"Unable to find an appropriate descriptor \"
                       \"function or a default keyword for %(name)s\")
                raise Errors.DescriptorError(msg)
        else:
            try:
                retval = self.descriptor_calculator.%(name)s(self, **args)
            except Exception as e:
                raise Errors.DescriptorError(e)
            
        %(pytypeimport)s
        ret = DescriptorValue( retval, 
                               format = format, 
                               name = \"%(name)s\",
                               keyword = keyword,
                               ad = self,
                               pytype = %(pytype)s )
        return ret
        
    except Errors.DescriptorError:
        if self.descriptor_calculator.throwExceptions == True:
            raise
        else:
            if not hasattr(self, \"exception_info\"):
                setattr(self, \"exception_info\", sys.exc_info()[1])
            ret = DescriptorValue( None,
                                   format = format, 
                                   name = \"%(name)s\",
                                   keyword = keyword,
                                   ad = self,
                                   pytype = None )
            return ret
    """

Wholeout = """import sys
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

# ------------------------------------------------------------------------------
def get_calculator_interface():
    """Combination of making and getting calc iface objects
    """
    from astrodata.ConfigSpace import ConfigSpace

    calcIfaces = []
    for cil_el in ConfigSpace.calc_iface_list:
        ifType = cil_el[0]
        ifFile = cil_el[1]
                
        if ifType == "CALCIFACE":
            cib = open(ifFile)
            d = globals()
            exec(cib, d)
            for key in d:
                if re.match(CALCIFACECLASSMARKER, key):
                    calcIfaces.append(d[key])
                    break;           # get the first, one per module
        if ifType == "DDLIST":
            cib = open(ifFile)
            cibsrc = cib.read()
            cib.close()
            d = {"DescriptorDescriptor":DD, 
                 "DD":DD,
                 "datetime":datetime
                }
            ddlist = eval(cibsrc, d)

            try:
                cisrc = mk_calc_iface_body(ddlist)
            except Errors.BadConfiguration as bc:
                bc.add_msg("FATAL CONFIG ERROR: %s" % ifFile)
                raise bc

            exec(cisrc, d)
            for key in d:
                if re.match(CALCIFACECLASSMARKER, key):
                    calcIfaces.append(d[key])
            
    CalculatorInterfaceClass = ComplexCalculatorInterface
    for calcIface in calcIfaces:
        CalculatorInterfaceClass.__bases__ += (calcIface, )
    return CalculatorInterfaceClass
                
# ------------------------------------------------------------------------------    
def mk_calc_iface_body(ddlist):
    out = ""
    for dd in ddlist:
        try:
            out += dd.funcbody()
        except Errors.BadConfiguration as bc:
            bc.add_msg("Problem with ddlist item #%d" % ddlist.index(dd))
            raise bc

    finalout = wholeout % {"descriptors": out}
    return finalout

# ------------------------------------------------------------------------------
class ComplexCalculatorInterface(object):
    pass

# ------------------------------------------------------------------------------
class DescriptorDescriptor:

    def __init__(self, name=None, pytype=None):
        self.name   = name
        self.pytype = pytype

    def funcbody(self):
        if self.pytype:
            pytypestr = self.pytype.__name__
        else:
            pytypestr = "None"

        if pytypestr == "datetime":
            pti = "from datetime import datetime"
        else:
            pti = ""

        if self.pytype:
            rtype = self.pytype.__name__
            if rtype == 'str':
                rtype = 'string'
            if rtype == 'int':
                rtype = 'integer'
        else:
            raise Errors.BadConfiguration("'DD' for '%s' needs a pytype" % 
                                          self.name)

        # Use docstring defined in docstrings mod.
        use_docstrings = False
        try:
            from docstrings import docstrings
            if hasattr(docstrings, self.name):
                use_docstrings = True
        except:
            pass
        
        if use_docstrings:
            doc = "docstrings.%(name)s.__doc__.__str__()" % {'name':self.name}
            description = eval(doc)
        else:
            doc = ("\n"
                   "        Return the %(name)s value\n\n"
                   "        :param dataset: the data set\n"
                   "        :type dataset: AstroData\n"
                   "        :param format: the return format\n"
                   "        :type format: string\n"
                   "        :rtype: %(rtype)s as default (i.e., format=None)\n"
                   "        :return: the %(name)s value\n"
                   "        ") % {'name':self.name, 'rtype':rtype}
            description = doc
        
        ret = function_buffer % {'name':         self.name,
                                 'pytypeimport': pti,
                                 'pytype':       pytypestr,
                                 'description':  description}
        return ret
DD = DescriptorDescriptor
