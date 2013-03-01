from datetime import datetime

from astrodata.ConfigSpace import CALCIFACEMARKER, DDLISTMARKER
CALCIFACECLASSMARKER = "CalculatorInterface"

import re
class DescriptorDescriptor:
    name = None
    pytype = None
    
    thunkfuncbuff = """
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
        except:
            raise
    """
    
    def __init__(self, name=None, pytype=None):
        self.name = name
        if pytype:
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
        #print "mkC150:", pti

        if self.pytype:
            rtype = self.pytype.__name__
            if rtype == 'str':
                rtype = 'string'
            if rtype == 'int':
                rtype = 'integer'
            
        # Use the docstring defined in the docstrings module, if it exists
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
        
        ret = self.thunkfuncbuff % {'name':self.name,
                                    'pytypeimport': pti,
                                    'pytype': pytypestr,
                                    'description':description}
        return ret
        
DD = DescriptorDescriptor
        
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
    # print "mci239:", repr(ConfigSpace.calc_iface_list)
    calcIfaces = []
    for cil_el in ConfigSpace.calc_iface_list:
        ifType = cil_el[0]
        ifFile = cil_el[1]
        if ifType == "CALCIFACE":
            # print CALCIFACEMARKER
            cib = open(ifFile)
            d = globals()
            exec(cib, d)
            for key in d:
                # print "key",key
                if re.match(CALCIFACECLASSMARKER, key):
                    #print "ADDING a calc iface"
                    calcIfaces.append(d[key])
                    break; # just get the first one, only one per module
        if ifType == "DDLIST":
            cib = open(ifFile)
            cibsrc = cib.read()
            cib.close()
            d = {"DescriptorDescriptor":DD, 
                 "DD":DD,
                 "datetime":datetime
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
