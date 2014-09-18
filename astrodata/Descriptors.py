#
#                                                                  gemini_python
#
#                                                         astrodata_X1/astrodata
#                                                                 Descriptors.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
"""
The Descriptors module provides the following definitions, which are used to
build the Descriptors interface for a given AstroData instance.

  -- Calculator
  -- Unit
  -- CouldNotCollapse
  -- DescriptorValue

It also provides the factory function, 

  -- get_calculator()

This factory will return the appropriate Calculator instance for the given 
dataset, according to the Descriptor configuration files. There is no explicit
Descriptor class. The classes in question are the Calculators. The function is 
used by AstroData instances to get the appropriate Calculator, which is then 
called by the base Descriptor related members defined in the AstroData class.

@Note: The calculator could be mixed into the AstroData instance, but this
would mean that before mixing those functions are not available. We have opted
the slightly more typing intensive solution of proxying descriptor access in 
AstroData member functions, which then call the appropriate member function of 
the calculator associated with the dataset.
"""
# ------------------------------------------------------------------------------
import os
import re
import sys
import inspect

from copy import copy

import Errors
from   ConfigSpace import config_walk

# As of September 26, 2012, FITS_KeyDict in FITS_Keywords.py is used
# instead of the previous globalStdkeyDict in StandardDescriptorKeyDict.py.
try:
    from FITS_Keywords import FITS_KeyDict
except ImportError:
    FITS_KeyDict = {}

# ------------------------------------------------------------------------------
DESCRIPTORSPACE = "descriptors"
# ------------------------------------------------------------------------------

# NOTE: to address the issue of Descriptors module being a singleton, instead
# of the approach used for the ClassificationLibrary, we use the descriptors
# module itself as the singleton and thus these module level "globals" which
# serve the purpose of acting as a central location for Descriptor behavior.
firstrun = True

# calculatorIndexREMask used to identify descriptorIndex files
# these files need to set descriptorIndex to a dictionary value
# relating AstroType names to descriptor calculator names, with the
# latter being of proper form to "exec" in python. 
calculatorIndexREMask = r"calculatorIndex\.(?P<modname>.*?)\.py$"

if (True):
    #note, the firstrun logic runs but is not needed, python only imports once
    # this module operates like a singleton
    centralCalculatorIndex = {}
    calculatorPackageMap = {}
    loadedCalculatorIndex = {}
    # WALK the config space as a directory structure
    for root, dirn, files in config_walk(DESCRIPTORSPACE):
        if root not in sys.path:
            sys.path.append(root)
        if True:
            for dfile in files:
                if (re.match(calculatorIndexREMask, dfile)):
                    fullpath = os.path.join(root, dfile)
                    diFile = open(fullpath)
                    exec diFile
                    diFile.close()
                    # file must declare calculatorIndex = {}
                    # check that this dict doesn't have keys already
                    # in the central dict
                    for key in calculatorIndex.keys():
                        if centralCalculatorIndex.has_key(key):
                            msg = "Descriptor Index CONFLICT\n"
                            msg += "... type %s redefined in\n" % key
                            msg += "... %s\n" % fullpath
                            msg += "... was already set to %s\n" \
                                   % centralCalculatorIndex[key]
                            msg += "... this is a fatal error"
                            raise Errors.DescriptorsError(msg)
                        
                    centralCalculatorIndex.update(calculatorIndex)
firstrun = False

# ------------------------------------------------------------------------------
# utility functions
def whoami():
    print repr(inspect.stack())
    return inspect.stack()[1][3]
    
def whocalledme():
    return inspect.stack()[2][3]

# ------------------------------------------------------------------------------
#   Module Level Function(s)
def get_calculator(dataset):
    """
    This function gets the Calculator instance appropriate for the specified
    dataset. Conflicts, arising from Calculators being associated with more than
    one AstroDataType classification, are resolved by traversing the type tree
    to see if one type is a subtype of the other so the more specific type can
    be used.

    @param dataset: dataset for which to determine a Calculator
    @type  dataset: <AstroData>

    @return: Calculator instance for passed dataset
    @rtype:  <Calculator>
    
    @note: OPEN ISSUE: how to deal with conflicts not resolved this way... i.e.
        if there are two assignments related to types which do not appear
        in the same type trees.
    """
    from gdpgutil import pick_config
    cfg = pick_config(dataset, centralCalculatorIndex, style="leaves")
    foundtypes = cfg.keys()

    # to the descriptor type.
    if not foundtypes:
        return Calculator()
    else:
        foundcalcs = []
        for calctype in foundtypes:
            if loadedCalculatorIndex.has_key(calctype):
                foundcalcs.append(loadedCalculatorIndex[calctype])
            else:
                # import and instantiate the basic calculator
                # note: module name is first part of calc string
                calcID = cfg[calctype]
                modname = calcID.split(".")[0]
                if calcID[-2:] == "()":
                    calcID = calcID[:-2]

                exec "import " + modname
                calcClass = eval(calcID)
                # add this calculator to the loadedCalculatorIndex
                loadedCalculatorIndex.update({calctype: calcClass})
                foundcalcs.append(calcClass)
                
        concreteCalcClass = type("CompositeCalcClass", tuple(foundcalcs), {})
        finalCalc = concreteCalcClass()
        
        from debugmodes import get_descriptor_throw
        finalCalc.throwExceptions = get_descriptor_throw()
        return finalCalc

# ------------------------------------------------------------------------------
# NOTE: Previously in their own modules, the Calculator and Unit classes are
# now incorporated into this Descriptors module.
# kra 14-08-14
# ------------------------------------------------------------------------------
class Calculator(object):
    """
    A Descriptor Calculator is an object with one member fucntion for 
    each descriptor (where descriptors are conceptually types of statistical
    or other values which can be thought of as applying to all data. A descriptor
    might not be 100% general; it may apply to a large majority of data types, 
    or require some generic handling.
    
    In practice, some descriptors may still not really apply to some data types
    they will return a valid descriptor value (e.g. if there was an instrument
    without a filter wheel, data from that instrument would still return a 
    sensible value for the filter descriptor (None).
    
    A Calculator is associated with particular classifications of data, such 
    that it can be assigned to AstroData instances cooresponding to that 
    classification. It is important that configurations not lead to multiple 
    Calculators associated with one DataSet (this can happen since AstroData 
    instances have more than one classification which can have a Calculator 
    associated with it.  The system handles one case of this where one of the two
    types contributing a Calculator is a subtype of the other, in which case the 
    system uses the subtypes descriptor, assuming it "overrides" the other.  
    In any other case the system will throw and exception when trying to assign 
    the calculator to the AstroData instance. 
    
    @note: the base class, besides being a parent class for defining new 
    Calculators is also the default Calculator  when none are specifically 
    assigned. It uses 'FITS_Keywords.py' to map variables for descriptors to 
    FITS specific header keywords, then does the retrieval from the headers in 
    the dataset, as appropriate. Ideally this method should work for all 
    prepared data, at which point we would like to have the standard values 
    stored in the header where it is directly retrieved rather than calculated.

    @ivar usage: Used to document this Descriptor.    
    """
    usage = ""
    stdkey_dict = None
    throwExceptions = False
    _specifickey_dict = None
    _update_stdkey_dict = None
    
    def __init__(self):
        stdkeydict = copy(FITS_KeyDict)
        selfmro = list(inspect.getmro(self.__class__))
        selfmro.reverse()
        for cls in selfmro:
            if not hasattr(cls, "_update_stdkey_dict"):
                continue

            if cls._update_stdkey_dict != None:
                stdkeydict.update(cls._update_stdkey_dict)
            else:
                pass

        self._specifickey_dict = stdkeydict
        self.stdkey_dict = stdkeydict
        
        
    def get_descriptor_key(self, name):
        try:
            return self.stdkey_dict[name]
        except KeyError:
            return None

# ------------------------------------------------------------------------------
# DescriptorValue unit conversion.

class Unit(object):
    def __init__(self, name="unitless", factor=None, mks=None):
        self.mks    = mks
        self.name   = name
        self.factor = factor

    def __str__(self):
        return self.name
    
    def convert(self, value, newunit):
        if newunit.mks != self.mks:
            if newunit.mks is None:
                print "MKS compatability not defined"
            raise ValueError("@L28 DescriptorUnits: Imcompatible unit types")
        return value*(self.factor/newunit.factor)

# meters, kilograms, seconds (mks, SI Units)
# as metadata in the nested dict for compatability testing
iunits = {"m"           :{1.          :"m"},
          "meter"       :{1.          :"m"},
          "meters"      :{1.          :"m"},
          "mile"        :{201168./125.:"m"},
          "miles"       :{201168./125.:"m"},
          "feet"        :{381./1250.  :"m"},
          "foot"        :{381./1250.  :"m"},
          "inch"        :{127./5000.  :"m"},
          "inches"      :{127./5000.  :"m"},
          "km"          :{1e3         :"m"},
          "kilometer"   :{1e3         :"m"},
          "kilometers"  :{1e3         :"m"},
          "cm"          :{1e-2        :"m"},
          "centimeter"  :{1e-2        :"m"},
          "centimeters" :{1e-2        :"m"},
          "mm"          :{1e-3        :"m"},
          "millimeter"  :{1e-3        :"m"},
          "millimeters" :{1e-3        :"m"},
          "um"          :{1e-6        :"m"},
          "micron"      :{1e-6        :"m"},
          "microns"     :{1e-6        :"m"},
          "micrometer"  :{1e-6        :"m"},
          "micrometers" :{1e-6        :"m"},
          "nm"          :{1e-9        :"m"},
          "nanometer"   :{1e-9        :"m"},
          "nanometers"  :{1e-9        :"m"},
          "angstrom"    :{1e-10       :"m"},
          "angstroms"   :{1e-10       :"m"},
          "kilogram"    :{1.          :"k"},
          "kilo"        :{1.          :"k"},
          "kilograms"   :{1.          :"k"},
          "second"      :{1.          :"s"},
          "sec"         :{1.          :"s"},
          "seconds"     :{1.          :"s"},
          "scalar"      :{1.          :"scalar"}
         }

# set all the Unit objects
for unit_name in iunits.keys():
    for unit_factor in iunits[unit_name].keys():
        for unit_mks in iunits[unit_name].values():
            g = globals()
            g[unit_name] = Unit(name=unit_name, factor=unit_factor, mks=unit_mks)

# ------------------------------------------------------------------------------
class CouldNotCollapse(object):
    pass

# ------------------------------------------------------------------------------
class DescriptorValue(object):
    """
    Because metadata may comprise a variety of data types, string, integer, 
    float, even mulitple values, etc., this class encapsulates that variety, 
    defines the container all for descriptor return values. This ensures a 
    consistent interface on descriptor calls: the return value of a all 
    descriptors is *always* a DescriptorValue instance.

    Eg.,
    >>> ad = AstroData(fitsfile)
    >>> dv = ad.pixel_scale()
    >>> dv
    <DescriptorValue object ...>

    The DescriptorValue.info() method displays the contents of the object:
    >>> dv.info()

    Descriptor Value Info:
    .name             = pixel_scale
    ._val             = 0.146
    type(._val)       = <type 'float'>
    .pytype           = <type 'float'>
    .unit             = scaler
    .dict_val         = {('SCI', 1):0.146,
                             ('SCI', 2):0.146,
                             ('SCI', 3):0.146}
    """
    def __init__(self, initval, format=None, name="unknown", keyword=None,
                 ad=None, pytype=None, unit=None, primary_extname="SCI"):

        self._val = None
        self.dict_val = None
        self.pytype = pytype
        self._extver_dict = {}
        self._valid_collapse = None
        self.originalinitval = initval
        self._primary_extname = primary_extname
        
        if pytype is None and self.pytype is None:
            self.pytype = pytype = type(initval)

        if isinstance(initval, DescriptorValue):
            initval = initval.dict_val
        
        # unit logic
        if unit:
            self.unit = unit
        else:
            self.unit = scalar
        
        if isinstance(initval, dict):
            # process initval dicts keyed by int
            keys = initval.keys()
            ntuplekeys = 0
            nintkeys = 0
            for key in keys:
                if type(key) == tuple:
                    ntuplekeys += 1
                elif type(key) == int:
                    nintkeys += 1
            if nintkeys > 0 and ntuplekeys == 0:
                newinitval = {}
                for key in keys:
                    newinitval[("*",key)] = initval[key]
                initval = newinitval
            
            self.dict_val = initval
            val = None
        else:
            self._val = initval
            self.dict_val = {}
            if ad:
                for ext in ad:
                    self.dict_val.update(
                      {(ext.extname(),ext.extver()) : self._val})
            else:
                self.dict_val = {("*",-1):self._val}
        #NOTE:
        # DO NOT SAVE AD INSTANCE, do not keep AD instances in memory 
        # due to descriptor values persisting.
        self.name = name
        self.format = format
        self.keyword = keyword

        # do after object is set up
        self._val = self.collapse_value() 
        self.collapse_by_extver()
    
    def __float__(self):
        value = self.collapse_dict_val()
        return float(value)
    
    def __int__(self):
        value = self.collapse_dict_val()
        return int(value)
    
    def __str__(self):
        format = self.format
        # do any automatic format heuristics
        if format is None:
            val = self.collapse_value()
            if not self._valid_collapse:
                format = "as_dict"
            else:
                format = "value"
        
        # produce known formats
        retstr = "Unknown Format For DescriptorValue"
        if format == "as_dict":
            retstr = str(self.dict_val)
        elif format == "db" or format == "value":
            self.collapse_value()
            if self._valid_collapse == True:
                retstr = str(self._val)
            else:
                #SEE IF IT COLLAPSES BY EXTVER
                bevdict = self.collapse_by_extver_if_valid()  
                if (not bevdict):
                    dv = self.dict_val
                else:
                    dv = bevdict                            

                parts = []
                keys = dv.keys()
                keys.sort()
                for k in keys:
                    parts.append(str(dv[k]))                       
                retstr = "+".join(parts)
        elif format == "value":
            val = self.collapse_value()
            retstr = str(val)
        return retstr

    @property
    def collapse_extname(self):
        return self._primary_extname

    @collapse_extname.setter
    def collapse_extname(self, val):
        self._val = None
        self._primary_extname = val
        
    primary_extname = None
    
    def collapse_dict_val(self):
        value = self.collapse_value()
        if value is None:
            err_msg = ("\n"
                       "Cannot convert DescriptorValue to scaler\n"
                       "as the value varies across extens versions\n"
                       "------------------------------------------\n"
                       "%s" % self._info())
            raise Errors.DescriptorsError(err_msg)
        # got here then all values were identical
        return value

    def collapse_by_extver(self):
        for extver in self.ext_vers():
            self.collapse_value(extver)
        return self._extver_dict
    
    def validate_collapse_by_extver(self, edict):
        for key in edict:
            if edict[key] == CouldNotCollapse:
                return False
        return True
        
    def collapse_by_extver_if_valid(self):
        edict = self.collapse_by_extver()
        valid = self.validate_collapse_by_extver(edict)
        if valid:
            return edict
        else:
            return None
    
    def convert_value_to(self, new_units, new_type=None):
        newDict = copy(self.dict_val)
        for key in self.dict_val:
            val    = self.dict_val[key]
            newval = self.unit.convert(val, new_units)
            if new_type:
                newval = new_type(newval)
            else:
                if self.pytype:
                    newval = self.pytype(newval)
            newDict.update({key:newval})
            
        if new_type:
            pytype = new_type
        else:
            pytype = self.pytype
            
        retval = DescriptorValue(newDict, name=self.name, unit=new_units, 
                                 pytype=pytype, format=self.format)
        return retval
    
    def for_db(self):
        oldformat = self.format
        self.format = "db"
        val = self.as_pytype()
        self.format = oldformat
        if type(val) == tuple:
            val = str(val)
        return val
    
    def as_pytype(self, as_type=None, convert_values=False):
        """This function will convert the DV instance to its pytype or
        any other type specified. Special handling is done for 
        dictionary and lists.
        """
        # case where the value cannot collapse (dict, list)
        if not self._valid_collapse:
            if convert_values is True and as_type is not None:
                if as_type == dict:
                    return self.dict_val
                elif as_type == list:
                    keys = self.dict_val.keys()
                    keys.sort()
                    retlist = []
                    for key in keys:
                        retlist.append(self.dict_val[key])
                    return retlist
                else:
                    retdict = self.dict_val
                    for key in retdict.keys():
                        if as_type == str:
                            retdict[key] = repr(retdict[key])
                        else:
                            retdict[key] = as_type(retdict[key])
                    return retdict
            elif convert_values is False and as_type is not None:
                if as_type == dict:
                    return self.dict_val
                elif as_type == list:
                    keys = self.dict_val.keys()
                    keys.sort()
                    retlist = []
                    for key in keys:
                        retlist.append(self.dict_val[key])
                    return retlist
                elif as_type == str:
                    return str(self)
                else:
                    raise TypeError("Not supported")
            elif convert_values is False and self.pytype == list:
                keys = self.dict_val.keys()
                keys.sort()
                retlist = []
                for key in keys:
                    retlist.append(self.dict_val[key])
                return retlist
            else:
                #print "D273"
                return str(self)
        
        # case where value is a single ext fits (int, str, float)
        elif as_type is not None:
            if as_type == dict:
                return self.dict_val
            elif as_type == list and self.pytype != list:
                newlist =[]
                newlist.append(self.pytype(self._val))
                return newlist
            elif self.pytype == list:
                return self._val
            else:
                return as_type(self._val) 
        else:
            # print repr(self.pytype)
            try:
                return self.pytype(self._val)
            except:
                return self._val
    # alias
    for_numpy = as_pytype
    
    # the as_<type> aliases
    def as_dict(self):
        return self.as_pytype(dict)
    
    def as_list(self):
        return self.as_pytype(list)
    
    def as_str(self):
        return self.as_pytype(str)
    
    def as_float(self):
        return self.as_pytype(float, convert_values=True)
    
    def as_int(self):
        return self.as_pytype(int, convert_values=True)
    
    def get_input_value(self, as_type=None, convert_values=False):
        return self.originalinitval
    
    def get_value(self, extver=None):
        return self.collapse_value(extver=extver)
    
    def is_none(self):
        if self._valid_collapse and self._val is None:
            return True
        else:
            return False

    def info(self):
        print self._info()
        return

    def collapse_value(self, extver=None):
        oldvalue = None
        primext = self.primary_extname
        if primext not in self.ext_names():
            primext = None

        for key in self.dict_val:
            ext_name,ext_ver = key
            if extver and ext_ver != extver:
                continue

            if primext and ext_name != primext:
                continue

            value = self.dict_val[key]
            if oldvalue is None:
                oldvalue = value
            else:
                if oldvalue != value:
                    if extver is None:
                        self._val = CouldNotCollapse
                        self._valid_collapse = False
                    else:
                        self._extver_dict[extver] = CouldNotCollapse
                    return None

        # got here then all values were identical
        if extver is None:
            self._val = value
            self._valid_collapse = True
        else:
            self._extver_dict[extver] = value
        return value
    
    def ext_names(self):
        enames = []
        for extname,extver in self.dict_val.keys():
            if extname not in enames:
                enames.append(extname)
        return enames

    def ext_vers(self):
        evers = []
        for extname, extver in self.dict_val.keys():
            if extver not in evers:
                evers.append(extver)
        evers.sort()
        return evers
    
    def overloaded(self, other):
        val = self.collapse_value()
        if not self._valid_collapse:
            mes = "DescriptorValue contains complex result (differs for"
            mes += "different extension) and cannot be used as a simple "
            mes += str(self.pytype)
            raise Errors.DescriptorValueTypeError(mes)
        else:
            if type(val) != self.pytype:
                val = self.as_pytype()

        myfuncname = whocalledme()
        if isinstance(other, DescriptorValue):
            other = other.as_pytype()
        # always try the following and let it raise #hasattr(val, myfuncname):
        if True:
            try:
                op = None
                hasop = eval('hasattr(self.%s,"operation")' % myfuncname)
                if hasop: 
                    op = eval("self.%s.operation" % myfuncname)
                if op:
                    retval = eval(op)
                else:
                    raise "problem"
                    retval = eval("val.%s(other)" % myfuncname)
                return retval
            except TypeError, e:
                raise Errors.DescriptorValueTypeError(str(e))
        else:
            raise Errors.DescriptorValueTypeError(
              "Unsupported operand, %s, for types %s and %s"
              % (myfuncname, str(self.pytype), str(type(other))))
        return
    
    def overloaded_cmp(self, other):
        val = self.collapse_value()
        
        if not self._valid_collapse:
            mes = "DescriptorValue contains complex result (differs for "
            mes += "different extension) and cannot be used as a simple"
            mes += str(self.pytype)
            raise Errors.DescriptorValueTypeError(mes)
        else:
            if type(val) != self.pytype:
                val = self.as_pytype()
                
        myfuncname = whocalledme()
        if isinstance(other, DescriptorValue):
            other = other.as_pytype()
        
        othertype = type(other)
        mine = self.as_pytype()
        if hasattr(mine, "__eq__"):
            try:
                retval = eval ("mine == other")
                if retval:
                    return 0
            except:
                pass

        if hasattr(mine, "__gt__"):
            try:
                retval = eval ("mine > other")
                if retval:
                    return 1
                else:
                    return -1
            except:
                pass

        if hasattr(mine, "__cmp__"):
            try:
                retval = eval ("cmp(mine,other)")
                return retval
            except:
                pass

        raise Errors.IncompatibleOperand("%s has no method %s"
                                         % (str(type(other)), myfuncname))
        return


    def _info(self):
        dvstr = ("\nDescriptor Value Info:\n"
                 "\t.name             = %s\n"
                 "\t._val             = %s\n"
                 "\ttype(._val)       = %s\n"
                 "\t.pytype           = %s\n"
                 "\t.unit             = %s\n" % (str(self.name),
                                                 repr(self._val),
                                                 str(type(self._val)),
                                                 str(self.pytype),
                                                 str(self.unit)))
        keys = self.dict_val.keys()
        keys.sort()
        lkeys = len(keys)
        count = 0
        for key in keys:
            if count == 0:
                dvstr += ("\t.dict_val         = {%s:%s,\n" %
                          (str(key), repr(self.dict_val[key])))
            elif count == lkeys - 1:
                dvstr += ("%s%s:%s}\n" %
                          (" " * 29, str(key), repr(self.dict_val[key])))
            else: 
                dvstr += ("%s%s:%s,\n" %
                          (" " * 29, str(key), repr(self.dict_val[key])))
            count += 1
        return dvstr
            
    # overloaded operators (used for int and float)
    def __add__(self, other):
        return self.overloaded(other)
    __add__.operation = "val + other"
    def __div__(self, other):
        return self.overloaded(other)
    __div__.operation = "val / other"
    def __floordiv__(self, other):
        return self.overloaded(other)
    __floordiv__.operation = "val // other"
    def __rfloordiv__(self, other):
        return self.overloaded(other)
    __rfloordiv__.operation = "other // val"
    def __divmod__(self, other):
        return self.overloaded(other)
    __divmod__.operation = "divmod(val, other)"
    
    def __mod__(self, other):
        return self.overloaded(other)
    __mod__.operation = "val % other"
    def __mul__(self, other):
        return self.overloaded(other)
    __mul__.operation = "val * other"
    def __pow__(self, other):
        return self.overloaded(other)
    __pow__.operation = "pow(val,other)"
    def __radd__(self, other):
        return self.overloaded(other)
    __radd__.operation = "other + val"
    def __rdiv__(self, other):
        return self.overloaded(other)
    __rdiv__.operation = "other / val"
    def __rdivmod__(self, other):
        return self.overloaded(other)
    __rdivmod__.operation = "divmod(other, val)"
    def __rmul__(self, other):
        return self.overloaded(other)
    __rmul__.operation = "other * val"
    def __rdiv__(self, other):
        return self.overloaded(other)
    __rdiv__.operation = "other / val"
    def __rdivmod__(self, other):
        return self.overloaded(other)
    __rdivmod__.operation = "divmod(other, val)"
    def __rmod__(self, other):
        return self.overloaded(other)
    __rmod__.operation = "other % val"
    def __rmul__(self, other):
        return self.overloaded(other)
    __rmul__.operation = "other * val"
    def __rpow__(self, other):
        return self.overloaded(other)
    __rpow__.operation = "pow(other, val)"
    def __rsub__(self, other):
        return self.overloaded(other)
    __rsub__.operation = "other - val"
    def __rtruediv__(self, other):
        return self.overloaded(other)
    __rtruediv__.operation = "other / val"
    def __sub__(self, other):
        return self.overloaded(other)
    __sub__.operation = "val - other"
    def __truediv__(self, other):
        return self.overloaded(other)
    __truediv__.operation = "val / other"
    
    # overloaded operators unique to int
    def __and__(self, other):
        return self.overloaded(other)
    __and__.operation = "val & other"
    def __cmp__(self, other):
        return self.overloaded_cmp(other)
    #__cmp__.operation = "cmp(val, other)"
    
    def __lshift__(self, other):
        return self.overloaded(other)
    __lshift__.operation = "val << other"
    def __or__(self, other):
        return self.overloaded(other)
    __or__.operation = "val | other"
    def __rand__(self, other):
        return self.overloaded(other)
    __rand__.operation = "other & val"
    def __rlshift__(self, other):
        return self.overloaded(other)
    __rlshift__.operation = "other << val"
    def __ror__(self, other):
        return self.overloaded(other)
    __ror__.operation = "other | val"
    def __rrshift__(self, other):
        return self.overloaded(other)
    __rrshift__.operation = "other >> val"
    def __rshift__(self, other):
        return self.overloaded(other)
    __rshift__.operation = "val >> other" 
    def __rxor__(self, other):
        return self.overloaded(other)
    __rxor__.operation = "other ^ val"
    def __xor__(self, other):
        return self.overloaded(other)
    __xor__.operation = "val ^ other"

