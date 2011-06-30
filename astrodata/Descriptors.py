
import sys,os
import re

"""This module contains a factory, L{get_calculator}, which will return 
the appropriate Calculator instance for the given dataset,
according to the Descriptor configuration files. There is no Descriptor
class, the classes in question are the Calculators. This function is used
by AstroData instances to get the appropriate Calculator, which is then
called by the base Descriptor related members defined in the AstroData
class.

@note: the calculator could be mixed into the AstroData instance, but this
would mean that before mixing those functions are not available. We have opted
the slightly more typing intensive (for ourselves the developers, not for users)
solution of proxying descriptor access in AstroData member functions which then
call the appropriate member function of the calculator associated with
their dataset.
"""
from astrodata import Errors
from ConfigSpace import config_walk
DESCRIPTORSPACE = "descriptors"

import inspect
from copy import copy

import DescriptorUnits as Units
from DescriptorUnits import Unit

# utility functions
def whoami():
    print repr(inspect.stack())
    return inspect.stack()[1][3]
    
def whocalledme():
    return inspect.stack()[2][3]
          
# NOTE: to address the issue of Descriptors module being a singleton, instead of the 
# approach used for the ClassificationLibrary, we use the descriptors module
# itself as the singleton and thus these module level "globals" which serve
# the purpose of acting as a central location for Descriptor behavior.

firstrun = True

class DescriptorValue():
    dict_val = None
    _val = None
    name = None
    pytype = None
    unit = None
    def __init__(self,  initval, 
                        format = None, 
                        name = "unknown", 
                        ad = None, 
                        pytype = None,
                        unit = None):
        # print "DV82:", repr(unit)
        if pytype == None and self.pytype == None:
            self.pytype = pytype = type(initval)
        originalinitval = initval
        if isinstance(initval, DescriptorValue):
            initval = initval.dict_val
            
        # pytype logic
        if pytype:
            self.pytype = pytype
        else:
            # careful moving this to a function, it gets the CALLER's function name!
            st = inspect.stack()
            callername = st[1][3]
            callerframe = inspect.stack()[1][0]
            fargs = inspect.getargvalues(callerframe)
            callercalc = fargs[3]["self"]
            try:
                self.pytype = eval("callercalc.%s.pytype" % callername)
            except:
                self.pytype = float
        pytype = self.pytype

        # unit logic
        if unit:
            self.unit = unit
        else:
            # careful moving this to a function, it gets the CALLER's function name!
            st = inspect.stack()
            callername = st[1][3]
            callerframe = inspect.stack()[1][0]
            fargs = inspect.getargvalues(callerframe)
            callercalc = fargs[3]["self"]
            try:
                self.unit = eval("callercalc.%s.unit" % callername)
            except:
                self.unit = Units.scaler # can be DescriptorUnits.scaler (or whatever)
        unit = self.unit

        if isinstance(initval, dict):
            self.dict_val = initval
            val = None
        else:
            self._val = initval
            self.dict_val = {}
            for ext in ad["SCI"]:
                self.dict_val.update({(ext.extname(),ext.extver()) : self._val})
        
        #NOTE:
        # DO NOT SAVE AD INSTANCE, we don't want AD instances kept 
        #   in memory due to descriptor values persisting
        # DO NOT SAVE AD INSTANCE, we don't want AD instances kept 
        #   in memory due to descriptor values persisting
        # DO NOT SAVE AD INSTANCE, we don't want AD instances kept 
        #   in memory due to descriptor values persisting
       
        self.name = name
        
        if format:
            self.format = format
        elif ad != None and ad.descriptorFormat:
            self.format = ad.descriptorFormat
        else:
            self.format = None
        # do after object is set up
        self._val = self.collapse_value() 
    
    
    def __float__(self):
        value = self.collapse_dict_val()
        return float(value)
    
    def __int__(self):
        value = self.collapse_dict_val()
        return int(value)
    
    def __str__(self):
        format = self.format
        # do any automatic format heuristics
        if format == None:
            val = self.collapse_value()
            if val == None:
                format = "as_dict"
            else:
                format = "value"
        
        # produce known formats
        retstr = "Unknown Format For DescriptorValue"
        if  format == "as_dict":
            retstr = str(self.dict_val)
        elif format == "db" or format == "value":
            val = self.collapse_value()
            if val != None:
                retstr = str(val)
            else:
                parts = [str(val) for val in self.dict_val.values()]
                retstr = "+".join(parts)
        elif format == "value":
            val = self.collapse_value()
        return retstr
    
    def collapse_dict_val(self):
        value = self.collapse_value()
        if value == None:
            raise Errors.DescriptorsError("\n"
                "Cannot convert DescriptorValue to scaler " 
                "as the value varies across extensions \n"
                "-------------------------------------\n"
                + self.info()
                )
        # got here then all values were identical
        return value

    def convert_value_to(self, new_units, new_type = None):
        # retval = self.unit.convert(self._val, new_units)
        newDict = copy(self.dict_val)
        for key in self.dict_val:
            val = self.dict_val[key]
            newval = self.unit.convert(val, new_units)
            if new_type:
                newval = new_type(newval)
            else:
                if self.pytype != None:
                    newval = self.pytype(newval)
            newDict.update({key:newval})
            
        if new_type:
            pytype = new_type
        else:
            pytype = self.pytype
            
        retval = DescriptorValue(   newDict, 
                                    unit= new_units, 
                                    pytype = pytype,
                                    name = self.name,
                                    format = self.format)
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
        if self._val == None:
            if convert_values is True and as_type != None:
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
            elif convert_values is False and as_type != None:
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
                return str(self)
        
        # case where value is a single ext fits (int, str, float)
        elif as_type != None:
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
            return self.pytype(self._val)
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
    
    def get_value(self, as_type=None, convert_values=False):
        return self.as_pytype(as_type=as_type, convert_values=convert_values)

    def info(self):
        dvstr = ""
        print("\nDescriptor Value Info:")
        print("\t.name             = %s" % self.name)
        print("\t._val              = %s" % repr(self._val))
        print("\ttype(._val)        = %s" % type(self._val))
        print("\t.pytype           = %s" % str(self.pytype))
        print("\t.unit             = %s" % str(self.unit))
        keys = self.dict_val.keys()
        keys.sort()
        lkeys = len(keys)
        count = 0
        for key in keys:
            if count == 0:
                print("\t.dict_val         = {%s:%s," % \
                    (str(key), repr(self.dict_val[key])))
            elif count == lkeys - 1:
                print("%s%s:%s}" % (" " * 29, str(key),\
                    repr(self.dict_val[key])))
            else: 
                print("%s%s:%s," % (" " * 29, str(key),\
                    repr(self.dict_val[key])))
            count += 1
    
    def collapse_value(self):
        oldvalue = None
        for key in self.dict_val:
            value = self.dict_val[key]
            if oldvalue == None:
                oldvalue = value
            else:
                if oldvalue != value:
                    self._val = None
                    return None
        # got here then all values were identical
        self._val = value
        return value
    
    def overloaded(self, other):
        val = self.collapse_value()
        
        if val == None:
            mes =  "DescriptorValue contains complex result (differs for"
            mes += "different extension) and cannot be used as a simple "
            mes += str(self.pytype)
            raise Errors.DescriptorValueTypeError(mes)
        else:
            if type(val) != self.pytype:
                val = self.as_pytype()
        myfuncname = whocalledme()
        
        if isinstance(other, DescriptorValue):
            other = other.as_pytype()
        if True: # always try the following and let it raise #hasattr(val, myfuncname):
            try:
                op = None
                hasop = eval('hasattr(self.%s,"operation")' % myfuncname)
                if hasop: 
                    op = eval("self.%s.operation" % myfuncname)
                if op:
                    retval = eval(op)
                else:
                    print "D306:",myfuncname
                    raise "problem"
                    retval = eval("val.%s(other)" % myfuncname)
                return retval
            except TypeError, e:
                # print "D296: I'm here"
                raise Errors.DescriptorValueTypeError(str(e))
        else:
            raise Errors.DescriptorValueTypeError("Unsupported operand, %s, for types %s and %s"
                        % (myfuncname, str(self.pytype), str(type(other))))
        print "DISASTERDISASTERDISASTERDISASTERDISASTERDISASTERDISASTER"
#        mytype = self.pytype
#        if isinstance(other, DescriptorValue):
#            other = other.as_pytype()
#            #print "D282:", other, type(other), self.as_pytype(), type(self.as_pytype()), self.pytype
#        othertype = type(other)
#        
#        if mytype == float and othertype == int:
#            outtype = float
#        elif mytype == int and othertype == float:
#            outtype = float
#        elif mytype == str:
#            outtype = str
#        else:
#            # by default, we use our type
#            outtype = self.pytype
#        
#        # convert other to the target type (possibly coerced)
#        if mytype != str:
#            other = outtype(other)
#        
#        myfuncname = whocalledme()
#        
#        if myfuncname =="__cmp__":
#            otherfuncname = "__cmp__"
#        else:
#            if myfuncname[0:3] == "__r" and myfuncname != "__rshift__":
#                otherfuncname = "__" + myfuncname[3:]
#            else:
#                otherfuncname = "__r"+myfuncname[2:]
#
#        
#        #print "D273:", myfuncname, "->", otherfuncname
#        if hasattr(self.as_pytype(), myfuncname):
#            evalstr = "self.as_pytype().%s(other)" % myfuncname
#            retval = eval(evalstr)
#            retval = outtype(retval)
#            return retval
#        elif hasattr(other, otherfuncname):
#            evalstr = "other.%s(outtype(self))" % otherfuncname
#            #print "D295:", evalstr
#            retval = eval(evalstr)
#            return retval
#
#        raise Errors.IncompatibleOperand("%s has no method %s" % (str(type(other)),otherfuncname))
    
    
    def overloaded_cmp(self,other):
        val = self.collapse_value()
        
        if val == None:
            mes =  "DescriptorValue contains complex result (differs for "
            mes += "different extension) and cannot be used as a simple"
            mes += str(self.pytype)
            raise Errors.DescriptorValueTypeError(mes)
        else:
            if type(val) != self.pytype:
                val = self.as_pytype()
                
        #myfuncname = whocalledme()
        
        if isinstance(other, DescriptorValue):
            other = other.as_pytype()

        othertype = type(other)
        
        mine = self.as_pytype()
        
        if hasattr(mine, "__eq__"):
            try:
                retval = eval ("mine == other")
                if retval == True:
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
        raise Errors.IncompatibleOperand("%s has no method %s" % (str(type(other)),myfuncname))
    

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
    __ror__.operation  = "other | val"
    def __rrshift__(self, other):
        return self.overloaded(other)
    __rrshift__.operation = "other >> val"
    def __rshift__(self, other):
        return self.overloaded(other)
    __rshift__.operation  = "val >> other" 
    def __rxor__(self, other):
        return self.overloaded(other)
    __rxor__.operation = "other ^ val"
    def __xor__(self, other):
        return self.overloaded(other)
    __xor__.operation = "val ^ other"
    
    #def __gt__(self, other):
#        return self.overloaded(other)
#    __gt__.operation = "val > other" 
#    def __lt__(self, other):
#        return self.overloaded(other)
#    __lt__.operation = "val < other"
#    def __eq__(self, other):
#        return self.overloaded(other)
#    __eq__.operation = "val == other"
#    

# calculatorIndexREMask used to identify descriptorIndex files
# these files need to set descriptorIndex to a dictionary value
# relating AstroType names to descriptor calculator names, with the
# latter being of proper form to "exec" in python. 

calculatorIndexREMask = r"calculatorIndex\.(?P<modname>.*?)\.py$"

if (True):
    #note, the firstrun logic runs but is not needed, python only imports once
    # this module operates like a singleton
    
    centralCalculatorIndex = {}
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
                
                    # note, it might be confusing to find out if
                    # one index entry stomps another... so I'm going to 
                    # check that this dict doesn't have keys already
                    # in the central dict
                
                    for key in calculatorIndex.keys():
                        if centralCalculatorIndex.has_key(key):
                            # @@log
                            msg = "Descriptor Index CONFLICT\n"
                            msg += "... type %s redefined in\n" % key
                            msg += "... %s\n" % fullpath
                            msg += "... was already set to %s\n" %centralCalculatorIndex[key]
                            msg += "... this is a fatal error"
                            raise Errors.DescriptorsError(msg)
                        
                    centralCalculatorIndex.update(calculatorIndex)

firstrun = False

# this is down here for a good reason... it imports the
# globalStdkeyDict from StandardDescriptorKeyDict.py, which was moved to the
# descriptors subdirectory on the command of the Feb 2008 Descriptors Code Review.
from Calculator import Calculator

# Module Level Function(s)
def get_calculator(dataset):
    """ This function gets the Calculator instance appropriate for 
    the specified dataset.
    Conflicts, arising from Calculators being associated with more than one
    AstroData type classification, are resolved by traversing the type tree to see if one
    type is a subtype of the other so the more specific type can be
    used.
    @param dataset: the dataset to load a calculator for
    @type dataset: AstroData
    @returns: the appropriate Calculator instance for this type of dataset
    @rtype: Calculator
    
    @note: OPEN ISSUE: how to deal with conflicts not resolved this way... i.e.
        if there are two assignments related to types which do not appear
        in the same type trees.
        
    """
    #NOTE: Handle hdulist as well as AstroData instance as 'dataset'?
    types = dataset.discover_types()
    calcs = []
    
    # use classification library from dataset's context
    cl = dataset.get_classification_library()
    
    calc = None
    for typ in types:
        try:
            newcalc = centralCalculatorIndex[typ]
            newcalctype = typ
            calcs.append(newcalc)
            if (calc == None):
                calc = newcalc
                calctype = newcalctype
            else:
                # if the new calc is related to a type that the old
                nt = cl.get_type_obj(newcalctype)
                if nt.is_subtype_of(calctype):
                # subtypes "win" calculator type assignment "conflicts"
                    calc = newcalc
                    calctype = newcalctype
                else:
                    ot = cl.get_type_obj(calctype)
                    if not ot.is_subtype_of(nt):
                        # if more than one type applies, they must have a subtype
                        raise Errors.DescriptorsError()
        except KeyError:
            pass  # just wasn't in dictionary, no problem, most types don't have
                  #  calculators

    # by now we should have calc, and calctype, both strings... get the object
    # first, if none were found, create the default, base Calculator
    # NOTE: the base calculator looks up the descriptor in the PHU or EXTENSIONS 
    # as appropriate for that descriptor
    
    # to the descriptor type.
    if (len(calcs) == 0):
        #then none were found, use default calculator
        return Calculator()
    # first check loadedDescriptorIndex
    elif loadedCalculatorIndex.has_key(calctype):
        return loadedCalculatorIndex[calctype]
    else:
        # if here we need to import and instantiate the basic calculator
        # note: module name is first part of calc string
        modname = calc.split(".")[0]
        exec "import " + modname
        calcObj = eval (calc)
        # add this calculator to the loadedCalculatorIndex (aka "calculator cache")
        loadedCalculatorIndex.update({calctype: calcObj})
        return calcObj
    
#@@DOCPROJECT@@ done pass 1
