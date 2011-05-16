
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
   
          
class DescriptorExcept:
    """This class is an exception class for the Descriptor module"""
    def __init__(self, msg="Exception Raised in Descriptor system"):
        """This constructor accepts a string C{msg} argument
        which will be printed out by the default exception 
        handling system, or which is otherwise available to whatever code
        does catch the exception raised.
        @param msg: a string description about why this exception was thrown
        @type msg: string
        """        
        self.message = msg
    def __str__(self):
        """This string operator allows the default exception handling to
        print the message associated with this exception.
        @returns: string representation of this exception, the self.message member
        @rtype: string"""
        return self.message
dExcept = DescriptorExcept

class DescriptorValueBadCast(DescriptorExcept):
    pass
    
# NOTE: to address the issue of Descriptors module being a singleton, instead of the 
# approach used for the ClassificationLibrary, we use the descriptors module
# itself as the singleton and thus these module level "globals" which serve
# the purpose of acting as a central location for Descriptor behavior.
firstrun = True



class DescriptorValue():
    dictVal = None
    val = None
    name = None
    pytype = None
    unit = None
    def __init__(self,  initval, 
                        format = None, 
                        asDict = None, 
                        name = "unknown", 
                        ad = None, 
                        pytype = None,
                        unit = None):
        # print "DV82:", repr(unit)
        if pytype == None and self.pytype == None:
            self.pytype = pytype = type(initval)
        originalinitval = initval
        if isinstance(initval, DescriptorValue):
            initval = initval.dictVal
            
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
            self.dictVal = initval
            val = None
        else:
            self.val = initval
            self.dictVal = {"*":initval}
        
        #NOTE:
        # DO NOT SAVE AD INSTANCE, we don't want AD instances kept in memory due to descriptor values persisting
        # DO NOT SAVE AD INSTANCE, we don't want AD instances kept in memory due to descriptor values persisting
        # DO NOT SAVE AD INSTANCE, we don't want AD instances kept in memory due to descriptor values persisting
        self.asDict = asDict
        self.name = name
        
        if format:
            self.format = format
        elif ad != None and ad.descriptorFormat:
            self.format = ad.descriptorFormat
        else:
            self.format = None
        # do after object is set up
        self.val = self.is_collapsable() # note, tricky thing, doesn't return true, returns value
    
    
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
            val = self.is_collapsable()
            if val == None:
                format = "as_dict"
            else:
                format = "value"
        
        # produce known formats
        retstr = "Unknown Format For DescriptorValue"
        if  format == "as_dict":
            retstr = str(self.dictVal)
        elif format == "db" or format == "value":
            val = self.is_collapsable()
            if val != None:
                retstr = str(val)
            else:
                parts = [str(val) for val in self.dictVal.values()]
                retstr = "+".join(parts)
        elif format == "value":
            val = self.is_collapsable()
        return retstr
    
    
    def collapse_dict_val(self):
        value = self.is_collapsable()
        if value == None:
            raise DescriptorValueBadCast("\n"
                "Cannot convert DescriptorValue to scaler " 
                "as the value varies across extensions \n"
                "-------------------------------------\n"
                + self.info()
                )
        # got here then all values were identical
        return value

   
    
    
    def convert_value_to(self, new_units, new_type = None):
        # retval = self.unit.convert(self.val, new_units)
        newDict = copy(self.dictVal)
        for key in self.dictVal:
            val = self.dictVal[key]
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
    def as_pytype(self):
        self.val = self.is_collapsable()
        if self.val == None:
            curform = self.format
            retstr =  str(self)
            return retstr
        elif self.pytype != type(self.val):
            return self.pytype(self.val)
        else:
            return self.val
    
    # alias
    forNumpy = as_pytype
            
    
    def info(self):
        dvstr = ""
        keys = self.dictVal.keys()
        keys.sort()
        for key in keys:
            
            dvstr += str(key)
            dvstr += ": "
            dvstr += str(self.dictVal[key])
            dvstr += "\n                      "
        retstr = """\
descriptor value for: %(name)s
        single value: %(val)s
    extension values: %(dictVal)s
        """ % {"name":self.name,
               "val": repr(self.val),
               "dictVal":dvstr
              }

    
    def is_collapsable(self):
        oldvalue = None
        for key in self.dictVal:
            value = self.dictVal[key]
            if oldvalue == None:
                oldvalue = value
            else:
                if oldvalue != value:
                    self.val = None
                    return None
        # got here then all values were identical
        self.val = value
        return value
    
        
    def overloaded(self, other):
        val = self.is_collapsable()
        
        if val == None:
            raise Errors.DescriptorValueTypeError("DescriptorValue contains complex result (differs for different extension) and cannot be used as a simple %s" % str(self.pytype))
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
        val = self.is_collapsable()
        
        if val == None:
            raise Errors.DescriptorValueTypeError("DescriptorValue contains complex result (differs for different extension) and cannot be used as a simple %s" % str(self.pytype))
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
                            raise DescriptorExcept(msg)
                        
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
                        raise dExcept()
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
