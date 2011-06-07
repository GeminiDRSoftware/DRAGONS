try:
    from StandardDescriptorKeyDict import globalStdkeyDict
except ImportError:
    globalStdkeyDict = {}
    pass
import inspect  
from copy import copy
class CalculatorExcept:
    """This class is an exception class for the Calculator module"""
    
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

class Calculator(object):
    """
    The Descriptor Calculator in an object with one member fucntion for 
    each descriptor (where descriptors are conceptually types of statistical
    or other values which can be thought of as applying to all data. In practice
    a descriptor might not be 100% general, it may apply only to a vast majority
    of data types, or require some generic handling.
    Though
    in practice some descriptors may still not really apply to some data types
    they will return a valid descriptor value (e.g. if there was an instrumen without
    a filter wheel, data from that instrument would still return a sensible value
    for the filter descriptor ("none" or "blank").
    
    A Calculator is associated with particular classifications of data, such that
    it can be assigned to AstroData instances cooresponding to that classification.
    It is important that configurations not lead to multiple Calculators associated
    with one DataSet (this can happen since AstroData instances have more than one
    classification can have a Calculator associated with it.  The system handles one
    case of this where one of the two types contributing a Calculator is a subtype
    of the other, in which case the system uses the subtypes descriptor, assuming it
    "overrides" the other.  In any other case the system will throw and exception
    when trying to assign the calculator to the AstroData instance. 
    
    @note: the base class, besides being a parent class for defining new Calculators
    is also the default Calculator for when none is specifically assigned. 
    It uses "StandardDescriptorKeyDict.py"
    to map the standard key names for descriptors to specific header keys, 
    then does the retrieval from the headers in the dataset, as appropriate.
    Ideally this method should work for all prepared data, at which point we would like
    to have stored the standard values in the data header where it is directly retrieved
    rather than calculated.
    
    @ivar usage: Used to document this Descriptor.    
    """
    usage = ""
    throwExceptions = True 
        
    stdkey_dict = None
    _specifickey_dict = None
    _update_stdkey_dict = None
    
    def __init__(self):
        #print "C69: %s \n%s \n%s" % (repr(self), 
        #                         repr(self._update_stdkey_dict),
        #                         repr(self.stdkey_dict) )
        stdkeydict = copy(globalStdkeyDict)
        
        selfmro = list(inspect.getmro(self.__class__))
        #print "C75:", repr(selfmro)
        selfmro.reverse()
        for cls in selfmro:
            if not hasattr(cls, "_update_stdkey_dict"):
                #print "C78:", repr(cls)
                continue
            if cls._update_stdkey_dict != None:
                #print "C81: updated",repr(cls), repr(cls._update_stdkey_dict)
                stdkeydict.update(cls._update_stdkey_dict)
            else:
                pass
                #print "C85: no update for", repr(cls)
        self._specifickey_dict = stdkeydict
        self.stdkey_dict = stdkeydict
        # print "C79:", self._specifickey_dict["key_central_wavelength"]
        
        
    def get_descriptor_key(self, name):
        try:
            return self.stdkey_dict[name]
        except KeyError:
            return None
            
    
