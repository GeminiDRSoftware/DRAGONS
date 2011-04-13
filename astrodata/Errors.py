# These exceptions allow the distinction between the None values given when a
# descriptor either cannot find the keyword, cannot do a calculation with the
# keyword etc. 

class Error(Exception):
    """Base class for exceptions in this module."""
    message = None
    def __init__(self, message=None):
        if message != None:
            self.message = message
            
    def __str__(self):
        return self.message
        
    def __repr__(self):
        return self.__class__.__name__+"(%s)" % repr(self.message)

# The following exceptions are listed alphabetically
class ArithError(Error):
    """
    For general Exceptions raised within the arith.py toolbox
    """
    message = 'Exception Raised in arith toolbox'
    
class CalcError(Error):
    """
    Exception raised for instances when the keyword required for calculation
    within a descriptor function is found but the descriptor code is unable to
    calculate and return a value
    """
    message = "Descriptor unable to calculate value"

class DescriptorDictError(Error):
    """
    Exception raised for instances where the asDict=True parameter should be
    used, but isn't
    """
    message = "Please use asDict=True to obtain a dictionary"

class DescriptorTypeError(Error):
    """
    Exception raised for instances when a descriptor function cannot return a
    value for a given AstroData Type i.e., dispersion_axis for IMAGE data
    """
    message = "Unable to return a value for data of this AstroData Type"
class DescriptorValueTypeError(Error):
    pass
    
class EmptyKeyError(Error):
    """
    Exception raised for errors when a PHU keyword was found to but was empty
    or ' '.
    """    
    message = "Keyword found but the value was empty"
    
class ExistError(Error):
    """
    Exception raised for instances when a descriptor doesn't exist for a
    particular instrument
    """
    message = "Descriptor does not exist for this instrument"

class IncompatibleOperand(Error):
    """
    Exception raised when the other operand in a DescriptorValue binary 
    operator does not have the required operator.
    """
    pass
    
class InvalidValueError(Error):
    """
    Exception raised for instances when the keyword required for calculation
    within a descriptor function is found but the value of the keyword has an
    invalid value (as determined by the validation code within the descriptor
    function)
    """
    message = "Keyword value outside allowed range"

class ManagersError(Error):
    """
    For general Exceptions raised within the managers.py toolbox
    """
    message = 'Exception Raised in managers toolbox'

class TableKeyError(Error):
    """
    Exception raised for instances when a key cannot be matched in a lookup
    table
    """
    message = "Key not matched in lookup table"

class TableValueError(Error):
    """
    Exception raised for instances when a value cannot be found in a lookup
    table
    """
    message = "Value not found in lookup table"

class PrimitiveError(Error):
    """
    Exception raised for general errors inside of a primitive.
    """
    message = "An error occurred during a primitive"

class ScienceError(Error):
    """
    Exception raised for general errors inside of a user level 'science'
    function.
    """
    message = "An error occurred during a user level 'science' function"

class ToolboxError(Error):
    """
    Exception raised for general errors inside of a 'toolbox' function.
    """
    message = "An error occurred during a 'toolbox' function"
    
class UndefinedKeyError(Error):
    """
    Exception raised for errors when a PHU keyword has a value that is 
    pyfits.core.Undefined
    """
    message = "Keyword found but the value was undefined"

