# These exceptions allow the distinction between the None values given when a
# descriptor either cannot find the keyword, cannot do a calculation with the
# keyword etc. 

class Error(Exception):
    """Base class for exceptions in this module."""
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

class AstroDataError(Error):
    """
    For general Exceptions raised in the AstroData Class Module
    """
    message = 'Exception Raised in AstroData.py'

class OutputExists(AstroDataError):
    def __init__(self, msg=None):
        if msg == None:
            self.message = "Cannot overwrite existing file."
        else:
            self.message = "Cannot overwrite existing file, " + msg

class SingleHDUMemberExcept(AstroDataError):
    def __init__(self, msg=None):
        if msg == None:
            tmpmes = "This member or method can only be called for"
            tmpmes += " Single HDU AstroData instances"
            self.message = tmpmes
        else:
            self.message = "SingleHDUMember: " + msg     

class CalcError(Error):
    """
    Exception raised for instances when the keyword required for calculation
    within a descriptor function is found but the descriptor code is unable to
    calculate and return a value
    """
    message = "Descriptor unable to calculate value"

class CorruptDataError(Error):
    """
    Exception raised for instances when an AstroData object was not
    automatically assigned (EXTNAME, EXTVER) data extensions
    """
    message = "The AstroData object was not assigned a 'SCI' extension"

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

class InputError(Error):
    """
    Exception raised for instances when an input value is invalid. This
    includes whether it is None or empty
    """
    message = "Input is None or empty"
    
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

class MatchShapeError(Error):
    """
    Exception raised for instances when two arrays do not match in shape
    """
    message = "The two arrays do not match in shape"

class OutputError(Error):
    """
    Exception raised for instances when an output value is None or empty
    """
    message = "Output is None or empty"

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
