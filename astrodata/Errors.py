class Error(Exception):
    """Base class for exceptions in this module."""
    def __init__(self, message=None):
        if message != None:
            self.message = message
    
    def __str__(self):
        return self.message
    
    def __repr__(self):
        return self.__class__.__name__+"(%s)" % repr(self.message)

class FatalDeprecation(Error):
    pass
    
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

class AstroDataReadonlyError(AstroDataError):
    """
    For general Exceptions raised in the AstroData Class Module
    """
    message = 'Readonly Exception Raised in AstroData.py'

class ADCCCommunicationError(Error):
    message = "ADCCCommunicationError"

class CalibrationDefinitionLibraryError(Error):
    """
    Exceptions raised for CallibrationDefinitionLibrary.py
    """
    message = "Exception raised in CalibrationDefinitionLibrary.py"

class DescriptorsError(Error):
    """
    For general exceptions raised in the Descriptors.py module
    """
    message = "Exception raised in Descriptors.py"

class DescriptorValueError(Error):
    pass

class DescriptorValueTypeError(Error):
    pass
    
class ExtTableError(Error):
    """
    General exception raised for Errors in ExtTable.py
    """
    message = "Exception raised in ExtTable.py"

class FatalDeprecation(Error):
    """
    Raised when code that isn't supposed to be called anymore is called anyway.
    """
    message = "Fatal Deprecation"

class gemutilError(Error):
    """
    General exception raised for Errors in adutils/gemutil.py
    """
    message = "Exception raised in gemutil.py"

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
    
class MatchShapeError(Error):
    """
    Exception raised for instances when two arrays do not match in shape
    """
    message = "The two arrays do not match in shape"

class NoLoggerError(AstroDataError):
    message = "Request for Logger preceded creation of logger."
    
class OutputError(Error):
    """
    Exception raised for instances when an output value is None or empty
    """
    message = "Output is None or empty"

class OutputExists(AstroDataError):
    def __init__(self, msg=None):
        if msg == None:
            self.message = "Cannot overwrite existing file."
        else:
            self.message = "Cannot overwrite existing file, " + msg

class PrimitiveError(Error):
    """
    Exception raised for general errors inside of a primitive.
    """
    message = "An error occurred during a primitive"

class PrimInspectError(Error):
    """
    Exception raised for general errors inside PrimInspect.py.
    """
    message = "An error occurred in PrimInspect.py"

class ReduceError(Error):
    """
    Exception raised for general errors in reduce.py
    """
    message = "An error occurred in reduce.py"
    

class ScienceError(Error):
    """
    Exception raised for general errors inside of a user level 'science'
    function.
    """
    message = "An error occurred during a user level 'science' function"

class SingleHDUMemberExcept(AstroDataError):
    def __init__(self, msg=None):
        if msg == None:
            tmpmes = "This member or method can only be called for"
            tmpmes += " Single HDU AstroData instances"
            self.message = tmpmes
        else:
            self.message = "SingleHDUMember: " + msg     

class ToolboxError(Error):
    """
    Exception raised for general errors inside of a 'toolbox' function.
    """
    message = "An error occurred during a 'toolbox' function"
    
class GDPGUtilError(Error):
    message = "Error in gpdgutil module."

# Descriptor exception; these exceptions allow the distinction between the None
# values given when a descriptor either cannot find the keyword, cannot do a
# calculation with the keyword etc. 

class DescriptorError(Error):
    """
    Base class for descriptor exceptions
    """
    message = "Descriptor error"

class CalcError(DescriptorError):
    """
    Exception raised for instances when the keyword required for calculation
    within a descriptor function is found but the descriptor code is unable to
    calculate and return a value
    """
    message = "Descriptor unable to calculate value"

class CorruptDataError(DescriptorError):
    """
    Exception raised for instances when an AstroData object does not contain
    any pixel data extensions
    """
    message = "The AstroData object does not contain any pixel data extensions"

class DescriptorTypeError(DescriptorError):
    """
    Exception raised for instances when a descriptor function cannot return a
    value for a given AstroData Type i.e., dispersion_axis for IMAGE data
    """
    message = "Unable to return a value for data of this AstroData Type"

class EmptyKeyError(DescriptorError):
    """
    Exception raised for errors when a PHU keyword was found to but was empty
    or ' '.
    """
    message = "Keyword found but the value was empty"
    
class ExistError(DescriptorError):
    """
    Exception raised for instances when a descriptor doesn't exist for a
    particular instrument
    """
    message = "Descriptor does not exist for this instrument"

class InvalidValueError(DescriptorError):
    """
    Exception raised for instances when the keyword required for calculation
    within a descriptor function is found but the value of the keyword has an
    invalid value (as determined by the validation code within the descriptor
    function)
    """
    message = "Keyword value outside allowed range"

class TableKeyError(DescriptorError):
    """
    Exception raised for instances when a key cannot be matched in a lookup
    table
    """
    message = "Key not matched in lookup table"

class TableValueError(DescriptorError):
    """
    Exception raised for instances when a value cannot be found in a lookup
    table
    """
    message = "Value not found in lookup table"

class UndefinedKeyError(DescriptorError):
    """
    Exception raised for errors when a PHU keyword has a value that is 
    pyfits.core.Undefined
    """
    message = "Keyword found but the value was undefined"
    
# Configuration system errors

class ConfigurationError(Error):
    message = "Configuration Error"

class RecipeImportError(ConfigurationError):
    message = "Recipe Import Error"
    
# High level errors (e.g. for reduce to throw)

class RecipeNotFoundError(Error):
    message = "Recipe not found."
