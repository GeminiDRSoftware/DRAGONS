# These exceptions allow the distinction between the None values given when a
# descriptor either cannot find the keyword, cannot do a calculation with the
# keyword etc.

class Error(Exception):
    """Base class for exceptions in this module."""
    def __init__(self, message = None):
        if message != None:
            self.message = message
            
    def __str__(self):
        return self.message
        
    def __repr__(self):
        return self.__class__.__name__+"(%s)" % repr(self.message)
            
class CalcError(Error):
    """
    Exception raised for instances when the keyword required for calculation
    within a descriptor function is found but the value of the keyword has an
    invalid value (as determined by the validation code within the descriptor
    function)
    """
    message = "Keyword value outside allowed range"
    
class ExistError(Error):
    """
    Exception raised for instances when the keyword doesn't exist and the
    value hasn't yet been decided
    """

    message = "Keyword value not yet determined for this instrument"

class DescriptorDictError(Error):
    """
    Exception raised for instances where the asDict=True parameter should be
    used, but isn't
    """
    message = "Please use asDict=True to obtain a dictionary"

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
    message = "Keyword value undefined, so returning None"

class EmptyKeyError(Error):
    """
    Exception raised for errors when a PHU keyword was found to but was empty
    or ' '.
    """    
    message = "Keyword value found but empty, so returning None"
