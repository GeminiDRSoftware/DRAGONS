# These exceptions allow the distinction between the None values given when a
# descriptor either cannot find the keyword, cannot do a calculation with the
# keyword etc.

class Error(Exception):
    """Base class for exceptions in this module."""
    pass
    
    
    def __str__(self):
        return self.message
        
    def __init__(self, message = None):
        if message != None:
            self.message = message
            
class CalcError(Error):
    """
    Exception raised for instances when the keyword is found but the value of
    the keyword cannot be handled by the descriptor code
    """
    message = "Keyword value outside allowed range"
    
class ExistError(Error):
    """
    Exception raised for instances when the keyword doesn't exist and the
    value hasn't yet been decided
    """

    message = "Keyword value not yet determined for this instrument"

class DescriptorListError(Error):
    """
    Exception raised for instances where the asList=True parameter should be
    used, but isn't
    """
    message = "Please use asList=True to obtain a list"
