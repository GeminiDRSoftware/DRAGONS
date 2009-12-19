# These exceptions allow the distinction between the None values given when a
# descriptor either cannot find the keyword, cannot do a calculation with the
# keyword etc.

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class CalcError(Error):
    """
    Exception raised for instances when the keyword is found but the value of
    the keyword cannot be handled by the descriptor code
    """

    def __init__(self, message = "Keyword value outside allowed range"):
        self.message = message

class ExistError(Error):
    """
    Exception raised for instances when the keyword doesn't exist and the
    value hasn't yet been decided
    """

    def __init__(self, message = "Keyword value not yet determined for this instrument"):
        self.message = message

