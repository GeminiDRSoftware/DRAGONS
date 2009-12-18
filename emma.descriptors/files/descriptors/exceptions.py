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

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
