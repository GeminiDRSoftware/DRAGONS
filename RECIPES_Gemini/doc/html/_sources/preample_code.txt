# Preamble code for primitives_GMOS_IMAGE

.. _preamble:

from primitives_GEMINI import GEMINIPrimitives

from adu2e import ADUToElectron

import os

class GMOS_IMAGEException:
    """ 
      This is the general exception the classes and functions in the
      Structures.py module raise.
    """

    def __init__(self, msg="Exception Raised in Recipe System"):

        """This constructor takes a message to print to the user."""

        self.message = msg

    def __str__(self):
        """
          This str conversion member returns the message given by the user 
          (or the default message) when the exception is not caught.
        """

        return self.message


class GMOS_IMAGEPrimitives(GEMINIPrimitives):

    def init(self, rc):

        GEMINIPrimitives.init(self, rc)

        return rc
   
    def ADU2electron(self, rc):

    # put your code here 
