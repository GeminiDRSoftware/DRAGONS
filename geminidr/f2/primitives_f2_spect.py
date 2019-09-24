#
#                                                                  gemini_python
#
#                                                          primtives_f2_spect.py
# ------------------------------------------------------------------------------
import numpy as np

from importlib import import_module
import os

from geminidr.core import Spect
from .primitives_f2 import F2
from . import parameters_f2_spect

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class F2Spect(F2, Spect):
    """
    This is the class containing all of the preprocessing primitives
    for the F2Spect level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set(["GEMINI", "F2", "SPECT"])

    def __init__(self, adinputs, **kwargs):
        super(F2Spect, self).__init__(adinputs, **kwargs)
        self._param_update(parameters_f2_spect)

    def _get_linelist_filename(self, ext, cenwave, dw):
        lookup_dir = os.path.dirname(import_module('.__init__', self.inst_lookups).__file__)
        filename = 'lowresargon.dat'
        return os.path.join(lookup_dir, filename)