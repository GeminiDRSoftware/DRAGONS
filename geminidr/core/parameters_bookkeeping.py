# This parameter file contains the parameters related to the primitives located
# in the primitives_bookkeeping.py file, in alphabetical order.

from geminidr import ParametersBASE

class ParametersBookkeeping(ParametersBASE):
    addToList = {
        "purpose"               : None,
    }
    getList = {
        "max_frames"            : None,
        "purpose"               : None,
    }
    selectFromInputs = {
        "tags"                  : None,
    }
    showInputs = {
        "purpose"               : "primitive",
    }
    showList = {
        "purpose"               : 'all',
    }
    writeOutputs = {
        "clobber"               : True,
        "outfilename"           : None,
        "prefix"                : '',
        "strip"                 : False,
        "suffix"                : '',
    }