# This parameter file contains the parameters related to the primitives located
# in the primitives_GEMINI.py file, in alphabetical order.

from parameters_CORE import ParametersCORE

class ParametersBookkeeping(ParametersCORE):
    addToList = {
        "purpose": None,
    }
    contextReport = {
        "report_history"    : False,
        "internal_dict"     : False,
        "context_vars"      : False,
        "report_inputs"     : False,
        "report_parameters" : False,
        "showall"           : False,
    }
    getList = {
        "purpose"           : None,
    }
    showList = {
        "purpose"           : None,
    }

    # BEGIN showParameters -- No idea what to do with this.
    showParameters = {
        "test":{
            "default"       : True,
            "type"          : "bool",
            "recipeOverride": True,
            "userOverride"  : False, #True,
            "uiLevel"       : "debug",
            "tags"          : ["test", "iraf"],
        },
        "otherTest":{
            "default"       : False,
            "userOverride"  : True,
        },
        "otherTest2":{
            "userOverride"  : True,
            "tags"          :["test", "wcslib"],
        },
    }
    # END showParameters

    sleep = {
        "duration"          : 5.0,
    }
    writeOutputs = {
        "suffix"            : None,
        "strip"             : False,
        "prefix"            : None,
        "outfilename"       : None,
    }
