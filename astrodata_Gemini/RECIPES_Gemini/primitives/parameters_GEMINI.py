# This parameter file contains the parameters related to the primitives located
# in the primitives_GEMINI.py file, in alphabetical order.
{"addDQ":{
    "suffix":{
        # String to be post pended to the output of addDQ
        "default"       : "_dqAdded",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    },
 "addToList":{
    "purpose":{
        "default"       : "",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    },
 "addVAR":{
    "suffix":{
        # String to be post pended to the output of addVAR
        "default"       : "_varAdded",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    },
 "aduToElectrons":{
    "suffix":{
        # String to be post pended to the output of aduToElectrons
        "default"       : "_aduToElect",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    },
 "divideByFlat":{
    "suffix":{
        # String to be post pended to the output of divideByFlat
        "default"       : "_flatCorrected",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    "flat":{
        "default"       : None,
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    },
 "getCal":{
    "source":{
        "default"       : "all",
        "type"          : "str",
        },
    "caltype":{
        "default"       : None,
        "type"          : "str",
        },
    },
 "getList":{
    "purpose":{
        "default"       : "",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    },
 "measureIQ":{
    "function":{
        # Default can be moffat, gauss or both
        "default"       : "both",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    "display":{
        "default"       : True,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        },
    "qa":{
        # A flag to use a grid of sub-windows for detecting the sources in the
        # image frames, rather than the entire frame all at once
        "default"       : True,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        },
    },
 "nonlinearityCorrect":{
    "suffix":{
        # String to be post pended to the output of nonlinearityCorrect
        "default"       : "_nonlinCorrected",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    },
 "pause":{
    "message":{
        "default"       : "Pausing Reduction by Control System Request",
        "type"          : "string",
        "a"             : "default comes first, the rest alphabetically",
        "note1"         : "these are just test parameters...",
        "note2"         : "pause doesn't need a 'message' parameter",
        },
    },
 "scaleFringeToScience":{
    "suffix":{
        # String to be post pended to the output of scaleFringeToScience
        "default"       : "_scaled",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    "fringe":{
        "default"       : None,
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    "stats_scale":{
        "default"       : True,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        },
    },
 "showList":{
    "purpose":{
        "default"       : "",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    },
 "showParameters":{
    "test":{
        "default"       : True,
        "recipeOverride": False,
        "uiLevel"       : "debug",
        "userOverride"  : True,
        "type"          : "bool",
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
    },
 "stackFrames":{
    "suffix":{
        # String to be post pended to the output of stackFrames
        "default"       : "_stacked",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    "operation":{
        "default"       : "average",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    },
 "storeProcessedBias":{
    "clob":{
        "default"       : False,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        },
    },
 "storeProcessedFlat":{
    "clob":{
        "default"       : False,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        },
    },
 "storeProcessedFringe":{
    "clob":{
        "default"       : False,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        },
    },
 "subtractDark":{
    "suffix":{
        # String to be post pended to the output of subtractDark
        "default"       : "_darkCorrected",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    "dark":{
        "default"       : None,
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    },
 "subtractFringe":{
    "suffix":{
        # String to be post pended to the output of subtractFringe
        "default"       : "_fringeCorrected",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    "fringe":{
        "default"       : None,
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    },
 "writeOutputs":{
    "suffix":{
        "default"       : "",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    "strip":{
        "default"       : False,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        },
    "clobber":{
        "default"       : False,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        },
    "prefix":{
        "default"       : "",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    "outfilename":{
        "default"       : "",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    },
}
