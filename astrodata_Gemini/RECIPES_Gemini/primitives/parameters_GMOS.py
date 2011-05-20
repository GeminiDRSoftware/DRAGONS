# This parameter file contains the parameters related to the primitives located
# in the primitives_GMOS.py file, in alphabetical order.
{"addBPM":{
    "suffix":{
        # String to be post pended to the output of addBPM
        "default"       : "_bpmAdded",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    },
 "mosaicDetectors":{
    "suffix":{
        # String to be post pended to the output of mosaicDetectors
        "default"       : "_mosaicked",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    "tile":{
        "default"       : False,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        "tag"           : ["cl_iraf","ui_advanced"],
        },
    "interpolator":{
        "default"       : "linear",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        "tag"           : ["cl_iraf","ui_advanced"],
         },
    },
"overscanSubtract":{
    "suffix":{
        # String to be post pended to the output of overscanSubtract
        "default"       : "_overSub",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    "trim":{
        "default"       : False,
        "recipeOverride": False,
        "type"          : "bool",
        "userOverride"  : False,
        },
    "overscan_section":{
        "default"       : "[1:25,1:2304],[1:32,1:2304],[1025:1056,1:2304]",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    },
"overscanTrim":{
    "suffix":{
        # String to be post pended to the output of overscanTrim
        "default"       : "_overTrim",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    "trim":{
        "default"       : False,
        "recipeOverride": False,
        "type"          : "bool",
        "userOverride"  : False,
        },
    "overscan_section":{
        "default"       : "[1:25,1:2304],[1:32,1:2304],[1025:1056,1:2304]",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    },
 "standardizeHeaders":{
    "suffix":{
        # String to be post pended to the output of standardizeHeaders
        "default"       : "_sdzHdrs",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    },
"standardizeStructure":{
    "suffix":{
        # String to be post pended to the output of standardizeStructure
        "default"       : "_sdzStruct",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    "add_mdf":{
        "default"       : True,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        },
    },
 "subtractBias":{
    "suffix":{
        # String to be post pended to the output of subtractBias
        "default"       : "_biasCorrected",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    "bias":{
        "default"       : None,
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    },
 "validateData":{
    "suffix":{
        # String to be post pended to the output of validateData
        "default"       : "_validated",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    "repair":{
        "default"       : True,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        },
    },
}
