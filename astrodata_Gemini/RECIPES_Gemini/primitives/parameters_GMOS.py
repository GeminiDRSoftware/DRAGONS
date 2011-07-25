# This parameter file contains the parameters related to the primitives located
# in the primitives_GMOS.py file, in alphabetical order.
{"mosaicDetectors":{
    "suffix":{
        # String to be post pended to the output of mosaicDetectors
        "default"       : "_mosaicked",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
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
 "standardizeHeaders":{
    "suffix":{
        # String to be post pended to the output of standardizeHeaders
        "default"       : "_sdzHdrs",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    },
"standardizeStructure":{
    "suffix":{
        # String to be post pended to the output of standardizeStructure
        "default"       : "_sdzStruct",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    "attach_mdf":{
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
        "userOverride"  : True,
        },
    "bias":{
        "default"       : None,
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    },
"subtractOverscan":{
    "suffix":{
        # String to be post pended to the output of overscanSubtract
        "default"       : "_overSub",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    "overscan_section":{
        "default"       : "",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    },
"trimOverscan":{
    "suffix":{
        # String to be post pended to the output of overscanTrim
        "default"       : "_overTrim",
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
        "userOverride"  : True,
        },
    "repair":{
        "default"       : True,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        },
    },
}
