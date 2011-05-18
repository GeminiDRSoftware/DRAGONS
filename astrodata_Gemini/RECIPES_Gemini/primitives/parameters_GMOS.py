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
        "default"       : "_mosaiced",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        "tag"           : ["cl_iraf","ui_advanced"],
        },
    "fl_paste":{
        "default"       : False,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        "tag"           : ["cl_iraf","ui_advanced"],
        },
    "interp_function":{
        # This is the new "user friendly" name for the geointer parameter
        "default"       : "linear",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        "tag"           : ["cl_iraf","ui_advanced"],
         },
    },
 "normalizeFlat":{
    "suffix":{
        # String to be post pended to the output of normalizeFlat
        "default"       : "_normalized",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    "fl_over":{
        "default"       : False,
        "recipeOverride": False,
        "type"          : "bool",
        "userOverride"  : False,
        "tag"           : ["cl_iraf","ui_advanced"],
        },
    "fl_trim":{
        "default"       : False,
        "recipeOverride": False,
        "type"          : "bool",
        "userOverride"  : False,
        "tag"           : ["cl_iraf","ui_advanced"],
        },
    "fl_vardq":{
        "default"       : True,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        "tag"           : ["cl_iraf","ui_advanced"],
        },
    },
 "overscanSubtract":{
    "suffix":{
        # String to be post pended to the output of overscanSubtract
        "default"       : "_overscanCorrected",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    "fl_trim":{
        "default"       : False,
        "recipeOverride": False,
        "type"          : "bool",
        "userOverride"  : False,
        },
    "fl_vardq":{
        "default"       : False,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        },
    "biassec":{
        "default"       : "[1:25,1:2304],[1:32,1:2304],[1025:1056,1:2304]",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    },
 "overscanTrim":{
    "suffix":{
        # String to be post pended to the output of overscanTrim
        "default"       : "_overscanTrimmed",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
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
    "addMDF":{
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
    "fl_over":{
        "default"       : False,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        },
    "fl_trim":{
        "default"       : False,
        "recipeOverride": False,
        "type"          : "bool",
        "userOverride"  : False,
        },
    "fl_vardq":{
        "default"       : True,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True    
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
