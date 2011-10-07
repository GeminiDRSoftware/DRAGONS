# This parameter file contains the parameters related to the primitives located
# in the primitives_GMOS_IMAGE.py file, in alphabetical order.
{"iqDisplay":{
    "suffix":{
        "default"       : "_iqMeasured",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    "frame":{
        "default"       : 1,
        "type"          : "int",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    "saturation":{
        "default"       : 58000,
        "type"          : "float",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    },
 "makeFringeFrame":{
    "suffix":{
        "default"       : "_fringe",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    "operation":{
        "default"       : "median",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    },
 "normalize":{
    "suffix":{
        "default"       : "_normalized",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    "saturation":{
        "default"       : 45000,
        "type"          : "float",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    },
 "removeFringe":{
    "suffix":{
        "default"       : "_fringeCorrected",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    "stats_scale":{
        "default"       : False,
        "type"          : "bool",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    },
 "stackFlats":{
    "suffix":{
        "default"       : "_stack",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    "grow":{
        "default"       : 0.0,
        "type"          : "float",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    "mask_type":{
        "default"       : "goodvalue",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        }, 
    "operation":{
        "default"       : "median",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    "reject_method":{
        "default"       : "minmax",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    },
# The standardizeStructure primitive is actually located in the
# primtives_GMOS.py file, but the attach_mdf parameter should be set to False
# as default for data with an AstroData Type of IMAGE.
 "standardizeStructure":{
    "suffix":{
        "default"       : "_structureStandardized",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    "attach_mdf":{
        "default"       : False,
        "type"          : "bool",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    },
 "storeProcessedFringe":{
    "suffix":{
        "default"       : "_fringe",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    }, 
}
