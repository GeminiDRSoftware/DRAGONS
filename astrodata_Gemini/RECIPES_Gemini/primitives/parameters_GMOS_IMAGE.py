# This parameter file contains the parameters related to the primitives located
# in the primitives_GMOS_IMAGE.py file, in alphabetical order.
{"normalizeFlat":{
    "trim":{
        "default"       : False,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : False,
        },
    "overscan":{
        "default"       : False,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        },
    },
# The standardizeStructure primitive is actually located in the
# primtives_GMOS.py file, but the addMDF parameter should be set to False as
# default for data with an AstroData Type of IMAGE.
 "standardizeStructure":{
    "suffix":{
        # String to be post pended to the output of standardizeStructure
        "default"       : "_sdzStruct",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : False,
        },
    "addMDF":{
        "default"       : False,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        },
    },
}
