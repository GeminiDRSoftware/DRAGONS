# This parameter file contains the parameters related to the primitives located
# in the primitives_GMOS_IMAGE.py file, in alphabetical order.
{"makeFringeFrame":{
    "suffix":{
        # String to be post pended to the output of makeFringe
        "default"       : "_fringe",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    "operation":{
        "default"       : "median",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    },
 "normalize":{
    "suffix":{
        # String to be post pended to the output of normalize
        "default"       : "_normalized",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    },
# The standardizeStructure primitive is actually located in the
# primtives_GMOS.py file, but the add_mdf parameter should be set to False as
# default for data with an AstroData Type of IMAGE.
 "standardizeStructure":{
    "suffix":{
        # String to be post pended to the output of standardizeStructure
        "default"       : "_sdzStruct",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    "add_mdf":{
        "default"       : False,
        "recipeOverride": True,
        "type"          : "bool",
        "userOverride"  : True,
        },
    },
}
