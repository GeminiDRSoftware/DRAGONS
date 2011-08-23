# This parameter file contains the parameters related to the primitives located
# in the primitives_F2_IMAGE.py file, in alphabetical order.

{"normalize":{
    "suffix":{
        "default"       : "_normalized",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    },
 # The standardizeStructure primitive is actually located in the
 # primtives_F2.py file, but the attach_mdf parameter should be set to False as
 # default for data with an AstroData Type of IMAGE. 
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
}
