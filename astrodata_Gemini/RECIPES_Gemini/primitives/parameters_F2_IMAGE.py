# This parameter file contains the parameters related to the primitives located
# in the primitives_F2_IMAGE.py file, in alphabetical order.

{"normalize":{
    "suffix":{
        # String to be post pended to the output of normalize
        "default"       : "_normalized",
        "recipeOverride": True,
        "type"          : "str",
        "userOverride"  : True,
        },
    },
 # The standardizeStructure primitive is actually located in the
 # primtives_F2.py file, but the add_mdf parameter should be set to False as
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
