{"determineWavelengthSolution":{
    "suffix":{
        "default"       : "_wavelengthSolution",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    },
 "attachWavelengthSolution":{
    "suffix":{
        "default"       : "_wavelengthSolution",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
     "arc":{
        # No type defined here so that user can pass
        # a string (eg. from command line) or an astrodata
        # instance (eg. from a script)
        "default"       : None,
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
   },
 "extract1DSpectra":{
    "suffix":{
        "default"       : "_extracted",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    },
 "makeFlat":{
    "suffix":{
        "default"       : "_flat",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    },
 "mosaicDetectors":{
    "suffix":{
        "default"       : "_mosaicked",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    "tile":{
        "default"       : False,
        "type"          : "bool",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    "interpolate_gaps":{
        "default"       : True,
        "type"          : "bool",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    "interpolator":{
        "default"       : "linear",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    },
 "rejectCosmicRays":{
    "suffix":{
        "default"       : "_crRejected",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    },
 "resampleToLinearCoords":{
    "suffix":{
        "default"       : "_linearCoords",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    },
 "skyCorrectFromSlit":{
    "suffix":{
        "default"       : "_skyCorrected",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    },
}
