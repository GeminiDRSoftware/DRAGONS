{"addDQ":{
    "suffix":{
        "default"       : "_dqAdded",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    # Currently (8/29/12) there are no spectroscopic BPMs for GMOS,
    # so turn off the BPM argument in addDQ for now.
    "bpm":{
        "default"       : None,
        # No default type defined, since the bpm parameter could be a string or
        # an AstroData object
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    },
 "appwave":{
    "suffix":{
        "default"       : "_appwave",
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
        "default"       : None,
        # No default type defined, since the arc parameter could be a string or
        # an AstroData object
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
   },
 "determineWaveCal":{
    "suffix":{
        "default"       : "_waveCalSolution",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    "linelist":{
        "default"       : None,
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    "fitfunction":{
        "default"       : None,
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    "fitorder":{
        "default"       : None,
        "type"          : "int",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    "match":{ 
        "default"       : None,
        "type"          : "float",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    "nsum":{
        "default"       : None,
        "type"          : "int",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    "ntmax":{
        "default"       : None,
        "type"          : "int",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
    "minsep":{
        "default"       : None,
        "type"          : "int",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
        },
   },
 "determineWavelengthSolution":{
    "suffix":{
        "default"       : "_wavelengthSolution",
        "type"          : "str",
        "recipeOverride": True,
        "userOverride"  : True,
        "uiLevel"       : "UIBASIC",
    },
    "interactive":{
        "default"       : False,
        "type"          : "bool",
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
