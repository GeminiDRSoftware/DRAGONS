# This parameter file contains the parameters related to the primitives located
# in the primitives_GMOS_SPECT.py file, in alphabetical order.
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
