# NOTE: this directory (specified in the mkPIF.py source code, as "primdicts"
#  by default.
#  You can split the primdict into multiple files, mkPIF.py will recurse the
#  primdicts directory and update the actual dictionary with any file
#  in primdicts or a primdicts subdirectory that matches the
#  primdict file naming convention of "pif2primdict.<ANYTHING>.py"

# FORMAT
# File contains a dict, each member is keyed by the primitive name for
# which a PIF will be created. The value is another dict which provides
# metadata for the PIF, at least via a "module" key which specifies
# where the PIF should appear

{
# bookkeeping
"showInputs"                       : {"module": "bookkeeping",
                                      "copy_input": False},
"writeOutputs"                     : {"module": "bookkeeping",
                                      "copy_input": False},

# display
"display"                          : {"module": "display",
                                      "copy_input": False},

# general
"add"                              : {"module": "general"},
"divide"                           : {"module": "general"},
"multiply"                         : {"module": "general"},
"subtract"                         : {"module": "general"},

# mask
"addObjectMaskToDQ"                : {"module": "mask",
                                      "pep8name": "add_object_mask_to_dq",},
"applyDQPlane"                     : {"module": "mask",
                                      "pep8name": "apply_dq_plane"},

# photometry
"addReferenceCatalog"              : {"module": "photometry"},
"detectSources"                    : {"module": "photometry"},
"measureCCAndAstrometry"           : {"module": "photometry",
                                      "pep8name": "measure_cc_and_astrometry",},

# preprocess
"ADUToElectrons"                   : {"module": "preprocess"},
"correctBackgroundToReferenceImage": {"module": "preprocess"},
"divideByFlat"                     : {"module": "preprocess"},
"nonlinearityCorrect"              : {"module": "preprocess"},
"normalize"                        : {"module": "preprocess"},
"subtractDark"                     : {"module": "preprocess"},
"thermalEmissionCorrect"           : {"module": "preprocess"},

# qa
"measureBG"                        : {"module": "qa"},
"measureCC"                        : {"module": "qa"},
"measureIQ"                        : {"module": "qa"},

# register
"correctWCSToReferenceCatalog"     : {"module": "register",
                                      "pep8name": "correct_wcs_to_reference_catalog"},
"correctWCSToReferenceFrame"       : {"module": "register",
                                      "pep8name": "correct_wcs_to_reference_frame"},
"determineAstrometricSolution"     : {"module": "register"},
"updateWCS"                        : {"module": "register"},

# resample
"alignToReferenceFrame"            : {"module": "resample"},

# stack
"stackFrames"                      : {"module": "stack"},

# standardize
"addDQ"                            : {"module": "standardize"},
"addMDF"                           : {"module": "standardize"},
"addVAR"                           : {"module": "standardize"},
"markAsPrepared"                   : {"module": "standardize"},
"prepare"                          : {"module": "standardize"},
"standardizeGeminiHeaders"         : {"module": "standardize"},
"standardizeHeaders"               : {"module": "standardize"},
"standardizeInstrumentHeaders"     : {"module": "standardize"},
"standardizeStructure"             : {"module": "standardize"},
"validateData"                     : {"module": "standardize"},

# gmos
"mosaicDetectors"                  : {"module": "gmos",
                                      "astrotype": "GMOS"},
"overscanCorrect"                  : {"module": "gmos",
                                      "astrotype": "GMOS"},
"subtractBias"                     : {"module": "gmos",
                                      "astrotype": "GMOS"},
"subtractOverscan"                 : {"module": "gmos",
                                      "astrotype": "GMOS"},
"tileArrays"                       : {"module": "gmos",
                                      "astrotype": "GMOS"},
"trimOverscan"                     : {"module": "gmos",
                                      "astrotype": "GMOS"},

# gmos_image
"makeFringeFrame"                  : {"module": "gmos_image",
                                      "astrotype": "GMOS_IMAGE"},
"scaleByIntensity"                 : {"module": "gmos_image",
                                      "astrotype": "GMOS_IMAGE"},
"scaleFringeToScience"             : {"module": "gmos_image",
                                      "astrotype": "GMOS_IMAGE"},
"stackFlats"                       : {"module": "gmos_image",
                                      "astrotype": "GMOS_IMAGE"},
"subtractFringe"                   : {"module": "gmos_image",
                                      "astrotype": "GMOS_IMAGE"},

# gmos_spect
"attachWavelengthSolution"         : {"module": "gmos_spect",
                                      "astrotype": "GMOS_SPECT"},
"determineWavelengthSolution"      : {"module": "gmos_spect",
                                      "astrotype": "GMOS_SPECT"},
"extract1DSpectra"                 : {"module": "gmos_spect",
                                      "pep8name": "extract_1d_spectra",
                                      "astrotype": "GMOS_SPECT"},
"makeFlat"                         : {"module": "gmos_spect",
                                      "astrotype": "GMOS_SPECT"},
"rejectCosmicRays"                 : {"module": "gmos_spect",
                                      "astrotype": "GMOS_SPECT"},
"resampleToLinearCoords"           : {"module": "gmos_spect",
                                      "astrotype": "GMOS_SPECT"},
"skyCorrectFromSlit"               : {"module": "gmos_spect",
                                      "astrotype": "GMOS_SPECT"},
}
