# This dictionary defines the default recipes to use to process a dataset of a 
# given AstroData Type

localAstroTypeRecipeIndex = {
                                "GMOS_IMAGE":["qaReduce"],
                                "GMOS_SPECT":["qaReduce"],
                                "F2_DARK": ["makeProcessedDark"],
                                "F2_IMAGE": ["reduce"],
                                "F2_IMAGE_FLAT": ["makeProcessedFlat"],
                                "GMOS_IMAGE_FLAT": ["makeProcessedFlat"],
                                "GMOS_LS_FLAT": ["makeProcessedFlat"],
                                "GMOS_LS_ARC": ["makeProcessedArc"],
                                "GMOS_BIAS": ["makeProcessedBias"],
                            }
