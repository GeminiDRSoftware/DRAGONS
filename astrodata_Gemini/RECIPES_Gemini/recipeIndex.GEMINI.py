# This dictionary defines the default recipes to use to process a dataset of a 
# given AstroData Type

localAstroTypeRecipeIndex = {
                                "F2_DARK": ["makeProcessedDark"],
                                "F2_IMAGE": ["reduce"],
                                "F2_IMAGE_FLAT": ["makeProcessedFlat"],
                                "GMOS_BIAS": ["makeProcessedBias"],
                                "GMOS_IMAGE": ["qaReduce"],
                                #"GMOS_IMAGE": ["reduce"],
                                "GMOS_IMAGE_FLAT": ["makeProcessedFlat"],
                                "GMOS_LS_FLAT": ["makeProcessedFlat"],
                                "GMOS_LS_ARC": ["makeProcessedArc"],
                                "GMOS_SPECT":["qaReduce"],
                                "NIRI_DARK": ["makeProcessedDark"],
                                "NIRI_IMAGE": ["qaReduce"],
                                "NIRI_IMAGE_FLAT": ["makeProcessedFlat"],
                            }
