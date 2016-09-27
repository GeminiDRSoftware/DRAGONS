# This recipe (copied from the legacy RECIPES_Gemini) is the putative QA 
# reduction for GMOS SPECT datasets. The legacy recipe was neither implemented 
# in primitives nor used under QAP. In all liklihood, this is not useful as is, 
# but is here to serve in the prototype to test recipe selection.
#
# 25-08-2016 kra

recipe_tags = set(['GMOS', 'SPECT'])

def qaReduce(p):
    p.prepare()
    #p.addDQ()
    #p.addVAR(read_noise=True)
    #p.overscanCorrect()
    #p.biasCorrect()
    #p.ADUToElectrons()
    #p.addVAR(poisson_noise=True)
    #p.measureIQ(display=True)
    #p.rejectCosmicRays()
    #p.flatCorrect()
    p.mosaicADdetectors()
    p.getProcessedArc()
    p.attachWavelengthSolution()
    #wcalResampleToLinearCoords()
    #skyCorrectFromSlit()
    #measureIQ(display=True)
    return

default = qaReduce
