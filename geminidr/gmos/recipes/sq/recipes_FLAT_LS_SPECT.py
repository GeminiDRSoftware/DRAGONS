"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS', 'FLAT'].
These are GMOS longslit observations.
Default is "reduce".
"""
# recipe_tags = {'GMOS', 'SPECT', 'LS', 'FLAT'}
#
# from ..ql.recipes_FLAT_LS_SPECT import (makeProcessedFlatStack,
#                                         makeProcessedFlatNoStack)
#
# _default = makeProcessedFlatNoStack
#
#
# def makeProcessedSlitIllum(p):
#     p.prepare()
#     p.addDQ(static_bpm=None)
#     p.addVAR(read_noise=True)
#     p.overscanCorrect()
#     p.getProcessedBias()
#     p.biasCorrect()
#     p.ADUToElectrons()
#     p.addVAR(poisson_noise=True)
#     p.stackFrames()
#     p.makeSlitIllum()
#     p.makeIRAFCompatible()
#     p.storeProcessedSlitIllum()
