"""
Demo recipes.

All recipes are invoked indentically and test recipes working with 
the new package structure prototype.

'p' is an instance of a usuable GeminiDR instrument primitive set.

Usage:

>>> from GMOS.primitives.primitives_GMOS import PrimitivesIMAGE
>>> from astrodata import AstroData
>>> ad = AstroData('<filename.fits>')
>>> p = PrimitivesIMAGE(ad)
>>> qaReduce(p)

:parameter p: instance of primitives_IMAGE
:type      p: <instance>, GMOS.primitives.primitives_GMOS.PrimitivesIMAGE

"""

# Test recipe demo working with the new package structure. 
# 'p' is an instance of a usuable GeminiDR package primitive set.
# E.g.,
#
#     from GMOS.primitives import primitives_IMAGE
#     p = primitives_IMAGE.PrimitivesIMAGE()

# The prototype primitives merely display values available to them, such as 
# file inputs, parameters. Some, like ADUToElectrons, do perform their full 
# task here in this demonstration package.
# 17-06-2016 kra

# The 'default' recipe is simply an alias for one recipe and can be changed to point
# to any other recipe herein.
# This defines the default operating recipe (i.e. when a recipe is not specified 
# by a caller, either as a `reduce` command line option (-r) or set as an attribute 
# on a ReduceNH instance.

def qaReduce(p):
    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.detectSources()
    p.measureIQ(display=True)
    p.measureBG()
    p.measureCCAndAstrometry()
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.flatCorrect()
    p.mosaicDetectors()
    p.makeFringe()
    p.fringeCorrect()
    p.detectSources()
    p.measureIQ(display=True)
    p.measureBG()
    p.measureCCAndAstrometry()
    p.addToList(purpose='forStack')
    return

default = qaReduce

def qaStack(p):
    p.getList(purpose='forStack')
    p.correctWCSToReferenceFrame()
    p.alignToReferenceFrame()
    p.correctBackgroundToReferenceImage()
    p.stackFrames()
    p.detectSources()
    p.measureIQ(display=True)
    p.measureBG()
    p.measureCCAndAstrometry(correct_wcs=True)
    return


def qaReduceAndStack(p):
    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.detectSources()
    p.measureIQ(display=True)
    p.measureBG()
    p.measureCCAndAstrometry()
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.flatCorrect()
    p.mosaicDetectors()
    p.makeFringe()
    p.fringeCorrect()
    p.detectSources()
    p.measureIQ(display=True)
    p.measureBG()
    p.measureCCAndAstrometry()
    p.alignAndStack()
    p.detectSources()
    p.measureIQ(display=True)
    p.measureBG()
    p.measureCCAndAstrometry(correct_wcs=True)
    return


def reduce(p):
    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.flatCorrect()
    p.mosaicDetectors()
    p.makeFringe()
    p.fringeCorrect()
    p.alignAndStack()
    return
