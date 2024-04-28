"""
"""
from igrinsdr.igrins.primitives_igrins import Igrins

recipe_tags = {'IGRINS', 'SKY'}
# recipe_tags = {'IGRINS'}

def makeSky(p: Igrins):
    """
    This recipe performs the standardization and corrections needed to convert
    the raw input dark images into a single stacked dark image. This output
    processed dark is stored on disk and its information added to the calibration
    database using storeProcessedDark and has a name
    equal to the name of the first input bias image with "_bias.fits" appended.

    Parameters
    ----------
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.
    """

    p.prepare(require_wcs=False)
    p.addDQ()
    p.addVAR(read_noise=True)
    # ADUToElectrons requires saturation_level and nonlinearity_level in the
    # header. Since IGRINS does not have these values defined, we add them
    # here.
    p.fixIgrinsHeader()
    p.referencePixelsCorrect()
    p.ADUToElectrons()
    #p.nonlinearityCorrect()

    p.stackFrames()
    p.extractSimpleSpec()

    p.identifyOrders()
    p.identifyLinesAndGetWvlsol()

    # we are skipping save_orderflat step.

    p.extractSpectraMulti()
    p.identifyMultiline()
    p.volumeFit()

    p.makeSpectralMaps()

    p.storeProcessedArc()

    return

_default = makeSky
