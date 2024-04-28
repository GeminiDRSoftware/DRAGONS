"""
"""
from igrinsdr.igrins.primitives_igrins import Igrins

recipe_tags = {'IGRINS', 'STANDARD'}
# recipe_tags = {'IGRINS'}

def makeStd(p: Igrins):
    """

    Parameters
    ----------
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.
    """

    p.checkCALDB(caltypes=("processed_flat", "processed_arc"))
    p.prepare(require_wcs=False)
    p.addDQ()

    # FIXME we need to figure out how to inject badpix mask from the flat.

    # FIXME check if read-noise is really added. It seems not.
    p.addVAR(read_noise=True, poisson_noise=True)

    # ADUToElectrons requires saturation_level and nonlinearity_level in the
    # header. Since IGRINS does not have these values defined, we add them
    # here.
    # p.fixIgrinsHeader()
    # p.referencePixelsCorrect()
    p.ADUToElectrons()
    #p.nonlinearityCorrect()

    p.makeAB()
    p.estimateSlitProfile()
    p.extractStellarSpec()

    # p.stackFrames(stream="A")
    # p.stackFrames(stream="B")
    # p.subtractAB(streamA="A", streamB="B")
    return

_default = makeStd
