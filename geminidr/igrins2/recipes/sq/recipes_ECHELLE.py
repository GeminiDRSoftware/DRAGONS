"""
Recipes available to data with tags ['IGRINS', 'ECHELLE']
Default is "reduce".
"""
from igrinsdr.igrins.primitives_igrins import Igrins

recipe_tags = {'IGRINS', 'ECHELLE'}

def reduceScience(p: Igrins):
    """
    This recipe processes IGRINS science data.

    Parameters
    ----------
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.nonlinearityCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.referencePixelsCorrect()
    p.flatCorrect()
    p.attachWavelengthSolution()

    # The current IGRINS way, stack all A, all B, then A-B
    p.separateAB()   # creates "A" stream and "B" stream, new primitive
    p.stackFrames(stream="A")    # inputs from "A", outputs to "A"
    p.stackFrames(stream="B")  # inputs from "B", outputs to "B"
    p.subtractAB()   # uses "A" and "B" streams internally, outputs to main, new primitive

    # Rectify 2D and write to disk before continuing
    p.distortionCorrect(outstream="2D")  # possibly new IGRINS primitive
    p.writeOutputs(stream="2D")

    # Continuing with extraction
    p.estimateSlitProfile()
    p.extractSpectra()
    p.storeProcessedScience()

_default = reduceScience

def reduceStandard(p):
    """
    This recipe processes IGRINS standard star.

    Parameters
    ----------
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.nonlinearityCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.referencePixelsCorrect()
    p.flatCorrect()
    p.attachWavelengthSolution()

    # The current IGRINS way, stack all A, all B, then A-B
    p.separateAB()   # creates "A" stream and "B" stream, new primitive
    p.stackFrames(stream="A")    # inputs from "A", outputs to "A"
    p.stackFrames(stream="B")  # inputs from "B", outputs to "B"
    p.subtractAB()   # uses "A" and "B" streams internally, outputs to main, new primitive

    # Rectify 2D and write to disk before continuing
    p.distortionCorrect(outstream="2D")
    p.writeOutputs(stream="2D")

    # Continuing with extraction
    p.estimateSlitProfile()
    p.extractSpectra()
    p.normalizeStandard()
    p.storeProcessedStandard()
