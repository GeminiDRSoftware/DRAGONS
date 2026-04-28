"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS', 'ARC'].
These are GMOS longslit arc-lamp calibrations.
Default is "makeProcessedArc".
"""
recipe_tags = {'GMOS', 'SPECT', 'LS', 'ARC'}


def makeProcessedArc(p):
    p.prepare(require_wcs=False)
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.mosaicDetectors()
    p.determineWavelengthSolution()
    p.determineDistortion()
    p.storeProcessedArc()
    p.writeOutputs()

def checkProcessedArc(p):
    """
    Extracts some values from the .WAVECAL extension for instrument monitoring
    :param p:
    :return:
    """

    p.monitorWavelengthSolution()
    p.writeOutputs()

def checkArc(p):
    """
    Reduce an arc, extract some values from the .WAVECAL extension for
    instrument monitoring
    :param p:
    :return:
    """
    # In QA mode, if determineWavelengthSolution cannot find a good arc fit it
    # uses a linear model based on the headers, whereas in SQ mode it raises an
    # exception. For the purposes of checking arcs, we want everything to
    # behave as if it is in SQ mode, so we just set that here
    p.mode = 'sq'

    p.prepare(require_wcs=False)
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.mosaicDetectors()
    # We force center=None to ensure it always gets the central row.
    # This is the default anyway, but this prevents it being overridden.
    p.determineWavelengthSolution(center=None)
    # We don't need determineDistortion to check Arcs, and it's slow.
    # p.determineDistortion()
    p.monitorWavelengthSolution()
    p.writeOutputs()

_default = makeProcessedArc
