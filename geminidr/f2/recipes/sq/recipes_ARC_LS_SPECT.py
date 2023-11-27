"""
Recipes available to data with tags ['F2', 'SPECT', 'LS', 'ARC']
These are F2 longslit arc-lamp or sky-line calibrations.
Default is "makeProcessedArc".
"""
recipe_tags = {'F2', 'SPECT', 'LS', 'ARC'}

def makeProcessedArc(p):
    """
    Create F2 longslit arc solution.  No stacking, arcs are reduced
    individually.

    Inputs are:
       * raw arc(s) with caldb requests:
          * procdark with matching exptime
          * procflat for HK and K-long arcs, actually a lamp-off flat
        (Questions remaining: see google doc.)
    """

    p.prepare()
    p.addDQ()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.darkCorrect()
    p.flatCorrect()
    p.makeIRAFCompatible()
    p.determineWavelengthSolution()
    p.determineDistortion(debug=True)
    p.storeProcessedArc()
    p.writeOutputs()


_default = makeProcessedArc
