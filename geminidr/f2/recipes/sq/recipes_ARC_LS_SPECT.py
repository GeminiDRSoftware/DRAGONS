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

    # Added p.flatCorrect with a new parameter "rectify" set to "False" as a temporary
    # workaround for improving distortion model in the frames with
    # large unilluminated areas by masking the regions beyond the slit into which
    # some lines may be traced.

    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.darkCorrect()
    p.flatCorrect(rectify=False) # temporaty workaround
    p.makeIRAFCompatible()
    p.determineWavelengthSolution()
    p.determineDistortion(debug=True)
    p.storeProcessedArc()
    p.writeOutputs()


_default = makeProcessedArc
