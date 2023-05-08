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

    # Added a temporary workaround for improving distortion model in the frames with
    # large unilluminated areas by using SLITEDGE table from the processed flat
    # for masking the regions beyond the slit into which some lines may be traced.

    #p.selectFromInputs(tags="FLAT", outstream="flat") # temporary workaround
    #p.removeFromInputs(tags="FLAT") # temporary workaround
    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.darkCorrect()
    #p.transferAttribute(stream="main", source="flat", attribute="SLITEDGE") # temporary workaround
    #p.maskBeyondSlit() # temporary workaround
    p.makeIRAFCompatible()
    p.determineWavelengthSolution()
    p.determineDistortion(debug=True)
    p.storeProcessedArc()
    p.writeOutputs()


_default = makeProcessedArc
