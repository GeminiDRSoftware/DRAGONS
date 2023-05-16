"""
Recipes available to data with tags ['GNIRS', 'SPECT', 'LS'],
excluding data with tags ['FLAT', 'DARK', 'BIAS'].
These are GNIRS longslit arc-lamp or sky-line calibrations.
Default is "makeProcessedArc".
"""
recipe_tags = {'GNIRS', 'SPECT', 'LS', 'ARC'}

def makeProcessedArc(p):
    """
    Process GNIRS longslist arc and calculate wavelength and distortion
    solutions.  Arcs get stacked if more than one is given.

    Inputs are:
      * raw arc - no other calibrations required.
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
    p.flatCorrect(rectify=False) # temporary workaround
    p.stackFrames()
    p.makeIRAFCompatible()
    p.determineWavelengthSolution()
    p.determineDistortion(debug=True)
    p.storeProcessedArc()
    p.writeOutputs()


_default = makeProcessedArc
