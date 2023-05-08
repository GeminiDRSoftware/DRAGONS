"""
MS: this is just an MVP, copying the corresponding GNIRS recipe.
Expect Olesja will improve this after she finds some bandwidth.  
 
Recipes available to data with tags ['NIRI', 'SPECT', 'LS'],
excluding data with tags ['FLAT', 'DARK', 'BIAS'].
These are NIRI longslit arc-lamp or sky-line calibrations.
Default is "makeProcessedArc".
"""
recipe_tags = {'NIRI', 'SPECT', 'LS', 'ARC'}

def makeProcessedArc(p):
    """
    Process NIRI longslist arc and calculate wavelength and distortion
    solutions.  No stacking, arcs are processed individually if more than
    one is given.

    Inputs are:
      * raw arc - no other calibrations required.
    """

    # Added a temporary workaround for improving distortion model in the frames with
    # large unilluminated areas by using SLITEDGE table from the processed flat
    # for masking the regions beyond the slit into which some lines may be traced.

    #p.selectFromInputs(tags="FLAT", outstream="flat") # temporary workaround
    #p.removeFromInputs(tags="FLAT") # temporary workaround
    p.prepare()
    p.addDQ()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.nonlinearityCorrect()
    #p.transferAttribute(stream="main", source="flat", attribute="SLITEDGE") # temporary workaround
    #p.maskBeyondSlit() # temporary workaround
    p.makeIRAFCompatible()
    p.determineWavelengthSolution()
    p.determineDistortion(debug=True)
    p.storeProcessedArc()
    p.writeOutputs()


_default = makeProcessedArc
