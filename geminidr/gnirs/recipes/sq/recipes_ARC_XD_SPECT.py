"""
Recipes available to data with tags ['GNIRS', 'SPECT', 'XD'],
excluding data with tags ['FLAT', 'DARK', 'BIAS'].
These are GNIRS cross-dispersed arc-lamp or sky-line calibrations.
Default is "makeProcessedArc".
"""
recipe_tags = {'GNIRS', 'SPECT', 'XD', 'ARC'}

def makeProcessedArc(p):
    """
    Process GNIRS cross-dispersed arc and calculate wavelength and distortion
    solutions.  Arcs get stacked if more than one is given.

    Inputs are:
      * raw arc
      * processed flat
    """
    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.stackFrames()
    # p.makeIRAFCompatible()  # no need for XD
    p.flatCorrect()
    p.attachPinholeModel()
    p.determineWavelengthSolution()
    p.determineDistortion(debug=True, spatial_order=1, step=4)
    p.storeProcessedArc()


_default = makeProcessedArc
