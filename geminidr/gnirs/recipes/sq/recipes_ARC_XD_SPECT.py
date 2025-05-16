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
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.flatCorrect()
    p.stackFrames()
    p.attachPinholeModel()
    p.determineWavelengthSolution()
    p.determineDistortion(spatial_order=1, step=4)
    p.storeProcessedArc()


_default = makeProcessedArc
