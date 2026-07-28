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
    p.applySlitModel()
    p.attachPinholeRectification()
    p.stackFrames()
    p.determineWavelengthSolution()
    p.determineDistortion()
    p.storeProcessedArc()

def combineWavelengthSolutions(p):
    """
    Combine wavelength solutions from two different sources, such as orders
    from sky lines and the others from the lamp.

    Inputs are:
      * First processed arc
      * Second processed arc
      * List of extensions to take from the second processed arc
    """
    p.rejectInputs(at_start=1, outstream="second_arc")
    p.rejectInputs(at_end=1)
    p.combineSlices(from_stream="second_arc")
    p.storeProcessedArc(suffix='_combinedArc')
    p.writeOutputs()

_default = makeProcessedArc
