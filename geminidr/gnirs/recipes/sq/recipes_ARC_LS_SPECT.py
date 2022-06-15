"""
Recipes available to data with tags ['GNIRS', 'SPECT', 'LS'],
excluding data with tags ['FLAT', 'DARK', 'BIAS'].
These are GNIRS longslit arc-lamp or sky-line calibrations.
Default is "makeProcessedArc".
"""
recipe_tags = {'GNIRS', 'SPECT', 'LS', 'ARC'}

def makeProcessedArc(p):
    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
  #  p.darkCorrect(do_cal=False) # opt for cals from skylines?
  #  p.flatCorrect(do_cal=False) # opt for cals from skylines?
    p.stackFrames()
    p.makeIRAFCompatible()
    p.determineWavelengthSolution()
    p.determineDistortion(debug=True)
    p.storeProcessedArc()
    p.writeOutputs()


_default = makeProcessedArc
