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
    p.prepare()
    p.addDQ()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.nonlinearityCorrect()
    p.makeIRAFCompatible()
    p.determineWavelengthSolution()
    p.determineDistortion(debug=True)
    p.storeProcessedArc()
    p.writeOutputs()


_default = makeProcessedArc