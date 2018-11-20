"""
Recipes available to data with tags ['GMOS', 'SPECT', 'LS'].
Default is "reduce".
"""
recipe_tags = set(['GMOS', 'SPECT', 'LS'])

def reduce(p):
    p.prepare()
    p.addDQ(static_bpm=None)
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    #p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.mosaicDetectors()
    p.determineWavelengthSolution()

default = reduce
