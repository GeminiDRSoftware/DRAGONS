recipe_tags = set(['GMOS', 'IMAGE', 'CAL', 'FLAT'])

def makeProcessedFlat(p):
    prepare
    addDQ
    addVAR(read_noise=True)
    display
    overscanCorrect
    biasCorrect
    ADUToElectrons
    addVAR(poisson_noise=True)
    stackFlats
    normalizeFlat
    storeProcessedFlat
    return
