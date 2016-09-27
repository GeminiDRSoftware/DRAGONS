
recipe_tags = set(['GMOS', 'CAL', 'BIAS'])

def makeProcessedBias(p):
    prepare
    addDQ
    addVAR(read_noise=True)
    overscanCorrect
    stackFrames
    storeProcessedBias
    return
