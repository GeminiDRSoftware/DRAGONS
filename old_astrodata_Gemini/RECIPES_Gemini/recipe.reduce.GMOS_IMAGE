# This recipe performs the standardization and corrections needed to convert
# the raw input science images into a single stacked science image

prepare
addDQ
addVAR(read_noise=True)
overscanCorrect
biasCorrect
ADUToElectrons
addVAR(poisson_noise=True)
flatCorrect
mosaicDetectors
makeFringe
fringeCorrect
alignAndStack
