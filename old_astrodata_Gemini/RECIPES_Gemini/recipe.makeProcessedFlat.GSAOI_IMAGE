# This recipe performs the standardization and corrections needed to convert 
# the raw input flat images into a single stacked and normalized flat image. 
# This output processed flat is stored on disk using storeProcessedFlat and 
# has a name equal to the name of the first input flat image with "_flat.fits" 
# appended. Since GSAOI Z- and J-band has only illuminated dome flats (the 
# dark current is not significant) and H-band and above have the dome-flat
# equivalent to lamp-on and lamp-off flats, the darkCorrectAndStackFlats
# primitive checks the waveband and either just stacks the flats (Z and J-bands)
# or calls the lampOnLampOff recipe, which stacks lamp-on and lamp-off frames
# separately and subtracts one from the other.

prepare
addDQ
ADUToElectrons
addVAR(read_noise=True, poisson_noise=True)
addToList(purpose=forFlat)
getList(purpose=forFlat)
darkCorrectAndStackFlats
normalizeFlat
thresholdFlatfield
storeProcessedFlat
