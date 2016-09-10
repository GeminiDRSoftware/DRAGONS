# GSAOI_IMAGE recipe
prepare
addDQ
nonlinearityCorrect
ADUToElectrons
addVAR(read_noise=True, poisson_noise=True)
flatCorrect
detectSources
measureIQ(display=True)
measureBG
measureCCAndAstrometry
addToList(purpose=forSky)
getList(purpose=forSky, max_frames=8)
makeSky
skyCorrect
detectSources
measureIQ(display=True)
measureCCAndAstrometry