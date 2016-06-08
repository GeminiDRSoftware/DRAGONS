# This recipe performs the standardization and corrections needed to convert 
# the raw input bias images into a single stacked bias image. This output 
# processed bias is stored on disk using storeProcessedBias and has a name 
# equal to the name of the first input bias image with "_bias.fits" appended.

prepare
addDQ
addVAR(read_noise=True)
overscanCorrect
addToList(purpose="forStack")
getList(purpose="forStack")
stackFrames
storeProcessedBias
