makeProcessedBias
====================

 ::


  # This recipe will perform the preparation and corrections
  # needed to convert the inputs into a single averaged bias calibration file.  
  # Output is stored in a directory specified
  # by the parameters of the storeProcessedBias primitive 
  # with the postfix '_preparedbias.fits' to the name of the first input file.

 ::

  prepare

  overscanCorrect

  addVARDQ

  #showInputs
  setStackable

  averageCombine

  storeProcessedBias(clob=True)
