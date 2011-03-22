averageCombine
================

 ::

  # This mini-recipe will retrieve the stack for the current input and then 
  # produce a single average combination of all the files in the stack.
  # The output will be a single file with the name of the first file in the stack
  # with the postfix '_avgcomb.fits'.

 ::

  #showInputs
  getStackable
  #showInputs
  combine(method="average", suffix="_avgcomb")
