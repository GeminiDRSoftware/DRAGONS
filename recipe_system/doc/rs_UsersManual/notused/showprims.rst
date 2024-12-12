
showprims
=========

The showprims_ command line displays what are the Primitives that
were used within a particular Recipe:::

    $ showprims raw/S20170505S0073.fits --mode sq --recipe makeProcessedBPM

    DRAGONS v2.1.x - show_recipes
    Input file: ./raw/S20170505S0073.fits
    Input mode: sq
    Input recipe: makeProcessedBPM
    Matched recipe: geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::makeProcessedBPM
    Primitives used:
      p.prepare()
      p.addDQ()
      p.addVAR(read_noise=True, poisson_noise=True)
      p.ADUToElectrons()
      p.selectFromInputs(tags="DARK", outstream="darks")
      p.selectFromInputs(tags="FLAT")
      p.stackFrames(stream="darks")
      p.makeLampFlat()
      p.normalizeFlat()
      p.makeBPM()

As the other commands, you can use the ``--help`` or ``-h`` flags in the command
line to display the help message.
