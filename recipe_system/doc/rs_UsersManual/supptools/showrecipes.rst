.. showrecipes.rst

.. _showrecipes:

showrecipes
===========
The Recipe System will select the best recipe for your data, which
can be overriden when necessary.  To see what sequence of primitives a
recipe will execute or which recipes are available for the dataset, one
can use ``showrecipes``.

Show Recipe Content
-------------------
To see the content of the best-matched default recipes::

    $ showrecipes S20170505S0073.fits

::

    Recipe not provided, default recipe (makeProcessedFlat) will be used.
    Input file: /path_to/S20170505S0073.fits
    Input tags: ['FLAT', 'LAMPOFF', 'AZEL_TARGET', 'IMAGE', 'DOMEFLAT',
    'GSAOI', 'RAW', 'GEMINI', 'NON_SIDEREAL', 'CAL', 'UNPREPARED', 'SOUTH']
    Input mode: sq
    Input recipe: makeProcessedFlat
    Matched recipe: geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::makeProcessedFlat
    Recipe location: /path_to/dragons/geminidr/gsaoi/recipes/sq/recipes_FLAT_IMAGE.py
    Recipe tags: set(['FLAT', 'IMAGE', 'GSAOI', 'CAL'])
    Primitives used:
       p.prepare()
       p.addDQ()
       p.nonlinearityCorrect()
       p.ADUToElectrons()
       p.addVAR(read_noise=True, poisson_noise=True)
       p.makeLampFlat()
       p.normalizeFlat()
       p.thresholdFlatfield()
       p.storeProcessedFlat()


To see the content of a specific recipe::

    $ showrecipes S20170505S0073.fits -r makeProcessedBPM

::

    Input file: /path_to/S20170505S0073.fits
    Input tags: ['FLAT', 'LAMPOFF', 'AZEL_TARGET', 'IMAGE', 'DOMEFLAT',
    'GSAOI', 'RAW', 'GEMINI', 'NON_SIDEREAL', 'CAL', 'UNPREPARED', 'SOUTH']
    Input mode: sq
    Input recipe: makeProcessedBPM
    Matched recipe: geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::makeProcessedBPM
    Recipe location: /path_to/dragons/geminidr/gsaoi/recipes/sq/recipes_FLAT_IMAGE.pyc
    Recipe tags: set(['FLAT', 'IMAGE', 'GSAOI', 'CAL'])
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



Show Index of Available Recipes
-------------------------------
Of course in order to ask for a specific recipe, it is useful to know
which recipes are available to the dataset.  To see the index of
available recipes::

    $ showrecipes S20170505S0073.fits --all

::

    Input file: /path_to/S20170505S0073.fits
    Input tags: set(['FLAT', 'LAMPOFF', 'AZEL_TARGET', 'IMAGE', 'DOMEFLAT',
    'GSAOI', 'RAW', 'GEMINI', 'NON_SIDEREAL', 'CAL', 'UNPREPARED', 'SOUTH'])
    Recipes available for the input file:
       geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::makeProcessedBPM
       geminidr.gsaoi.recipes.sq.recipes_FLAT_IMAGE::makeProcessedFlat
       geminidr.gsaoi.recipes.qa.recipes_FLAT_IMAGE::makeProcessedFlat

The output shows that there are two recipes for the SQ (Science Quality)
mode and one recipe for the QA (Quality Assesment) mode. By default,
the Recipe System uses the SQ mode for processing the data.

As for the other commands, you can use the ``--help`` or ``-h`` flags on
the command line to display the help message.
