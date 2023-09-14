makeProcessedFlat
=================

| **Recipe Library**: geminidr.gmos.recipes.sq.recipes_FLAT_IMAGE
| **Astrodata Tags**: {'GMOS', 'FLAT', 'CAL', 'IMAGE'}

This recipe performs the standardization and corrections needed to
convert the raw input flat images into a single stacked and normalized
flat image.  This output processed flat is stored on disk using
storeProcessedFlat and has a name equal to the name of the first input
flat image with "_flat.fits" appended.

::

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.

::

    def makeProcessedFlat(p):
        p.prepare()
        p.addDQ()
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.biasCorrect()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        p.stackFlats()
        p.normalizeFlat()
        p.makeIRAFCompatible()
        p.storeProcessedFlat()
        return

