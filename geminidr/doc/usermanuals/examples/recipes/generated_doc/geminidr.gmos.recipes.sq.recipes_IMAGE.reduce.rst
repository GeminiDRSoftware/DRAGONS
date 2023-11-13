reduce
======

| **Recipe Library**: geminidr.gmos.recipes.sq.recipes_IMAGE
| **Astrodata Tags**: {'GMOS', 'IMAGE'}

This recipe performs the standardization and corrections needed to
convert the raw input science images into a stacked image.

::

    Parameters
    ----------
    p : GMOSImage object
        A primitive set matching the recipe_tags.

::

    def reduce(p):

        p.prepare()
        p.addDQ()
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.biasCorrect()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        p.flatCorrect()
        p.fringeCorrect()
        p.QECorrect()
        p.mosaicDetectors()
        p.detectSources()
        p.adjustWCSToReference()
        p.resampleToCommonFrame()
        p.flagCosmicRaysByStacking()
        p.scaleCountsToReference()
        p.stackFrames(zero=True)
        p.storeProcessedScience(suffix="_image")
        return

