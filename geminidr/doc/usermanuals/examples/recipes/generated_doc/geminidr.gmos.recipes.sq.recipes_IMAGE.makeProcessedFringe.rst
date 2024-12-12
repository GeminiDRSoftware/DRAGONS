makeProcessedFringe
===================

| **Recipe Library**: geminidr.gmos.recipes.sq.recipes_IMAGE
| **Astrodata Tags**: {'GMOS', 'IMAGE'}

This recipe creates a fringe frame from the inputs files. The output
is stored on disk using storeProcessedFringe and has a name equal
to the name of the first input bias image with "_fringe.fits" appended.

::

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.

::

    def makeProcessedFringe(p):
        p.prepare()
        p.addDQ()
        #p.addIllumMaskToDQ()
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.biasCorrect()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        p.flatCorrect()
        p.makeFringeFrame()
        p.storeProcessedFringe()
        return
