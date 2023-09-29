makeProcessedBias
=================

| **Recipe Library**: geminidr.gmos.recipes.sq.recipes_BIAS
| **Astrodata Tags**: {'GMOS', 'BIAS', 'CAL'}

This recipe performs the standardization and corrections needed to convert
the raw input bias images into a single stacked bias image. This output
processed bias is stored on disk using storeProcessedBias and has a name
equal to the name of the first input bias image with "_bias.fits" appended.

::

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.

::

    def makeProcessedBias(p):

        p.prepare()
        p.addDQ(add_illum_mask=False)
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.stackBiases()
        p.makeIRAFCompatible()
        p.storeProcessedBias()
        return

