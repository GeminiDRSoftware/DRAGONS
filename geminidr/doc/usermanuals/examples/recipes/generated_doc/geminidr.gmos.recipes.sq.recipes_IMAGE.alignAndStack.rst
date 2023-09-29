alignAndStack
=============

| **Recipe Library**: geminidr.gmos.recipes.sq.recipes_IMAGE
| **Astrodata Tags**: {'GMOS', 'IMAGE'}

This recipe stack already preprocessed data.

::

    Parameters
    ----------
    p : PrimitivesBASEE object
        A primitive set matching the recipe_tags.

::

    def alignAndStack(p):

        p.detectSources()
        p.adjustWCSToReference()
        p.resampleToCommonFrame()
        p.scaleCountsToReference()
        p.stackFrames(zero=True)
        return

