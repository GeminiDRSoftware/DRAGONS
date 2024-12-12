makeIRAFCompatible
==================

| **Recipe Library**: geminidr.gmos.recipes.sq.recipes_LS_SPECT
| **Recipe Imported From**: geminidr.gmos.recipes.sq.recipes_common
| **Astrodata Tags**: {'LS', 'GMOS', 'SPECT'}

Add header keywords needed to run some Gemini IRAF tasks.  This is needed
only if the reduced file will be used as input to Gemini IRAF tasks.

::

    Parameters
    ----------
    p : PrimitivesBASEE object
        A primitive set matching the recipe_tags.

::

    def makeIRAFCompatible(p):

        p.makeIRAFCompatible()
        p.writeOutputs()
        return
