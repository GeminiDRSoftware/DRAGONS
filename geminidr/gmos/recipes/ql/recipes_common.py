def makeIRAFCompatible(p):
    """
    Add header keywords needed to run some Gemini IRAF tasks.  This is needed
    only if the reduced file will be used as input to Gemini IRAF tasks.

    Parameters
    ----------
    p : PrimitivesBASEE object
        A primitive set matching the recipe_tags.
    """

    p.makeIRAFCompatible()
    p.writeOutputs()
    return
