"""
Recipe for raw IGRINS bundles, to split the into separate H and K files.
There's only one recipe.
"""
recipe_tags = {'IGRINS-2', 'BUNDLE', 'RAW', 'UNPREPARED'}

def processBundle(p):
    """
    This recipe processes IGRINS-2 observation bundles.

    Parameters
    ----------
    p : Primitives object
        A primitive set matching the recipe_tags.
    """
    p.splitBundle()
    return


_default = processBundle
