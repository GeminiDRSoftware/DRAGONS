"""
Recipes available to data with tags ``['GHOST', `BUNDLE`]``.
Recipes imported from :any:`qa.recipes_BUNDLE`.
"""
recipe_tags = set(['GHOST', 'BUNDLE', 'RAW', 'UNPREPARED'])

def processBundle(p):
    """
    This recipe processes GHOST observation bundles.

    Parameters
    ----------
    p : Primitives object
        A primitive set matching the recipe_tags.
    """
    p.splitBundle()
    return


_default = processBundle
