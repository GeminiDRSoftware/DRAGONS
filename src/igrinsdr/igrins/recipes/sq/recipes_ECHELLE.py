"""
Recipes available to data with tags ['IGRINS', 'ECHELLE']
Default is "reduce".
"""

recipe_tags = {'IGRINS', 'ECHELLE'}

def reduce(p):
    """
    This recipe processes IGRINS echelle science data.

    Parameters
    ----------
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    #....
    #....
    return

_default = reduce
