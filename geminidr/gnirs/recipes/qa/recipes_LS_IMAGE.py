"""
Recipes available to data with tags ['GNIRS', 'IMAGE', 'LS].
Default is "reduce".
"""
recipe_tags = {'GNIRS', 'IMAGE', 'LS'}

def reduce(p):
    """
    This recipe performs the standardization and corrections needed to
    process a raw through-slit science image in order to obtain an IQ measurement.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """

    p.prepare(attach_mdf=False)
    p.addDQ(add_illum_mask=False)
    p.ADUToElectrons()
    p.measureIQ(display=True)
    p.writeOutputs()
    return

_default = reduce
