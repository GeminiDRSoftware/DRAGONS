"""
Recipes available to data with tags ['GMOS', 'IMAGE', 'LS].
Default is "reduce".
"""
recipe_tags = {'GMOS', 'IMAGE', 'LS'}

def reduce(p):
    """
    This recipe performs the standardization and corrections needed to
    process a raw through-slit science image in order to obtain an IQ measurement.

    Parameters
    ----------
    p : PrimitivesBASE object
        A primitive set matching the recipe_tags.
    """

    p.prepare(attach_mdf=True)
    p.addDQ()
    p.maskFaultyAmp(instrument='GMOS-S', bad_amps=5, valid_from='20220128')
    #p.addIllumMaskToDQ()
    p.subtractOverscan()
    p.ADUToElectrons()
    p.measureIQ(display=True)
    p.writeOutputs()
    return

_default = reduce
