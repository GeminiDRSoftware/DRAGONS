# Test recipe demo working with the new package structure. 
# 'p' is an instance of a usuable GeminiDR package. Currently,
# only one package is available for demonstration purposes:
# GMOS.

# The demo primitives merely display values available to them, 
# such as file inputs, parameters. Some, like ADUToElectrons,
# do perform their full task here in this demonstration package.
# 17-06-2016 kra

def test_recipe_gmos(p):
    """
    Demo recipe.

    Usage:

    >>> from GMOS.primitives.primitives_GMOS import PrimitivesGMOS
    >>> from astrodata import AstroData
    >>> ad = AstroData('<filename.fits>')
    >>> gmos = PrimitivesGMOS(ad)
    >>> test_recipe_gmos(gmos)

    :parameter p: instance of primitives_GMOS
    :type      p: <instance>, GMOS.primitives.primitives_GMOS.PrimitivesGMOS

    """
    p.prepare()
    p.addDQ()
    p.addVAR()
    p.ADUToElectrons()
    p.getProcessedBias()
    p.storeProcessedBias()
    p.correctWCSToReferenceFrame()
    p.display()
    return
