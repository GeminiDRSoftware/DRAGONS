"""
Recipes available to data with tags ['IGRINS', 'CAL', 'FLAT'].
Default is "makeProcessedDark"
"""

recipe_tags = {'IGRINS', 'CAL', 'FLAT'}

def estimateNoise(p):
    """
    """

    p.selectFrame(frmtype="OFF"),
    p.prepare()
    p.streamPatternCorrected(rpc_mode="full")
    p.estimateNoise()
    p.selectStream(stream_name="LEVEL3_REMOVED")
    p.stackDarks()
    p.addNoiseTable()
    p.setSuffix()
    return

def makeProcessedFlat(p):
    """
    This recipe performs the standardization and corrections needed to convert
    the raw input dark images into a single stacked dark image. This output
    processed dark is stored on disk and its information added to the calibration
    database using storeProcessedDark and has a name
    equal to the name of the first input bias image with "_bias.fits" appended.

    Parameters
    ----------
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    # p.makeIRAFCompatible()
    p.storeProcessedFlat()
    return

# _default = makeProcessedFlat

# This is for temporary for testing purpose.
_default = estimateNoise
