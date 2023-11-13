reduceSeparateCCDsCentral
=========================

| **Recipe Library**: geminidr.gmos.recipes.sq.recipes_IMAGE
| **Astrodata Tags**: {'GMOS', 'IMAGE'}

This recipe performs the standardization and corrections needed to
convert the raw input science images into a stacked image. To deal
with different color terms on the different CCDs, the images are
split by CCD midway through the recipe and subsequently reduced
separately. The relative WCS is determined from the central CCD
(CCD2) and then applied to CCDs 1 and 3.

::

    Parameters
    ----------
    p : GMOSImage object
        A primitive set matching the recipe_tags.

::

    def reduceSeparateCCDsCentral(p):
        p.prepare()
        p.addDQ()
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.biasCorrect()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        p.flatCorrect()
        p.fringeCorrect()
        p.tileArrays(tile_all=False)
        p.sliceIntoStreams(root_stream_name="ccd")
        p.detectSources(stream="ccd2")
        p.adjustWCSToReference(stream="ccd2")
        p.applyWCSAdjustment(stream="main", reference_stream="ccd2")
        p.clearAllStreams()
        p.detectSources()
        p.scaleCountsToReference()
        p.sliceIntoStreams(root_stream_name="ccd")
        for ccd in (1, 2, 3):
            p.resampleToCommonFrame(instream=f"ccd{ccd}")
            p.detectSources()
            p.flagCosmicRaysByStacking()
            p.stackFrames(zero=True)
            p.appendStream(stream="all", from_stream="main", copy=False)
        p.mergeInputs(instream="all")
        p.storeProcessedScience(suffix="_image")

