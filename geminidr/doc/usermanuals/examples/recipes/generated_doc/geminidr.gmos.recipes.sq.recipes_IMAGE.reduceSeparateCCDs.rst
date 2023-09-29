reduceSeparateCCDs
==================

| **Recipe Library**: geminidr.gmos.recipes.sq.recipes_IMAGE
| **Astrodata Tags**: {'GMOS', 'IMAGE'}

This recipe performs the standardization and corrections needed to
convert the raw input science images into a stacked image. To deal
with different color terms on the different CCDs, the images are
split by CCD midway through the recipe and subsequently reduced
separately. The relative WCS is determined from mosaicked versions
of the images and then applied to each of the CCDs separately.

::

    Parameters
    ----------
    p : GMOSImage object
        A primitive set matching the recipe_tags.

::

    def reduceSeparateCCDs(p):
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
        p.mosaicDetectors(outstream="mosaic")
        p.detectSources(stream="mosaic")
        p.adjustWCSToReference(stream="mosaic")
        p.applyWCSAdjustment(reference_stream="mosaic")
        p.clearStream(stream="mosaic")
        p.detectSources()
        p.scaleCountsToReference()
        p.sliceIntoStreams(root_stream_name="ccd", copy=False)
        p.clearStream(stream="main")
        for ccd in (1, 2, 3):
            p.resampleToCommonFrame(instream=f"ccd{ccd}")
            p.detectSources()
            p.flagCosmicRaysByStacking()
            p.stackFrames(zero=True)
            p.appendStream(stream="all", from_stream="main", copy=False)
        p.mergeInputs(instream="all")
        p.storeProcessedScience(suffix="_image")

