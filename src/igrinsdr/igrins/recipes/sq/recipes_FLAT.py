"""
Recipes available to data with tags ['IGRINS', 'CAL', 'FLAT'].
"""
from igrinsdr.igrins.primitives_igrins import Igrins

recipe_tags = {'IGRINS', 'FLAT'}

def estimateNoise(p: Igrins):
    """This recipe performs the analysis of irs readout pattern noise in flat off
    images. It creates a stacked image of pattern removed images and add a
    table that descibes its noise characteristics. The result is stored on disk
    and has a name equal to the name of the first input image with
    "_pattern_noise.fits" appended.

    Parameters
    ----------
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.

    """

    # Given the list of adinputs of both flat on and off images, we first
    # select the only the off images.
    p.selectFrame(frmtype="OFF")
    p.prepare()
    # it creates pattern corrected images with several methods (guard, level2,
    # level3). The images are then added to the streams.
    p.streamPatternCorrected(rpc_mode="full")
    # Estimate some noise characteristics of images in each stream. A table is
    # created and added to a 'ESTIMATED_NOISE' stream.
    p.estimateNoise()
    # Select the "level3_removed" stream and make it the output (i.e., input of
    # next primitive)
    p.selectStream(stream_name="LEVEL3_REMOVED")
    p.stackFlats()
    # The table from 'ESTIMATED_NOISE' stream is appended to the stacked image.
    p.addNoiseTable()
    # Set the suffix.
    p.setSuffix(suffix="_pattern_noise")
    return

def makeProcessedFlat(p: Igrins):
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
    p.addDQ()
    p.addVAR(read_noise=True)
    # ADUToElectrons requires saturation_level and nonlinearity_level in the
    # header. Since IGRINS does not have these values defined, we add them
    # here.
    p.fixIgrinsHeader()
    p.referencePixelsCorrect()

    p.ADUToElectrons()
    #p.nonlinearityCorrect()
    p.makeLampFlat() # This separates the lamp-on and lamp-off flats, stacks
                     # them, subtracts one from the other, and returns that
                     # single frame. It requires LAMPON/LAMPOFF tags.

    # # ported IGRINS's version of slit edge detection
    p.determineSlitEdges()
    # # version that support multiple aperture.
    p.maskBeyondSlit()
    # # very primitive implementation for multiple apertures

    # FIXME : maybe incorporate PLP version of algorithm.
    p.normalizeFlat()
    # We are using dragons's version of thresholdFlatfield. Do we need to mask
    # out low value pixels from the un-normarlized flat too? This will set DQ
    # with DQ.unilluminated for pixels whose value outsied the range.

    p.thresholdFlatfield()
    p.storeProcessedFlat()

    return

_default = makeProcessedFlat

# We set 'estimateNoise' as a default recipe for temporary, just for testing
# purpose.
# _default = estimateNoise


def makeProcessedBPM(p: Igrins):
    """
    This recipe requires flats and uses the lamp-off as short darks.
    """

    p.prepare(require_wcs=False)
    p.addDQ()
    p.fixIgrinsHeader()
    p.referencePixelsCorrect()
    p.ADUToElectrons()

    p.selectFromInputs(tags="LAMPOFF", outstream="darks")
    p.stackFrames(stream="darks")

    p.make_hotpix_mask(sigma_clip1 = 100., sigma_clip2 = 10.)
    # It will use "darks" stream to make a hotpix_maxk.

    p.storeBPM()
    return

# _default = makeProcessedBPM
