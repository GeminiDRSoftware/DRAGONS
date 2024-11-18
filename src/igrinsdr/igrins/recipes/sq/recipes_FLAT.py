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
    This recipe takes flat images and reduce them to prepare a processed flat image.
    The raw input should have both flat on and off images. The flat off and images are
    separatedly combined into a single stacked flat off and a stacked flat on image,
    then subtracted. For each order, an average spectrum is estimated as
    a function of columns (x-pixels) and the pixels belong in that order are nomarlized
    by the average spectrum. This is saved as a processed flat. The recipe will
    identify upp and lower boundary of each order and which is added to the processed flat
    with an extention of a "SLITEDGE" as a table. The combined flat before the normalization
    is also stored with an extension name of "FLAT_ORIGINAL".

    Parameters
    ----------
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.
    """

    p.prepare()

    p.referencePixelsCorrect() # FIXME For now, this does nothing as the
                               # reference pixel correction is not correctly
                               # done for IGRINS2 data dues to issues with
                               # detector tuning (a workaround is needed.) This
                               # recipe needs to be applied before addVar as we will
                               # add poisson_noise.

    p.addDQ() # FIXME : will use non_linear_level and saturation_level for
              # additional masking.
    p.addVAR(read_noise=True, poisson_noise=True) # readout noise from header

    # ADUToElectrons requires saturation_level and nonlinearity_level in the
    # header. Since IGRINS does not have these values defined, we add them
    # here.
    p.fixIgrinsHeader() # FIXME descriptor needed for saturation_level and nonlinear_level.
    p.ADUToElectrons()
    #p.nonlinearityCorrect()
    p.makeLampFlat() # This separates the lamp-on and lamp-off flats, stacks
                     # them, subtracts one from the other, and returns that
                     # single frame. It requires LAMPON/LAMPOFF tags.

    p.determineSlitEdges()
    # ported IGRINS's version of slit edge detection.
    # Will create SLITEDGE table.

    p.maskBeyondSlit()
    # set unilluminated flags for the pixel not illuminated by the slit.

    p.normalizeFlat()
    # The primitive will store the original flat in as 'FLAT_ORIGINAL'

    # We are using dragons's version of thresholdFlatfield. Do we need to mask
    # out low value pixels from the un-normarlized flat too? This will set DQ
    # with DQ.unilluminated for pixels whose value outsied the range.
    # FIXME : maybe we incorporate PLP version of algorithm.

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
