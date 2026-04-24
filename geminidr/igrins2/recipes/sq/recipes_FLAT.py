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

    p.readoutPatternCorrectFlatOff() # This primitive needs to be applied
                                     # before addVar as we will add
                                     # poisson_noise.

    p.addDQ() # FIXME : will use non_linear_level and saturation_level for
              # additional masking.
    p.addVAR(read_noise=True, poisson_noise=True) # readout noise from header

    # p.nonlinearityCorrect()
    p.ADUToElectrons() # It tries to use saturation_level and nonlinear_level
                       # values in the header. However, if those keys are not
                       # difined in the `__keyword_dict` of IGRINS adclass,
                       # they are simply ignored, which is the case.

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
    p.addDQ() # FIXME : While I don't think this is required, we still need
              # this. Otherwise, it will raise error in stackFrames.

    p.readoutPatternCorrectFlatOff() # This recipe needs to be applied before
                                     # addVar as we will add poisson_noise.
    p.readoutPatternCorrectFlatOn() # This recipe needs to be applied before

    p.selectFromInputs(tags="LAMPOFF", outstream="flat-off")
    p.stackFrames(stream="flat-off")

    p.selectFromInputs(tags="LAMPON", outstream="flat-on")
    p.stackFrames(stream="flat-on")

    p.makeIgrinsBPM() # hotpix mask is created from the flat-off stream and the
                      # deadpix mask is from flat-on stream.

    p.storeBPM()
    return


def makeTestBadpix(p: Igrins):

    p.prepare(require_wcs=False)
    p.readoutPatternCorrectFlatOff() # This primitive needs to be applied
                                     # before addVar as we will add
                                     # poisson_noise.

    p.addDQ()
    p.addVAR(read_noise=True, poisson_noise=True) # readout noise from header
    p.fixIgrinsHeader()
    p.ADUToElectrons()

    p.selectFromInputs(tags="LAMPOFF", outstream="flatoff")
    p.stackFrames(stream="flatoff")

    p.selectFromInputs(tags="LAMPON", outstream="flaton")
    p.stackFrames(stream="flaton")


    adlist = p.selectStream(stream_name="flatoff")
    adlist[0].write(overwrite=True)
    adlist = p.selectStream(stream_name="flaton")
    adlist[0].write(overwrite=True)

    return


# _default = makeProcessedBPM
