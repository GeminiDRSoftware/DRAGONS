"""
Recipes available to data with tags ['GNIRS', 'SPECT', 'LS', 'FLAT'].
These are GNIRS longslit observations.
Default is "makeProcessedFlat".
"""
recipe_tags = {'GNIRS', 'SPECT', 'LS', 'FLAT'}

def makeProcessedFlat(p):
    """
    Create a processed flat for GNIRS longslit data.
    Inputs are:
      * raw LAMPON flats - no other calibrations required.
      (Questions remaining, see google doc)
    No darks are needed due to the short exposures.  It was found that using
    darks was just adding to the noise.
    """
    p.prepare()
    p.addDQ()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.stackFlats()
    p.determineSlitEdges()
    p.maskBeyondSlit()
    p.normalizeFlat()
    p.thresholdFlatfield()
    p.makeIRAFCompatible()
    p.storeProcessedFlat()

_default = makeProcessedFlat

def makeProcessedBPM(p):

    p.prepare(require_wcs=False, bad_wcs="ignore")
    p.addDQ(static_bpm=None, add_illum_mask=False)
    p.ADUToElectrons()
    p.selectFromInputs(tags="DARK", outstream="darks")
    p.selectFromInputs(tags="FLAT")
    p.stackFrames(stream="darks")
    p.stackFlats()
    p.determineSlitEdges()
    p.maskBeyondSlit()
    p.writeOutputs()
    p.normalizeFlat()
    p.makeBPM(dark_lo_thresh=-100, dark_hi_thresh=150, flat_lo_thresh=0.25, flat_hi_thresh=1.25)
