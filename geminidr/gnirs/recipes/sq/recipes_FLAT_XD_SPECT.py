"""
Recipes available to data with tags ['GNIRS', 'SPECT', 'XD', 'FLAT'].
These are GNIRS cross-dispersed (XD) observations.
Default is "makeProcessedFlat".
"""
recipe_tags = {'GNIRS', 'SPECT', 'XD', 'FLAT'}

def makeProcessedFlat(p):
    """
    Create a processed flat for GNIRS cross-dispersed data.
    Inputs are:
      * raw LAMPON flats (for [spectral] order 3)
      * raw LAMPOFF flats (confusingly, made with QH lamp on) (for orders 4-8)
    No darks are needed due to the short exposures.  It was found that using
    darks was just adding to the noise.

    GNIRS XD flats are taken with two different lamps (the IR lamp and the QH
    lamp). The IR lamp provides illumination to the first order (#3), while the
    remaining orders are handled by the QH lamp. Due to a deep absorption
    feature in the wavelength range of the first order the QH lamp isn't used
    for flatfielding it, but it provides enough illumination that it can be
    used for determining the edges of the slits (actually all the same slit,
    but split out across the detector). Thus, the edges are traced using the
    QH lamp stack of flats, then the individual slits are cut out in both stacks,
    and orders 4-8 merged into the file with order 3.
    """
    p.prepare()
    p.addDQ()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.selectFromInputs(tags='GCAL_IR_OFF,LAMPOFF', outstream='QHLamp')
    p.removeFromInputs(tags='GCAL_IR_OFF,LAMPOFF')
    p.stackFlats(stream='main')
    p.stackFlats(stream='QHLamp')
    p.writeOutputs(stream='QHLamp')
    # Illumination of all orders from QH lamp is sufficient to find edges.
    p.determineSlitEdges(stream='QHLamp', search_radius=30)
    # Transfer SLITEDGE table and slit rectification model (in WCS) back over.
    p.transferAttribute(stream='main', source='QHLamp', attribute='SLITEDGE')
    p.cutSlits(stream='main')
    p.cutSlits(stream='QHLamp')
    # Bring slts 2-6 from QHLamp stream to main (1-indexed).
    p.combineSlices(from_stream='QHLamp', ids='2,3,4,5,6')
    p.clearStream(stream='QHLamp')
    p.maskBeyondSlit()
    p.normalizeFlat()
    p.thresholdFlatfield()
    p.storeProcessedFlat()

_default = makeProcessedFlat
