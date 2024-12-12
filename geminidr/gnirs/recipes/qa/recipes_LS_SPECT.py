"""
Recipes available to data with tags ['GNIRS', 'SPECT', LS'].
Default is "reduceScience".
"""
recipe_tags = {'GNIRS', 'SPECT', 'LS'}


def reduceScience(p):
    """
    To be updated as development continues: This recipe processes GNIRS longslit
    spectroscopic data, currently up to basic spectral extraction without telluric correction.

    Parameters
    ----------
    p : :class:`geminidr.gnirs.primitives_gnirs_longslit.GNIRSLongslit`

    """
    p.prepare()
    p.addDQ()
    # p.nonlinearityCorrect() # non-linearity correction tbd
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True, read_noise=True)
    p.measureIQ(display=True)
    p.flatCorrect()
    p.attachWavelengthSolution()
    p.addToList(purpose='forSky')
    p.getList(purpose='forSky')
    p.separateSky()
    p.associateSky()
    p.skyCorrect()
    p.distortionCorrect()
    p.findApertures()
    p.measureIQ(display=True)

    # side stream to generate 1D spectra from individual frame, pre-stack
    p.traceApertures(outstream='prestack')
    p.extractSpectra(stream='prestack')
    p.plotSpectraForQA(stream='prestack')
    # The GUI polls for new data every 3 seconds.  The next steps can be
    # quicker than that leading to the second plotSpectra to hijack this one.
    # Hijacking issues were highlighted in integration tests.
    from time import sleep
    sleep(3)

    # continuing with the main stream of 2D pre-stack
    p.adjustWCSToReference()
    p.resampleToCommonFrame()
    p.stackFrames()
    p.findApertures()
    p.measureIQ(display=True)
    p.traceApertures()
    p.extractSpectra()
    p.plotSpectraForQA()


_default = reduceScience
