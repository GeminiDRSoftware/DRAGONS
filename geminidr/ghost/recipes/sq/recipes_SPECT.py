"""
Recipes available to data with tags ``['GHOST', 'SPECT']``.
Default is ``reduce``. SQ is identical to QA recipe, and is imported
from there.
"""
recipe_tags = set(['GHOST', 'SPECT'])

def reduceScience(p):
    """
    This recipe processes GHOST science data.

    Parameters
    ----------
    p : Primitives object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.darkCorrect()
    p.tileArrays()
    p.applyFlatBPM() # Bitwise combine the flat BPM with the current BPM
                     # for the data. This is necessary because the flat isn't
                     # subtracted in the classical sense - rather, it's profile
                     # is subtracted from the object profile. Therefore, we
                     # must apply the BPM of the flat to the object file
                     # separately, before we extract its profile.
    p.removeScatteredLight()
    p.writeOutputs()
    p.extractProfile()
    #p.flatCorrect(skip=True)  # Need to write our own, NOT USE GMOS - extract the flat
    #                 # profile, then simple division
    p.addWavelengthSolution()  # should be able to accept multiple input
                               # arcs, e.g. from start and end of night,
                               # and interpolate in time
    p.barycentricCorrect()  # trivial - multiply wavelength scale
    p.responseCorrect()
    p.writeOutputs(suffix="_calibrated", strip=True)  # output this data product
    p.interpolateAndCombine()
    p.standardizeSpectralFormat()


def reduceStandard(p):
    """
    This recipe processes GHOST telluric standard data.

    Parameters
    ----------
    p : Primitives object
        A primitive set matching the recipe_tags.
    """

    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.darkCorrect()
    p.tileArrays()
    #p.rejectCosmicRays()
    p.applyFlatBPM() # Bitwise combine the flat BPM with the current BPM
                     # for the data. This is necessary because the flat isn't
                     # subtracted in the classical sense - rather, it's profile
                     # is subtracted from the object profile. Therefore, we
                     # must apply the BPM of the flat to the object file
                     # separately, before we extract its profile.
    p.writeOutputs()
    p.extractProfile()
    #p.flatCorrect() # Need to write our own, NOT USE GMOS - extract the flat profile,
    #                # then simple division
    p.addWavelengthSolution(suffix="_standard")


_default = reduceScience
