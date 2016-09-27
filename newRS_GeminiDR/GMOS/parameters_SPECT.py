# This parameter file contains the parameters related to the primitives located
# in the primitives_SPECT.py file, in alphabetical order.

from parameters_GMOS import ParametersGMOS

class ParametersSPECT(ParametersGMOS):

    # Currently (10-06-2016) there are no spectroscopic BPMs for GMOS,
    # so turn off the BPM argument in addDQ for now.
    # No default type defined, since the bpm parameter could be a string or
    # an AstroData object
    addDQ ={
        "suffix"            : "_dqAdded",
        "bpm"               : None,
    }
    determineWavelengthSolution = {
        "suffix"            : "_wavelengthSolution",
        "interactive"       : False,
    }
    extract1DSpectra = {
        "suffix"            : "_extracted",
    }
    findAcquisitionSlits = {
        "suffix"            : "_acqSlitsAdded",
    }
    makeFlat ={
        "suffix"            : "_flat"
    }
    rejectCosmicRays = {
        "suffix"            : "_crRejected",
    }
    resampleToLinearCoords = {
        "suffix"            : "_linearCoords",
    }
    skyCorrectFromSlit = {
        "suffix"            : "_skyCorrected",
    }
    skyCorrectNodShuffle = {
        "suffix"            : "_skyCorrected",
    }
