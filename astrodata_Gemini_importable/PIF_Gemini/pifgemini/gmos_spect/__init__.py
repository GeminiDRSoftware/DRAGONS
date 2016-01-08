#test
from recipe_system.reduction.mkro import *


def extract_1d_spectra(*args, **argv):
    ro = mkRO(astrotype="GMOS_SPECT", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("extract1DSpectra", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def make_flat(*args, **argv):
    ro = mkRO(astrotype="GMOS_SPECT", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("makeFlat", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def determine_wavelength_solution(*args, **argv):
    ro = mkRO(astrotype="GMOS_SPECT", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("determineWavelengthSolution", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def sky_correct_from_slit(*args, **argv):
    ro = mkRO(astrotype="GMOS_SPECT", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("skyCorrectFromSlit", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def attach_wavelength_solution(*args, **argv):
    ro = mkRO(astrotype="GMOS_SPECT", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("attachWavelengthSolution", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def resample_to_linear_coords(*args, **argv):
    ro = mkRO(astrotype="GMOS_SPECT", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("resampleToLinearCoords", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def reject_cosmic_rays(*args, **argv):
    ro = mkRO(astrotype="GMOS_SPECT", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("rejectCosmicRays", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    