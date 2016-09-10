#test
from recipe_system.reduction.mkro import *


def correct_background_to_reference_image(*args, **argv):
    ro = mkRO(astrotype="GEMINI", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("correctBackgroundToReferenceImage", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def subtract_dark(*args, **argv):
    ro = mkRO(astrotype="GEMINI", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("subtractDark", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def divide_by_flat(*args, **argv):
    ro = mkRO(astrotype="GEMINI", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("divideByFlat", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def thermal_emission_correct(*args, **argv):
    ro = mkRO(astrotype="GEMINI", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("thermalEmissionCorrect", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def normalize(*args, **argv):
    ro = mkRO(astrotype="GEMINI", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("normalize", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def ADU_to_electrons(*args, **argv):
    ro = mkRO(astrotype="GEMINI", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("ADUToElectrons", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def nonlinearity_correct(*args, **argv):
    ro = mkRO(astrotype="GEMINI", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("nonlinearityCorrect", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
