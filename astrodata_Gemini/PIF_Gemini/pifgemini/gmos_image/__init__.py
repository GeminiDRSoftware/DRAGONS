#test
from recipe_system.reduction.mkro import *


def make_fringe_frame(*args, **argv):
    ro = mkRO(astrotype="GMOS_IMAGE", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("makeFringeFrame", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def scale_fringe_to_science(*args, **argv):
    ro = mkRO(astrotype="GMOS_IMAGE", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("scaleFringeToScience", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def subtract_fringe(*args, **argv):
    ro = mkRO(astrotype="GMOS_IMAGE", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("subtractFringe", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def stack_flats(*args, **argv):
    ro = mkRO(astrotype="GMOS_IMAGE", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("stackFlats", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def scale_by_intensity(*args, **argv):
    ro = mkRO(astrotype="GMOS_IMAGE", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("scaleByIntensity", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    