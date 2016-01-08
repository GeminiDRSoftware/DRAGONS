#test
from recipe_system.reduction.mkro import *


def trim_overscan(*args, **argv):
    ro = mkRO(astrotype="GMOS", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("trimOverscan", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def mosaic_detectors(*args, **argv):
    ro = mkRO(astrotype="GMOS", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("mosaicDetectors", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def tile_arrays(*args, **argv):
    ro = mkRO(astrotype="GMOS", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("tileArrays", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def overscan_correct(*args, **argv):
    ro = mkRO(astrotype="GMOS", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("overscanCorrect", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def subtract_bias(*args, **argv):
    ro = mkRO(astrotype="GMOS", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("subtractBias", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def subtract_overscan(*args, **argv):
    ro = mkRO(astrotype="GMOS", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("subtractOverscan", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    