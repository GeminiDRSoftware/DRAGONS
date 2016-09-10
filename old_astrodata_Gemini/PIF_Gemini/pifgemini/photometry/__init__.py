#test
from recipe_system.reduction.mkro import *


def detect_sources(*args, **argv):
    ro = mkRO(astrotype="GEMINI", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("detectSources", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def add_reference_catalog(*args, **argv):
    ro = mkRO(astrotype="GEMINI", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("addReferenceCatalog", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def measure_cc_and_astrometry(*args, **argv):
    ro = mkRO(astrotype="GEMINI", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("measureCCAndAstrometry", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    