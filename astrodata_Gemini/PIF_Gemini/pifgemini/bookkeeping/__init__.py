#test
from recipe_system.reduction.mkro import *


def write_outputs(*args, **argv):
    ro = mkRO(astrotype="GEMINI", copy_input=False, 
              args=args, argv=argv)
    ro.runstep("writeOutputs", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def show_inputs(*args, **argv):
    ro = mkRO(astrotype="GEMINI", copy_input=False, 
              args=args, argv=argv)
    ro.runstep("showInputs", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    