#test
from recipe_system.reduction.mkro import *


def display(*args, **argv):
    ro = mkRO(astrotype="GEMINI", copy_input=False, 
              args=args, argv=argv)
    ro.runstep("display", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    