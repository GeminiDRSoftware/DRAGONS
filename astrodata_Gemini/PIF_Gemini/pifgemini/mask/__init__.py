#test
from recipe_system.reduction.mkro import *


def apply_dq_plane(*args, **argv):
    ro = mkRO(astrotype="GEMINI", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("applyDQPlane", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def add_object_mask_to_dq(*args, **argv):
    ro = mkRO(astrotype="GEMINI", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("addObjectMaskToDQ", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    