#test
from recipe_system.reduction.mkro import *


def divide(*args, **argv):
    ro = mkRO(astrotype="GEMINI", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("divide", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def add(*args, **argv):
    ro = mkRO(astrotype="GEMINI", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("add", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def multiply(*args, **argv):
    ro = mkRO(astrotype="GEMINI", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("multiply", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    
def subtract(*args, **argv):
    ro = mkRO(astrotype="GEMINI", copy_input=True, 
              args=args, argv=argv)
    ro.runstep("subtract", ro.context)
    outputs = ro.context.get_outputs(style="AD")
    if len(outputs)==0:
        return None
    elif len(outputs)==1:
        return outputs[0]
    else:
        return outputs
    
    