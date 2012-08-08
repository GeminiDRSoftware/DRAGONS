import sys, os
from copy import deepcopy

from pyraf import iraf
from astrodata import AstroData

from astrodata.RecipeManager import RecipeLibrary, ReductionContext
from astrodata import ReductionObjects # to stuff module level, should use func
from astrodata.ReductionObjects import command_clause
from astrodata.adutils import gemLog
from astrodata import Proxies


# .par file
# name,type,mode,default,min,max,prompt

from astrodata.adutils import logutils

def mkRO(dataset="", astrotype="", copy_input=False, args = None, argv = None):

    log = logutils.get_logger(__name__)
    
    rl = RecipeLibrary()
    if dataset != "":
        ad = AstroData(dataset)
        ro = rl.retrieve_reduction_object(ad)
    elif astrotype != "":
        ad = None
        ro = rl.retrieve_reduction_object(astrotype = astrotype)

    # using standard command clause supplied in RecipeLibrary module
    ro.register_command_clause(command_clause)
    rc = ReductionContext(adcc_mode="start_lazy")
    rc.ro = ro
    ro.context = rc
    reductionObject = ro
    
    # Override copy_input argument if passed in argv
    if argv is not None:
        if argv.has_key("copy_input"):
            copy_input = argv["copy_input"]

    # Add input passed in args
    if args:
        arglist = []
        for arg in args:
            if isinstance(arg,list):
                for subarg in arg:
                    if copy_input:
                        subarg = deepcopy(subarg)
                    arglist.append(subarg)
            else:
                if copy_input:
                    arg = deepcopy(arg)
                arglist.append(arg)
        rc.populate_stream(arglist)

    rc.initialize_inputs()
    rc.set_context("pif")
    rc.update({'logindent':logutils.SW})
    rc.update(argv)
    ro.init(rc)
    
    return reductionObject

