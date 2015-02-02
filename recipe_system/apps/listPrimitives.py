#!/usr/bin/env python
#
#                                                                     QAP Gemini
#
#                                                              listPrimitives.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
import os
import sys
from optparse import OptionParser

from recipe_system.reduction.priminspect import PrimInspect
# ------------------------------------------------------------------------------
# set up commandline args and options
def handleClArgs():
    parser = OptionParser()
    parser.set_description( "Gemini Observatory Primitive Inspection Tool, "
                            "v1.0 2011")
    parser.add_option("-c", "--use-color", action="store_true", dest="use_color",
                      default=False, help="apply color output scheme")
    parser.add_option("-i", "--info", action="store_true", dest="info",
                      default=False, help="show more information")
    parser.add_option("-p", "--parameters", action="store_true", dest="parameters",
                      default=False, help="show parameters")
    parser.add_option("-r", "--recipes", action="store_true", dest="recipes",
                      default=False, help="list top recipes")
    parser.add_option("-s", "--primitive-set", action="store_true", 
                      dest="primitive_set", default=False,
                      help="show primitive sets (Astrodata types)")
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose",
                      default=False, help="set verbose mode")
    parser.add_option("--view-recipe", dest="view_recipe", 
                      default=None, help="display the recipe")
    options, args = parser.parse_args()

    return options, args

# ------------------------------------------------------------------------------
def run(options, args):
    opti = options.info
    optp = options.parameters
    oset = options.primitive_set
    oview = options.view_recipe
    if options.verbose:
        optp = True
        opti = True

    # parse arguments
    datasets = []
    adtypes  = []

    for arg in args:
        if os.path.exists(arg) and not os.path.isdir(arg):
            datasets.append(arg)
        else:
            adtypes.append(arg.upper())

    pin = PrimInspect(use_color=options.use_color)

    if options.recipes or oview:                            # Show Recipes
        pin.list_recipes(pkg="Gemini", view=oview)
    elif oset:                                              # OR Primitives
        pin.list_primsets(info=opti)
    else: 
        if datasets:
            for data in datasets:
                pin.list_primitives(data=data, info=opti, params=optp)
        else:
            if len(adtypes) == 0:
                adtype = None
                pin.list_primitives(adtype=adtype, info=opti, params=optp)
            else:
                for adt in adtypes:
                    pin.list_primitives(adtype=adt, info=opti, params=optp)
    return
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    opts, args = handleClArgs()
    sys.exit(run(opts, args))

