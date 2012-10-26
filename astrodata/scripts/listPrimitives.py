#!/usr/bin/env python
import os
import sys
from astrodata import jsutil
from optparse import OptionParser
from astrodata import AstroDataType
from astrodata.priminspect import PrimInspect
import pprint
from astrodata.ConfigSpace import cs
import json

# set up commandline args and options
parser = OptionParser()
parser.set_description( "Gemini Observatory Primitive Inspection Tool "
                       "(v_1.0 2011)")
parser.add_option("-c", "--use-color", action="store_true", dest="use_color",
                  default=False, help="apply color output scheme")
parser.add_option("-e", "--engineering", action="store_true", dest="engineering",
                  default=False, help="show engineering recipes")
parser.add_option("-i", "--info", action="store_true", dest="info",
                  default=False,
                  help="show more information")
parser.add_option("-p", "--parameters", action="store_true", dest="parameters",
                  default=False,
                  help="show parameters")
parser.add_option("-r", "--recipes", action="store_true", dest="recipes",
                  default=False, help="list top recipes")
parser.add_option("-s", "--primitive-set", action="store_true", 
                  dest="primitive_set", default=False,
                  help="show primitive sets (Astrodata types)")
parser.add_option("-u", "--usage", action="store_true", dest="usage",
                  default=False, help="show usage")
parser.add_option("-v", "--verbose", action="store_true", dest="verbose",
                  default=False, help="set verbose mode")
parser.add_option("--view-recipe", dest="view_recipe", 
                  default=None, help="display the recipe")
(options,  args) = parser.parse_args()
opte = options.engineering
opti = options.info
optp = options.parameters
optu = options.usage
oview = options.view_recipe
oset = options.primitive_set
# server = options.server
if options.verbose:
    optp = True
    opti = True

# parse arguments
datasets = []
adtypes = []
for arg in args:
    if os.path.exists(arg) and not os.path.isdir(arg):
        datasets.append(arg)
    else:
        adtypes.append(arg.upper())
pin = PrimInspect(use_color=options.use_color)

# show recipes
if options.recipes or oview:
    pin.list_recipes(pkg="Gemini",eng=opte, view=oview)
# or show primitives
elif oset:
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

