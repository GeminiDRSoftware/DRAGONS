#!/usr/bin/env python
#
#                                                                        DRAGONS
#                                                                  gempy.scripts
#                                                                    showpars.py
# ------------------------------------------------------------------------------
from __future__ import print_function

import sys
from argparse import ArgumentParser

import astrodata
import gemini_instruments

from recipe_system.mappers import primitiveMapper
# ------------------------------------------------------------------------------
__version__ = '2.0.0 (beta)'
# ------------------------------------------------------------------------------
def buildArgs():
    parser = ArgumentParser(
        description="Primitive parameter display, v{}".format(__version__))
    parser.add_argument(
        "-v", "--version", action="version", version="v{}".format(__version__))
    parser.add_argument('filen', nargs=1, help="filename")
    parser.add_argument('primn', nargs=1, help="primitive name")
    args = parser.parse_args()
    return args

def get_pars(filename):
    ad = astrodata.open(filename)
    pm = primitiveMapper.PrimitiveMapper([ad])
    p = pm.get_applicable_primitives()
    return p.parameters, ad.tags

def showpars(pobj, primname, tags):
    for i in dir(pobj):
        if i.startswith("_"):
            continue

    pars = pobj[primname]
    print("Dataset tagged as {}".format(tags))
    print("Settable parameters on '{}':".format(primname))
    print("="*40)
    print(" Name\t\t\tCurrent setting")
    print()
    for k,v in pars.items():
        if len(k) <= 4:
            print("{} :\t\t\t{}".format(k,v))
        elif len(k) > 12:
            print("{} : \t{}".format(k,v))
        else:
            print("{} : \t\t{}".format(k,v))
    print()
    return

if __name__ == '__main__':
    args = buildArgs()
    fname = args.filen[0]
    pname = args.primn[0]
    paro, tags = get_pars(fname)
    sys.exit(showpars(paro, pname, tags))
