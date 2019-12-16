#!/usr/bin/env python
#
#                                                                        DRAGONS
#                                                                  gempy.scripts
#                                                                    showpars.py
# ------------------------------------------------------------------------------
from __future__ import print_function

import sys
from argparse import ArgumentParser

from importlib import import_module

import astrodata
import gemini_instruments

from gempy import __version__

from recipe_system.mappers import primitiveMapper
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def buildArgs():
    parser = ArgumentParser(
        description="Primitive parameter display, v{}".format(__version__))
    parser.add_argument(
        "-v", "--version", action="version", version="v{}".format(__version__))
    parser.add_argument('filen', nargs=1, help="filename")
    parser.add_argument('primn', nargs=1, help="primitive name")
    parser.add_argument('--adpkg', nargs=1,
                        dest='adpkg', action='store', required=False,
                        help='Name of the astrodata instrument package to use'
                             'if not gemini_instruments')
    parser.add_argument('--drpkg', nargs=1,
                        dest='drpkg', action='store', required=False,
                        help='Name of the DRAGONS instrument package to use'
                             'if not geminidr')

    args = parser.parse_args()

    if args.adpkg is not None:
        args.adpkg = args.adpkg[0]
    if args.drpkg is not None:
        args.drpkg = args.drpkg[0]

    return args

def get_pars(filename, adpkg=None, drpkg=None):
    if adpkg is not None:
        import_module(adpkg)

    ad = astrodata.open(filename)
    if drpkg is None:
        pm = primitiveMapper.PrimitiveMapper([ad])
    else:
        pm = primitiveMapper.PrimitiveMapper([ad], drpkg=drpkg)
    p = pm.get_applicable_primitives()
    return p.params, ad.tags

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
    for k, v in pars.items():
        if not k.startswith("debug"):
            print("{:20s} {:20s} {}".format(k, repr(v), pars.doc(k)))
    print()
    return

if __name__ == '__main__':
    args = buildArgs()
    fname = args.filen[0]
    pname = args.primn[0]
    paro, tags = get_pars(fname, adpkg=args.adpkg, drpkg=args.drpkg)
    sys.exit(showpars(paro, pname, tags))
